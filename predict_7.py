# predict_7.py
# --- ABS PATH SHIM ---
import sys

sys.path.insert(
    0,
    r"C:\Users\Asus\Desktop\Project Msc\Deepfake_detection_using_deep_learning\Django_Application",
)
# ----------------------
from ml_app.models_lstm import LSTMClassifier
import os, sys, glob, json, argparse, numpy as np, torch, csv
from collections import defaultdict

try:
    from Django_Application.ml_app.models_lstm import LSTMClassifier
except Exception:
    from ml_app.models_lstm import LSTMClassifier

DEFAULT_WEIGHTS = r"C:\Users\Asus\Desktop\Project Msc\Deepfake_detection_using_deep_learning\lstm_binary_weights.pth"
CALIB_JSON = r"C:\Users\Asus\Desktop\Project Msc\Deepfake_detection_using_deep_learning\best_threshold.json"
MAX_SEQ_LEN = 24
CLASSES = ["fake", "real"]  # fixed binary order


def safe_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_sequence(npy_path):
    seq = np.load(npy_path, allow_pickle=False).astype(np.float32)
    try:
        print(
            f"[feat] {os.path.basename(npy_path)} shape={seq.shape} "
            f"mean={seq.mean():.4f} std={seq.std():.4f}"
        )
    except Exception:
        pass
    T = seq.shape[0]
    if T >= MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    else:
        pad = np.zeros((MAX_SEQ_LEN - T, seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    return torch.tensor(seq).unsqueeze(0)


def load_thresholds(calib_json=CALIB_JSON):
    if not os.path.exists(calib_json):
        print(f"[WARN] {calib_json} not found; using thr=0.50, margin=0.00")
        return 0.5, 0.0
    with open(calib_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("thr", 0.5)), float(data.get("margin", 0.0))


def decide_strict(prob, classes, thr, margin, require_top1=False, delta=0.0):
    # classes must contain 'real' and 'fake'
    real_idx = classes.index("real")
    p_real = float(prob[real_idx])
    p_fake_any = (
        float(1.0 - p_real) if len(classes) == 2 else float(prob.sum() - p_real)
    )
    is_real = (p_real >= thr) and ((p_real - p_fake_any) >= margin)
    if require_top1:
        winner = int(np.argmax(prob))
        is_real = is_real and (winner == real_idx)
        if delta > 0.0 and winner == real_idx:
            runner_up = float(np.max(prob[np.arange(len(classes)) != real_idx]))
            is_real = is_real and ((p_real - runner_up) >= float(delta))
    return ("REAL" if is_real else "FAKE"), p_real, p_fake_any


@torch.no_grad()
def run_model_on_array(xb, model):
    logits = model(xb)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return prob


def classify_file(
    npy_path,
    model,
    classes,
    thr,
    margin,
    topk=2,
    device="cpu",
    quiet=False,
    require_top1=False,
    delta=0.0,
):
    xb = load_sequence(npy_path).to(device)
    prob = run_model_on_array(xb, model)
    order = np.argsort(-prob)
    top = [(classes[i], float(prob[i])) for i in order[: min(topk, len(classes))]]
    label, p_real, p_fake_any = decide_strict(
        prob, classes, thr, margin, require_top1=require_top1, delta=delta
    )
    if not quiet:
        print(f"\n=== {os.path.basename(npy_path)} ===")
        print(
            "Top-{}: {}".format(
                len(top), ", ".join([f"{n}={p*100:.2f}%" for n, p in top])
            )
        )
        print(
            f"Decision [strict] @thr={thr:.2f}, margin={margin:.2f}: {label} "
            f"(P(real)={p_real*100:.2f}%, P(fake)={p_fake_any*100:.2f}%)"
        )
    return {
        "file": npy_path,
        "label": label,
        "p_real": p_real,
        "p_fake_any": p_fake_any,
        **{f"p_{classes[i]}": float(prob[i]) for i in range(len(classes))},
    }


def main():
    ap = argparse.ArgumentParser("Predict REAL/FAKE from feature sequences (binary)")
    ap.add_argument("input", nargs="?", default="extracted_features")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--csv-out", type=str, default=None)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--require-top1", action="store_true")
    ap.add_argument("--delta", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isdir(args.input):
        npys = sorted(
            glob.glob(os.path.join(args.input, "**", "*.npy"), recursive=True)
        )
        if not npys:
            print(f"No .npy files found in {args.input}")
            sys.exit(1)
        if args.latest:
            npys = [max(npys, key=os.path.getmtime)]
    else:
        if not (os.path.exists(args.input) and args.input.lower().endswith(".npy")):
            print("Provide a .npy file or a folder containing .npy files.")
            sys.exit(1)
        npys = [args.input]

    # Load model
    chk = safe_load_checkpoint(args.weights, device)
    state = chk.get("state_dict") or chk
    hparams = chk.get("hparams", {})
    feat_dim = None
    try:
        _arr = np.load(npys[0], allow_pickle=False)
        if _arr.ndim == 2:
            feat_dim = int(_arr.shape[1])
    except Exception:
        pass
    input_size = chk.get("input_size", feat_dim if feat_dim is not None else 2048)
    hidden_size = hparams.get("hidden_size", 512)
    num_layers = hparams.get("num_layers", 2)
    dropout = hparams.get("dropout", 0.5)
    bidirectional = hparams.get("bidirectional", True)
    num_classes = chk.get("num_classes", 2)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    thr, margin = load_thresholds()
    print(f"[thresholds] source: {CALIB_JSON} | thr={thr:.2f} | margin={margin:.2f}")
    classes = CLASSES[:]  # ['fake','real']
    print("Classes:", classes)

    rows = []
    for i, path in enumerate(npys):
        quiet = (args.csv_out is not None) or (i >= 10)
        rows.append(
            classify_file(
                path,
                model,
                classes,
                thr,
                margin,
                topk=args.topk,
                device=device,
                quiet=quiet,
                require_top1=args.require_top1,
                delta=args.delta,
            )
        )

    if args.stats:
        n = len(rows)
        n_real = sum(1 for r in rows if r["label"] == "REAL")
        print(
            f"\n[Stats] N={n} | REAL={n_real} | FAKE={n-n_real} | mean P(real)={np.mean([r['p_real'] for r in rows])*100:.2f}%"
        )

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        headers = ["file", "label", "p_real", "p_fake_any"] + [
            f"p_{c}" for c in classes
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in headers})
        print(f"Saved CSV -> {args.csv_out}")


if __name__ == "__main__":
    main()

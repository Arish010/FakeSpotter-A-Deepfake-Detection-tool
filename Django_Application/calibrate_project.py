# calibrate_project.py — works with .npy features OR videos
import os, csv, argparse, numpy as np, torch

# --- bootstrap Django/app so we can reuse your code ---
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_settings.settings")
django.setup()

from django.conf import settings
from ml_app.views import (
    get_feature_model,
    get_lstm_model,
    _read_frames_bgr,
    _tensor_sequence_from_frames,
    extract_features_batch,
    _device,
)


def resolve_path(p, base_dir):
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return None
    ap = p if os.path.isabs(p) else os.path.join(base_dir, p)
    return os.path.normpath(ap)


def _to_bt2048(arr):
    """
    Normalize various npy shapes to [1, T, 2048].
    Accepts (T,2048), (1,T,2048), (T,2048,1,1). Raises if last dim != 2048.
    """
    a = np.array(arr)
    if a.ndim == 2 and a.shape[1] == 2048:
        a = a[None, ...]  # [1,T,2048]
    elif a.ndim == 3:
        if a.shape[-1] == 2048:
            # either [1,T,2048] or [T,2048,1] (handled below)
            pass
        elif a.shape[1] == 2048:  # [T,2048,?]
            a = np.transpose(a, (2, 0, 1)) if a.shape[0] != 1 else a
        else:
            raise ValueError(f"Unexpected feature shape {a.shape}")
    elif a.ndim == 4 and a.shape[1] == 2048:
        # [T,2048,1,1] -> [T,2048]
        a = a[..., 0, 0]
        a = a[None, ...]
    else:
        raise ValueError(f"Unexpected feature shape {a.shape}")
    if a.shape[-1] != 2048:
        raise ValueError(f"Last dim must be 2048, got {a.shape}")
    return a.astype("float32", copy=False)


@torch.no_grad()
def p_fake_from_path(path):
    """
    Return P(fake) from either a .npy feature file or a raw video file.
    """
    device = _device()
    lstm = get_lstm_model()

    if path.lower().endswith(".npy"):
        # Load precomputed features
        feats_np = np.load(path, allow_pickle=True)
        try:
            feats_np = _to_bt2048(feats_np)  # [1,T,2048]
        except Exception as e:
            print(f"[skip] bad feature shape {path}: {e}")
            return None
        feats = torch.from_numpy(feats_np).to(device)
        logits = lstm(feats)  # [1,2]
    else:
        # Fallback: raw video -> frames -> CNN features
        frames = _read_frames_bgr(path)
        if not frames:
            return None
        x = _tensor_sequence_from_frames(frames, 60)
        feat_model = get_feature_model()
        feats = extract_features_batch(x, feat_model)  # [1,T,2048]
        logits = lstm(feats)

    if getattr(settings, "SWAP_OUTPUT", False):
        logits = logits[:, [1, 0]]
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # mapping (idx->class) with safe default
    idx_to_class = {0: "fake", 1: "real"}
    if getattr(settings, "FORCE_IDX_TO_CLASS", None):
        idx_to_class = {
            int(k): str(v).lower() for k, v in settings.FORCE_IDX_TO_CLASS.items()
        }
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    return float(probs[class_to_idx.get("fake", 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV with path in first column and label in last column",
    )
    ap.add_argument("--base", default="", help="Base folder to resolve relative paths")
    ap.add_argument(
        "--out",
        default="",
        help="Output threshold file (defaults to settings.CALIBRATED_THRESHOLD_PATH)",
    )
    args = ap.parse_args()

    base_dir = args.base or settings.PROJECT_DIR
    out_path = args.out or settings.CALIBRATED_THRESHOLD_PATH

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            # Skip header-ish first row
            if i == 0 and any(
                tok in ",".join(row).lower()
                for tok in ("path", "label", "video", "split")
            ):
                continue
            vpath = resolve_path(row[0], base_dir)
            if not vpath or not os.path.exists(vpath):
                print(f"[skip] missing file: {row[0]}")
                continue
            label = (row[-1] or "").strip().lower()
            if label not in {"real", "fake"}:
                label = "fake" if label in {"1", "f"} else "real"
            rows.append((vpath, 1 if label == "fake" else 0))

    if len(rows) < 8:
        print(f"Not enough usable samples (found {len(rows)}).")
        return

    y, s = [], []
    for vpath, lab in rows:
        pf = p_fake_from_path(vpath)
        if pf is None:
            print(f"[skip] could not read: {vpath}")
            continue
        s.append(pf)
        y.append(lab)

    if len(s) < 8:
        print("Too few successful evaluations after decoding. Try more clips.")
        return

    y = np.array(y, dtype=int)
    s = np.array(s, dtype=float)

    # Sweep thresholds
    ts = np.linspace(0.05, 0.95, 91)
    best = {"t": 0.5, "f1": -1}
    for t in ts:
        pred = (s >= t).astype(int)  # 1=fake
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best["f1"]:
            best = {"t": float(t), "f1": float(f1)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{best['t']:.4f}\n")

    print(f"Calibrated threshold written to:\n  {out_path}")
    print(f"t={best['t']:.4f}  F1={best['f1']:.3f}")


if __name__ == "__main__":
    main()

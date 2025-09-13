# ml_app/inference.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings
from .models_lstm import LSTMClassifier
import json


def _normalize_classes(classes):
    if isinstance(classes, (list, tuple)):
        return [str(x).lower() for x in classes]
    if isinstance(classes, dict):
        keys = [k for k in classes.keys() if isinstance(k, int)]
        if keys:
            return [str(classes[i]).lower() for i in sorted(keys)]
        try:
            return [
                str(lbl).lower()
                for lbl, _ in sorted(classes.items(), key=lambda kv: kv[1])
            ]
        except Exception:
            pass
    return ["real", "fake"]


def _load_thresholds():

    thr_path = getattr(settings, "CALIBRATED_THRESHOLD_PATH", None)
    thr = None
    margin = None

    if thr_path and os.path.exists(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    thr = float(data.get("thr", data.get("threshold", 0.5)))
                    margin = float(data.get("margin", 0.0))
                elif isinstance(data, (int, float, str)):
                    thr = float(data)
            except Exception:
                try:
                    thr = float(raw)
                except ValueError:
                    if "=" in raw:
                        # e.g. "thr=0.45" or "threshold=0.45"
                        parts = raw.split("=", 1)
                        thr = float(parts[1].strip())
        except Exception:
            thr = None
            margin = None

    if thr is None:
        thr = 0.5
    if margin is None:
        margin = float(getattr(settings, "UNCERTAIN_BAND", 0.0) or 0.0)

    return thr, margin


def _decide_strict(prob, classes, thr, margin, require_top1=False, delta=0.0):
    import numpy as np
    from django.conf import settings

    prob = np.asarray(prob, dtype=float).ravel()

    if isinstance(classes, dict):
        try:
            order = [classes[i] for i in sorted(classes.keys())]
        except Exception:
            order = list(classes.values())
    else:
        order = list(classes) if classes is not None else []

    if not order or len(order) != len(prob):
        mapping = getattr(settings, "FORCE_IDX_TO_CLASS", None)
        if isinstance(mapping, dict):
            try:
                order = [mapping[i] for i in sorted(mapping.keys())][: len(prob)]
            except Exception:
                pass

    # Truncate to match prob length
    order = (order or [])[: len(prob)]
    order_low = [str(c).lower() for c in order]

    # Build label->prob map
    label2p = {order_low[i]: float(prob[i]) for i in range(len(order_low))}

    if "real" not in label2p and len(prob) == 2:
        if "fake" in label2p:
            other_idx = 1 if order_low and order_low[0] == "fake" else 0
            label2p["real"] = float(prob[other_idx])
        else:
            try:
                mapping = getattr(settings, "FORCE_IDX_TO_CLASS", {})
                inv = {str(v).lower(): k for k, v in mapping.items()}
                if "real" in inv:
                    idx = inv["real"]
                    if 0 <= idx < len(prob):
                        label2p["real"] = float(prob[idx])
            except Exception:
                pass

    p_real = float(label2p.get("real", 0.0))
    p_fake_any = float(sum(v for k, v in label2p.items() if k != "real"))

    # Decision
    is_real = (p_real >= float(thr)) and ((p_real - p_fake_any) >= float(margin))

    if require_top1:
        # top label by probability
        if label2p:
            top_label = max(label2p.items(), key=lambda kv: kv[1])[0]
        else:
            top_label = "fake"
        is_real = is_real and (top_label == "real")
        if delta > 0.0 and top_label == "real":
            vals = sorted(label2p.values(), reverse=True)
            runner_up = vals[1] if len(vals) > 1 else 0.0
            is_real = is_real and ((p_real - runner_up) >= float(delta))

    decision = "REAL" if is_real else "FAKE"

    # Helpful debug (keeps running even if logging off)
    try:
        print(
            f"[decide] classes={order} prob={prob.tolist()} "
            f"mapped(real={p_real:.3f}, fake_any={p_fake_any:.3f}) -> {decision}"
        )
    except Exception:
        pass

    return decision, p_real, p_fake_any


# ---------------- model loader ----------------


def load_model():
    import warnings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load checkpoint/state ---
    chk = torch.load(settings.LSTM_WEIGHTS_PATH, map_location=device)
    state = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
    if not isinstance(state, dict):
        raise RuntimeError("[inference] Bad checkpoint format (no state_dict).")

    keys = list(state.keys())

    bidirectional = any("reverse" in k for k in keys)

    fwd_layers = sorted(
        {
            int(k.split("_l")[1].split(".")[0])
            for k in keys
            if k.startswith("lstm.weight_ih_l") and "_reverse" not in k
        }
    )
    num_layers = (max(fwd_layers) + 1) if fwd_layers else 2  # sensible default

    hidden_size = 256
    if "fc.weight" in state:
        in_feats = state["fc.weight"].shape[1]
        num_dir = 2 if bidirectional else 1
        if in_feats % num_dir == 0:
            hidden_size = in_feats // num_dir
    else:
        try:
            w_hh = state["lstm.weight_hh_l0"]
            hidden_size = w_hh.shape[1]
        except Exception:
            pass

    num_classes = 2

    model = LSTMClassifier(
        input_size=2048,
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        num_classes=int(num_classes),
        dropout=0.5,
        bidirectional=bool(bidirectional),
    ).to(device)

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception as e:
        warnings.warn(f"[inference] strict=False load failed: {e}")
        missing, unexpected = [], []

    if missing:
        warnings.warn(
            f"[inference] missing keys: {missing[:8]}{' …' if len(missing)>8 else ''}"
        )
    if unexpected:
        warnings.warn(
            f"[inference] unexpected keys: {unexpected[:8]}{' …' if len(unexpected)>8 else ''}"
        )

    model.eval()

    mapping = None
    if isinstance(chk, dict) and "idx_to_class" in chk:
        try:
            mapping = {int(k): str(v).lower() for k, v in chk["idx_to_class"].items()}
        except Exception:
            mapping = None
    if mapping is None and getattr(settings, "FORCE_IDX_TO_CLASS", None):
        mapping = {
            int(k): str(v).lower() for k, v in settings.FORCE_IDX_TO_CLASS.items()
        }
    if mapping is None:
        mapping = {0: "fake", 1: "real"}

    globals()["_IDX_TO_CLASS"] = mapping
    print(
        f"[inference] LSTM inferred -> bidir={bidirectional} layers={num_layers} hidden={hidden_size}  classes={mapping}"
    )

    return model, mapping, device


# ---------------- probability computation ----------------


def _logits_to_probs(logits):
    """Convert model outputs to well-formed probabilities array [B, 2]."""
    with torch.no_grad():
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        B, D = logits.shape
        if D == 2:
            probs = F.softmax(logits, dim=1)
        elif D == 1:
            # binary sigmoid: D=1 means "fake" logit
            p_fake = torch.sigmoid(logits)
            p_real = 1.0 - p_fake
            probs = torch.cat([p_real, p_fake], dim=1)
        else:
            # multi-class fallback
            probs = F.softmax(logits, dim=1)
    return probs


# ---------------- main API ----------------


@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
def predict_npy_files(npy_paths, require_top1=False, delta=0.0, thr=None, margin=None):

    # ---- Load model + mapping
    model, idx_to_class, device = load_model()

    if isinstance(idx_to_class, dict):
        classes = [str(idx_to_class[i]).lower() for i in sorted(idx_to_class.keys())]
    else:
        classes = _normalize_classes(idx_to_class)

    if not classes or len(classes) < 2 or not {"real", "fake"} <= set(classes):
        m = getattr(settings, "FORCE_IDX_TO_CLASS", {0: "fake", 1: "real"})
        classes = [str(m.get(0, "fake")).lower(), str(m.get(1, "real")).lower()]

    # Indices inside the model's probability vector
    try:
        i_fake = classes.index("fake")
        i_real = classes.index("real")
    except ValueError:
        i_fake, i_real = 0, 1

    # Thresholds
    if thr is None or margin is None:
        t = _load_thresholds()
        if isinstance(t, (tuple, list)) and len(t) >= 2:
            thr, margin = float(t[0]), float(t[1])
        elif isinstance(t, dict):
            thr, margin = float(t.get("thr", 0.5)), float(t.get("margin", 0.0))
        else:
            thr, margin = 0.5, 0.0

    target_len = int(getattr(settings, "SEQ_LEN", 24) or 24)
    results = []
    model.eval()

    for npy_path in npy_paths:
        fname = os.path.basename(npy_path)
        try:
            # ---- Load features -> [1,T,D]
            arr = np.load(npy_path, allow_pickle=False)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2:
                T, F = arr.shape
                if T > target_len:
                    idx = np.linspace(0, T - 1, target_len).astype(int)
                    arr = arr[idx]
                elif T < target_len:
                    pad = np.zeros((target_len - T, F), dtype=arr.dtype)
                    arr = np.vstack([arr, pad])
                arr = arr[None, ...]
            if arr.ndim != 3:
                raise ValueError(
                    f"Unexpected feature shape {arr.shape}, expected [1,T,D]"
                )

            # Basic NaN/const check
            if not np.isfinite(arr).all() or np.allclose(arr.std(), 0.0, atol=1e-6):
                print(
                    f"[infer] {fname} features look degenerate (NaN/const) -> defaulting to argmax later"
                )

            # ---- Forward pass
            x = torch.from_numpy(arr).float().to(device)
            probs = _logits_to_probs(model(x))[0].detach().cpu().numpy().astype(float)

            p_fake = float(probs[i_fake])
            p_real = float(probs[i_real])

            # Argmax decision
            argmax_label = "REAL" if p_real >= p_fake else "FAKE"
            top_gap = abs(p_real - p_fake)

            # Threshold decision
            thr_label = (
                "REAL" if (p_real >= thr and (p_real - p_fake) >= margin) else "FAKE"
            )

            if argmax_label != thr_label and top_gap >= 0.15:
                label = argmax_label
                guard = "argmax_override"
            else:
                label = thr_label
                guard = "threshold"

            results.append(
                {
                    "file": fname,
                    "label": label.lower(),
                    "prob_real": p_real,
                    "prob_fake": p_fake,
                    "confidence": float(max(p_real, p_fake)),
                    "class_probs": [
                        {"label": classes[i_fake], "prob": p_fake},
                        {"label": classes[i_real], "prob": p_real},
                    ],
                }
            )

            print(
                f"[infer] {fname} order={classes} probs={probs.tolist()} "
                f"(i_real={i_real} i_fake={i_fake}) thr={thr:.3f} margin={margin:.3f} "
                f"argmax={argmax_label} gap={top_gap:.3f} final={label} via {guard}"
            )

        except Exception as e:
            results.append(
                {
                    "file": fname,
                    "label": "error",
                    "prob_real": 0.0,
                    "prob_fake": 0.0,
                    "confidence": 0.0,
                    "class_probs": [],
                    "error": str(e),
                }
            )

    info = {
        "thresholds": f"thr={float(thr):.3f} margin={float(margin):.3f}",
        "classes": classes,
    }
    return results, info


@torch.no_grad()
def predict_folder(
    features_dir,
    recursive=True,
    latest=True,
    require_top1=False,
    delta=0.0,
    thr=None,
    margin=None,
):
    all_paths = []
    for root, _dirs, files in os.walk(features_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                all_paths.append(os.path.join(root, f))
        if not recursive:
            break

    if latest and all_paths:
        all_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        all_paths = all_paths[:1]

    return predict_npy_files(
        all_paths, require_top1=require_top1, delta=delta, thr=thr, margin=margin
    )

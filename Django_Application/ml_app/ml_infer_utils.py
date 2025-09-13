# ml_app/ml_infer_utils.py
import numpy as np


def aggregate_video_probs(chunk_probs):
    return np.mean(np.stack(chunk_probs, axis=0), axis=0)


def decide_real_vs_fake(prob_vec, classes, thr, margin=0.05, mode="strict"):
    assert "real" in classes, "Expected a 'real' class in classes list"
    real_idx = classes.index("real")
    p_real = float(prob_vec[real_idx])
    p_fake_any = float(np.sum(prob_vec[np.arange(len(classes)) != real_idx]))

    if mode == "balanced":
        is_real = p_real >= thr
    else:  # strict
        is_real = (p_real >= thr) and ((p_real - p_fake_any) >= margin)

    return ("REAL" if is_real else "FAKE"), p_real, p_fake_any

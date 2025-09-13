# make_confusion_matrix.py
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools, os

CLASS_NAMES = ["Real", "Fake"]  # display order for the axes
LABELS = [0, 1]  # numeric ids corresponding to CLASS_NAMES

CALIBRATED_THRESHOLD_PATH = "best_threshold.json"


def plot_cm(
    cm,
    class_names,
    normalize=True,
    out_pdf="fig_confusion_matrix.pdf",
    out_png="fig_confusion_matrix.png",
):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # guard against division by zero
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)  # compact for IEEE
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)

    # numbers inside each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_pdf} and {out_png}")


# ---- load arrays saved by evaluate_test_6.py ----
y_true = np.load("y_true_test.npy")  # shape [N]
y_prob = np.load("y_prob_test.npy")  # shape [N] (binary) or [N, C] (multiclass)

# ---- get predictions ----
if y_prob.ndim == 1:
    try:
        with open(CALIBRATED_THRESHOLD_PATH) as f:
            tau = json.load(f).get("best_threshold", 0.5)
    except FileNotFoundError:
        tau = 0.5
    print(f"Using threshold τ = {tau:.3f}")
    y_pred = (y_prob >= tau).astype(int)
else:
    # Multiclass
    y_pred = y_prob.argmax(axis=1)

# ---- compute and plot confusion matrix ----
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
plot_cm(cm, CLASS_NAMES, normalize=True)  # normalized for paper

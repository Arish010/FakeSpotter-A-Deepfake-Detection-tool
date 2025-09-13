# make_roc_pr_curves.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---- inputs produced by evaluate_test_6.py ----
y_true = np.load("y_true_test.npy")  # shape [N], with 0='fake', 1='real'
y_prob = np.load("y_prob_test.npy")  # shape [N], probabilities of 'fake'

POS_LABEL = 0
y_pos = (y_true == POS_LABEL).astype(int)

# ---- ROC ----
fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=POS_LABEL)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(3.0, 3.0), dpi=300)  # compact for IEEE two-column
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Positive: fake)")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("fig_roc_curve.pdf", bbox_inches="tight")
plt.savefig("fig_roc_curve.png", bbox_inches="tight", dpi=300)
print("Saved: fig_roc_curve.pdf / fig_roc_curve.png")

# ----  mark your calibrated operating point on the ROC ----
tau = None
for path in ["best_threshold.json", "calibrated_threshold.json"]:
    if os.path.exists(path):
        try:
            d = json.load(open(path, "r"))
            tau = d.get("thr") or d.get("best_threshold")
            if tau is not None:
                break
        except Exception:
            pass
if tau is not None:
    TP = np.sum((y_prob >= tau) & (y_pos == 1))
    FN = np.sum((y_prob < tau) & (y_pos == 1))
    FP = np.sum((y_prob >= tau) & (y_pos == 0))
    TN = np.sum((y_prob < tau) & (y_pos == 0))
    tpr_op = TP / (TP + FN + 1e-12)
    fpr_op = FP / (FP + TN + 1e-12)

    plt.figure(figsize=(3.0, 3.0), dpi=300)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.scatter([fpr_op], [tpr_op], s=14)  # operating point
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC with Calibrated Operating Point")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig("fig_roc_curve_with_op.pdf", bbox_inches="tight")
    plt.savefig("fig_roc_curve_with_op.png", bbox_inches="tight", dpi=300)
    print(f"Saved: fig_roc_curve_with_op.* (τ={tau:.3f})")

# ----  Precision–Recall curve (good for imbalance) ----
prec, rec, _ = precision_recall_curve(y_pos, y_prob)  # PR uses y_pos explicitly
ap = average_precision_score(y_pos, y_prob)

plt.figure(figsize=(3.0, 3.0), dpi=300)
plt.plot(rec, prec, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall (Positive: fake)")
plt.legend(loc="lower left", frameon=False)
plt.tight_layout()
plt.savefig("fig_pr_curve.pdf", bbox_inches="tight")
plt.savefig("fig_pr_curve.png", bbox_inches="tight", dpi=300)
print("Saved: fig_pr_curve.pdf / fig_pr_curve.png")

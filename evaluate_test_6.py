# evaluate_test_6.py
import os, json, random, numpy as np, torch
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DJANGO_APP_ROOT = os.path.join(PROJECT_ROOT, "Django_Application")
if DJANGO_APP_ROOT not in sys.path:
    sys.path.insert(0, DJANGO_APP_ROOT)

# --- ABS PATH SHIM (force-resolve ml_app on Windows) ---
import sys

sys.path.insert(0, DJANGO_APP_ROOT)
# pyright: reportMissingImports=false
# -------------------------------------------------------

from ml_app.models_lstm import LSTMClassifier

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "extracted_features")
WEIGHTS = os.path.join(PROJECT_ROOT, "lstm_binary_weights.pth")
print("[info] Using weights at:", os.path.abspath(WEIGHTS))

CALIB_JSON = os.path.join(PROJECT_ROOT, "best_threshold.json")

CLASSES = ["fake", "real"]  # fixed
MAX_SEQ_LEN = 24
BATCH_SIZE = 64
SEED = 42


# ---- utils ----
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_features(features_dir, allowed_classes):
    rows = []
    for c in allowed_classes:
        cdir = os.path.join(features_dir, c)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if fn.lower().endswith(".npy"):
                fp = os.path.join(cdir, fn)
                if os.path.isfile(fp):
                    rows.append((fp, c))
    return rows


def stratified_split(rows, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=SEED):
    rng = random.Random(seed)
    by_cls = {}
    for p, c in rows:
        by_cls.setdefault(c, []).append((p, c))
    train, val, test = [], [], []
    for c, items in by_cls.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio)))
        n_te = max(1, n - n_tr - n_va)
        while n_tr + n_va + n_te > n:
            if n_te > 1:
                n_te -= 1
            elif n_va > 1:
                n_va -= 1
            else:
                n_tr -= 1
        train += items[:n_tr]
        val += items[n_tr : n_tr + n_va]
        test += items[n_tr + n_va : n_tr + n_va + n_te]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def class_index_map(cls):
    return {c: i for i, c in enumerate(cls)}


class SeqFeatureDataset(Dataset):
    def __init__(self, rows, class_to_idx, max_seq_len=60):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.max_seq_len = max_seq_len
        self.feat_dim = None
        for fp, _ in rows:
            arr = np.load(fp, allow_pickle=False)
            if arr.ndim == 2:
                self.feat_dim = arr.shape[1]
                break
        if self.feat_dim is None:
            raise ValueError("Could not infer feature dim.")

    def __len__(self):
        return len(self.rows)

    def _pad_or_trunc(self, x):
        T, F = x.shape
        if T == self.max_seq_len:
            return x
        if T > self.max_seq_len:
            idx = np.linspace(0, T - 1, self.max_seq_len).astype(int)
            return x[idx]
        pad = np.zeros((self.max_seq_len - T, F), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx):
        path, cname = self.rows[idx]
        y = self.class_to_idx[cname]
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected [T,F], got {arr.shape} in {path}")
        arr = self._pad_or_trunc(arr).astype(np.float32)
        x = torch.from_numpy(arr)
        return x, y


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_thresholds(path):
    if not os.path.exists(path):
        print(f"[WARN] Threshold file not found at {path}; using defaults (0.5, 0.0).")
        return 0.5, 0.0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("thr", 0.5)), float(data.get("margin", 0.0))


# ---- main ----
def main():
    set_seed()
    rows = scan_features(FEATURES_DIR, CLASSES)
    if not rows:
        raise FileNotFoundError(f"No .npy features under {FEATURES_DIR}")
    tr, va, te = stratified_split(rows, 0.70, 0.15, 0.15, seed=SEED)
    print(f"[info] Auto-split -> train:{len(tr)} val:{len(va)} test:{len(te)}")
    print(
        f"[info] Val counts: {Counter([c for _,c in va])} | Test counts: {Counter([c for _,c in te])}"
    )

    c2i = class_index_map(CLASSES)
    val_ds = SeqFeatureDataset(va, c2i, max_seq_len=MAX_SEQ_LEN)
    test_ds = SeqFeatureDataset(te, c2i, max_seq_len=MAX_SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    chk = load_checkpoint(WEIGHTS, DEVICE)
    state = chk.get("state_dict") or chk
    hparams = chk.get("hparams", {})
    input_size = chk.get("input_size", getattr(val_ds, "feat_dim", 2048))
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
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    def run_split(loader, name):
        y_true, y_pred = [], []
        y_prob = []
        fake_idx = CLASSES.index("fake")
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.numpy().tolist())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_prob.extend(probs[:, fake_idx].tolist())

        print(f"\n{name.upper()} report:")
        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=2))
        print(f"{name.upper()} confusion:")
        print(confusion_matrix(y_true, y_pred))

        if str(name).lower() == "test":
            np.save("y_true_test.npy", np.array(y_true, dtype=np.int64))
            np.save("y_prob_test.npy", np.array(y_prob, dtype=np.float32))
            print("[info] Saved arrays: y_true_test.npy, y_prob_test.npy")

    print("Classes:", CLASSES)
    run_split(val_loader, "val")
    run_split(test_loader, "test")


if __name__ == "__main__":
    main()

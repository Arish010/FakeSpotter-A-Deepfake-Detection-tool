# calibrate_threshold_5.py
import os, json, random, numpy as np, torch
from collections import Counter
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

import sys, os

_this_dir = os.path.abspath(os.path.dirname(__file__))
_project_dir = os.path.abspath(os.path.join(_this_dir, os.pardir))
_django_app_dir = os.path.join(_project_dir, "Django_Application")
if _django_app_dir not in sys.path:
    sys.path.insert(0, _django_app_dir)
# -------------------------------------------------------------

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project_settings.settings")
# pyright: reportMissingImports=false
# -------------------------------------------------------

try:
    from Django_Application.ml_app.models_lstm import LSTMClassifier
except Exception:
    from models_lstm import LSTMClassifier

# ---- CONFIG ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_DIR = os.path.join(_this_dir, "extracted_features")
SAVE_JSON = os.path.join(_project_dir, "best_threshold.json")
WEIGHTS = os.path.join(_this_dir, "lstm_binary_weights.pth")
print("[info] Using weights at:", os.path.abspath(WEIGHTS))

CLASSES = ["fake", "real"]  # fixed binary order
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
        while (n_tr + n_va + n_te) > n:
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
            raise ValueError("Could not infer feature dim from dataset.")

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


def softmax_logits(logits):
    return torch.softmax(logits, dim=1)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# ---- main ----
def main():
    set_seed()

    if not os.path.isfile(WEIGHTS):
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")
    rows = scan_features(FEATURES_DIR, CLASSES)
    if not rows:
        raise FileNotFoundError(f"No .npy features under {FEATURES_DIR}")
    tr, va, te = stratified_split(rows, 0.70, 0.15, 0.15, seed=SEED)
    print(
        f"Auto-split -> train:{len(tr)} val:{len(va)} test:{len(te)} "
        f"| counts val: {Counter([c for _,c in va])}"
    )

    c2i = class_index_map(CLASSES)
    val_ds = SeqFeatureDataset(va, c2i, max_seq_len=MAX_SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # load model
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

    # collect probs, labels
    probs_real, y_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = softmax_logits(logits).cpu().numpy()  # [B,2] => [fake, real]
            probs_real.extend(probs[:, 1].tolist())
            y_true.extend(yb.numpy().tolist())

    probs_real = np.asarray(probs_real, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    best_thr, best_f1 = 0.5, -1.0
    thr_grid = np.linspace(0.05, 0.95, 19)  # step 0.05
    for thr in thr_grid:
        y_pred = (probs_real >= thr).astype(int)  # 1 = real, 0 = fake
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
    result = {"thr": round(best_thr, 2), "margin": 0.0}
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved best threshold: {best_thr:.2f} (F1={best_f1:.3f}) -> {SAVE_JSON}")


if __name__ == "__main__":
    main()

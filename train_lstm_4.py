# train_lstm_4.py — Binary LSTM trainer (fake vs real), balanced + robust
# --- ABS PATH SHIM ---
import sys

sys.path.insert(
    0,
    r"C:\Users\Asus\Desktop\Project Msc\Deepfake_detection_using_deep_learning\Django_Application",
)
# ----------------------
from ml_app.models_lstm import LSTMClassifier as LSTMNet

import argparse, os, random, time, json
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score


# ---------- utils ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


FAKE_ALIASES = {"fake", "deepfakes", "deepfake", "df", "faceswap", "forgery", "spoof"}
REAL_ALIASES = {"real", "genuine", "authentic"}


def norm_label(name: str):
    n = name.strip().lower()
    if n in FAKE_ALIASES:
        return "fake"
    if n in REAL_ALIASES:
        return "real"
    return None


def list_dirs(d):
    return [x for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x))]


def scan_binary_features(features_dir: str):
    rows = []
    ignored = []
    for d in list_dirs(features_dir):
        canon = norm_label(d)
        if canon is None:
            ignored.append(d)
            continue
        cdir = os.path.join(features_dir, d)
        for fn in os.listdir(cdir):
            if fn.lower().endswith(".npy"):
                rows.append((os.path.join(cdir, fn), canon))
    return rows, ["fake", "real"], ignored


def counts_str(rows):
    cnt = Counter([c for _, c in rows])
    return " ".join([f"{k}:{cnt.get(k,0)}" for k in ["fake", "real"]])


def limit_balanced(rows, per_class=0, total_cap=0, seed=42, shuffle=True):
    rng = random.Random(seed)
    by = defaultdict(list)
    for p, c in rows:
        by[c].append((p, c))
    for c in by:
        if shuffle:
            rng.shuffle(by[c])
        if per_class and per_class > 0:
            by[c] = by[c][:per_class]
    if total_cap and total_cap > 0:
        half = max(1, total_cap // 2)
        by["fake"] = by["fake"][:half]
        by["real"] = by["real"][: total_cap - len(by["fake"])]
    out = by["fake"] + by["real"]
    if shuffle:
        rng.shuffle(out)
    return out


def stratified_split(rows, tr=0.70, va=0.15, te=0.15, seed=123):
    rng = random.Random(seed)
    by = defaultdict(list)
    for p, c in rows:
        by[c].append((p, c))
    train, val, test = [], [], []
    for c, items in by.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = max(1, int(round(n * tr)))
        n_va = max(1, int(round(n * va)))
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


# ---------- dataset ----------
class SeqFeatureDataset(Dataset):
    def __init__(self, rows, class_to_idx, max_seq_len=60):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.max_seq_len = max_seq_len
        self.feat_dim = None
        for fp, _ in self.rows:
            try:
                arr = np.load(fp, allow_pickle=False)
                if arr.ndim == 2:
                    self.feat_dim = arr.shape[1]
                    break
            except Exception:
                pass
        if self.feat_dim is None:
            self.feat_dim = 2048

    def __len__(self):
        return len(self.rows)

    def _pad_or_trunc(self, x):
        T, F = x.shape
        if T == self.max_seq_len:
            return x
        if T > self.max_seq_len:
            idxs = np.linspace(0, T - 1, self.max_seq_len).astype(int)
            return x[idxs]
        pad = np.zeros((self.max_seq_len - T, F), dtype=x.dtype)
        return np.concatenate([x, pad], 0)

    def __getitem__(self, i):
        path, cname = self.rows[i]
        y = self.class_to_idx[cname]
        try:
            arr = np.load(path, allow_pickle=False)
            if arr.ndim != 2:
                raise ValueError("bad ndim")
        except Exception:
            arr = np.zeros((self.max_seq_len, self.feat_dim), dtype=np.float32)
        arr = self._pad_or_trunc(arr.astype(np.float32))
        return torch.from_numpy(arr), y


# ---------- training ----------
@dataclass
class TrainConfig:
    features_dir: str
    save_path: str
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    max_seq_len: int = 24  # <— match extractor (was 60)
    hidden_size: int = 512
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.5
    patience: int = 6
    seed: int = 42
    balance: str = "sampler"  # 'sampler' | 'weights' | 'none'
    max_videos_per_class: int = 0
    max_videos_total: int = 0


def build_model(input_size, num_classes, cfg: TrainConfig, device):
    return LSTMNet(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=num_classes,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)


def acc_from_logits(logits, y):
    return (torch.argmax(logits, 1) == y).float().mean().item()


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    tl = ta = tn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        bs = y.size(0)
        tl += loss.item() * bs
        ta += acc_from_logits(logits, y) * bs
        tn += bs
    return tl / tn, ta / tn


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tl = ta = tn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        bs = y.size(0)
        tl += loss.item() * bs
        ta += acc_from_logits(logits, y) * bs
        tn += bs
    return tl / tn, ta / tn


def calibrate_threshold(model, val_loader, device, out_json, class_to_idx):
    model.eval()
    all_p_real, all_y = [], []
    real_idx = class_to_idx["real"]
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            p = torch.softmax(model(X), dim=1)[:, real_idx]
            all_p_real += p.cpu().tolist()
            all_y += y.cpu().tolist()
    all_p_real = np.asarray(all_p_real)
    all_y = np.asarray(all_y)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 17):
        preds = (all_p_real >= thr).astype(int)  # 0=fake, 1=real
        f1 = f1_score(all_y, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"thr": best_thr, "margin": 0.05}, f)
    print(f"✅ Saved best threshold: {best_thr:.2f} (F1={best_f1:.3f}) -> {out_json}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default=None)
    ap.add_argument("--save_path", default="lstm_binary_weights.pth")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq_len", type=int, default=24)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--balance", default="sampler", choices=["sampler", "weights", "none"]
    )
    ap.add_argument("--max_videos_per_class", type=int, default=0)
    ap.add_argument("--max_videos_total", type=int, default=0)
    ap.add_argument("--no_shuffle_cap", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    # locate features_dir
    script_dir = os.path.abspath(os.path.dirname(__file__))
    if args.features_dir is None:
        cand = os.path.join(
            os.path.abspath(os.path.join(script_dir, os.pardir)), "extracted_features"
        )
        args.features_dir = (
            cand
            if os.path.isdir(cand)
            else os.path.join(script_dir, "extracted_features")
        )
    if not os.path.isdir(args.features_dir):
        print("❌ Could not find 'extracted_features' directory")
        sys.exit(1)

    rows, class_names, ignored = scan_binary_features(args.features_dir)
    print(f"Features dir: {args.features_dir}")
    if ignored:
        print(f"Ignoring folders (not fake/real): {ignored}")
    print(f"Classes (fixed): {class_names}")
    print(f"Total samples (before caps): {len(rows)}  |  {counts_str(rows)}")
    if not rows:
        print("❌ No usable features found.")
        sys.exit(1)

    # must have both classes
    cnt = Counter([c for _, c in rows])
    if cnt.get("fake", 0) == 0 or cnt.get("real", 0) == 0:
        print("❌ Need both classes present (fake & real). Re-run extractor with both.")
        sys.exit(1)

    rows = limit_balanced(
        rows,
        per_class=args.max_videos_per_class,
        total_cap=args.max_videos_total,
        seed=args.seed,
        shuffle=(not args.no_shuffle_cap),
    )
    print(f"Using capped dataset: {len(rows)}  |  {counts_str(rows)}")

    tr, va, te = stratified_split(rows, 0.70, 0.15, 0.15, seed=args.seed)
    print(
        f"Split sizes -> train:{len(tr)} ({counts_str(tr)}) | val:{len(va)} ({counts_str(va)}) | test:{len(te)} ({counts_str(te)})"
    )

    class_to_idx = {"fake": 0, "real": 1}
    ds_tr = SeqFeatureDataset(tr, class_to_idx, max_seq_len=args.max_seq_len)
    ds_va = SeqFeatureDataset(va, class_to_idx, max_seq_len=args.max_seq_len)
    ds_te = SeqFeatureDataset(te, class_to_idx, max_seq_len=args.max_seq_len)
    input_size = ds_tr.feat_dim
    num_classes = 2
    print(f"Input feature dim: {input_size} | Num classes: {num_classes}")

    split_path = os.path.join(os.path.dirname(args.features_dir), "splits_binary.json")

    def _rel(p):
        return os.path.relpath(p, os.path.dirname(args.features_dir))

    payload = {
        "train": [_rel(p) for p, _ in tr],
        "val": [_rel(p) for p, _ in va],
        "test": [_rel(p) for p, _ in te],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[split] Saved -> {split_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cfg = TrainConfig(
        features_dir=args.features_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        balance=args.balance,
        max_videos_per_class=args.max_videos_per_class,
        max_videos_total=args.max_videos_total,
    )

    model = build_model(input_size, num_classes, cfg, device)

    # balancing
    if args.balance == "weights":
        ytr = [class_to_idx[c] for _, c in tr]
        cnts = Counter(ytr)
        total = sum(cnts.values())
        w = torch.tensor(
            [total / (len(cnts) * cnts.get(i, 1)) for i in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=w)
        train_sampler = None
        print(f"Using class-weighted CE: {w.detach().cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        ytr = [class_to_idx[c] for _, c in tr]
        cnts = Counter(ytr)
        train_sampler = None
        if args.balance == "sampler" and len(cnts) == 2 and cnts.get(0) != cnts.get(1):
            total = sum(cnts.values())
            class_w = {cls: total / (len(cnts) * cnt) for cls, cnt in cnts.items()}
            sample_w = [class_w[class_to_idx[c]] for _, c in tr]
            train_sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_w),
                num_samples=len(sample_w),
                replacement=True,
            )
            print(f"Using WeightedRandomSampler (class weights): {class_w}")
        else:
            if args.balance == "sampler":
                print("Sampler disabled (split already balanced).")

    loader_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        sampler=train_sampler if train_sampler else None,
        shuffle=(train_sampler is None),
    )
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, noimp = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loader_tr, criterion, opt, device)
        va_loss, va_acc = evaluate(model, loader_va, criterion, device)
        print(
            f"[{ep:02d}/{args.epochs}] train {tr_loss:.4f}/{tr_acc*100:.2f}% | val {va_loss:.4f}/{va_acc*100:.2f}%"
        )
        if va_acc > best_val + 1e-6:
            best_val = va_acc
            best_state = {"state_dict": model.state_dict()}
            sp_dir = os.path.dirname(args.save_path)
            if sp_dir:
                os.makedirs(sp_dir, exist_ok=True)
            torch.save(best_state, args.save_path)
            print(f"💾 Saved best -> {args.save_path} (val acc {va_acc*100:.2f}%)")
            noimp = 0
        else:
            noimp += 1
            if noimp >= args.patience:
                print("Early stopping.")
                break

    print(
        f"Training finished in {(time.time()-t0)/60.0:.1f} min. Best val acc: {best_val*100:.2f}%"
    )

    # test + threshold
    if os.path.isfile(args.save_path):
        chk = torch.load(args.save_path, map_location=device)
        model.load_state_dict(chk["state_dict"])
        te_loss, te_acc = evaluate(model, loader_te, criterion, device)
        print(f"TEST -> loss {te_loss:.4f} | acc {te_acc*100:.2f}%")
        thr_path = os.path.join(
            os.path.dirname(args.features_dir), "best_threshold.json"
        )
        calibrate_threshold(model, loader_va, device, thr_path, class_to_idx)
    else:
        print("⚠️ No model found, skipping test/calibration.")


if __name__ == "__main__":
    main()

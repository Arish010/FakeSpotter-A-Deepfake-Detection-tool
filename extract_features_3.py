# extract_features_3.py — fast extractor with balanced caps and alias mapping
import os, re, argparse, random, sys
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from PIL import Image
import cv2
from pathlib import Path

# ----------- Paths (project-relative, portable) -----------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "frames_dataset"  # class folders and/or video files
DEFAULT_OUTPUT = PROJECT_DIR / "extracted_features"  # .npy out

# quick demo defaults (you can bump these later)
DEF_IMG_SIZE = 112
DEF_MAX_FRAMES = 24
DEF_BATCH_SIZE = 64
DEF_PER_CLASS_CAP = 150
DEF_TOTAL_CAP = 300
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

# ----- label aliasing -----
FAKE_ALIASES = {"fake", "deepfakes", "deepfake", "df", "faceswap", "forgery", "spoof"}
REAL_ALIASES = {"real", "genuine", "authentic"}


def norm_label(name: str):
    n = name.strip().lower()
    if n in FAKE_ALIASES:
        return "fake"
    if n in REAL_ALIASES:
        return "real"
    return None


# ----- util -----
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def evenly_spaced_indices(n_total: int, k: int):
    if n_total <= 0:
        return []
    if k >= n_total:
        return list(range(n_total))
    step = n_total / float(k)
    idxs = [min(int(i * step), n_total - 1) for i in range(k)]
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def select_frame_paths(folder: str, max_frames: int):
    frames = [
        f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    frames.sort(key=natural_key)
    if not frames:
        return []
    idxs = evenly_spaced_indices(len(frames), max_frames)
    return [os.path.join(folder, frames[i]) for i in idxs]


def load_even_frames_from_video(video_path: str, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return []
        idxs = evenly_spaced_indices(len(frames), max_frames)
        return [
            Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in idxs
        ]
    idxs = evenly_spaced_indices(total, max_frames)
    imgs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return imgs


@torch.no_grad()
def feats_for_pils(pils, model, tfm, batch, device):
    outs = []
    use_amp = device.type == "cuda"
    for i in range(0, len(pils), batch):
        chunk = pils[i : i + batch]
        xb = torch.stack([tfm(im) for im in chunk], 0).to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                f = model(xb)
        else:
            f = model(xb)
        if f.ndim == 4:
            f = torch.flatten(nn.AdaptiveAvgPool2d(1)(f), 1)
        outs.append(f.float().cpu().numpy())
    return np.concatenate(outs, 0) if outs else None


@torch.no_grad()
def feats_for_images(paths, model, tfm, batch, device):
    pils = []
    for p in paths:
        try:
            with Image.open(p) as im:
                pils.append(im.convert("RGB"))
        except Exception as e:
            print(f"[warn] load {p}: {e}")
    if not pils:
        return None
    return feats_for_pils(pils, model, tfm, batch, device)


@torch.no_grad()
def feats_for_video_file(video_path, model, tfm, batch, device, max_frames):
    pils = load_even_frames_from_video(video_path, max_frames)
    if not pils:
        return None
    return feats_for_pils(pils, model, tfm, batch, device)


def main():
    ap = argparse.ArgumentParser("Extract 2048-D features (balanced, alias-aware)")
    ap.add_argument(
        "--frames_root",
        default=DEFAULT_INPUT,
        help="Root with class dirs and/or video files inside each class dir",
    )
    ap.add_argument("--features_out", default=DEFAULT_OUTPUT)
    ap.add_argument("--img_size", type=int, default=DEF_IMG_SIZE)
    ap.add_argument("--max_frames", type=int, default=DEF_MAX_FRAMES)
    ap.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument(
        "--classes", default="", help="Subset to include (aliases ok), e.g. 'real,fake'"
    )
    ap.add_argument("--max_videos_per_class", type=int, default=DEF_PER_CLASS_CAP)
    ap.add_argument("--max_videos_total", type=int, default=DEF_TOTAL_CAP)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.features_out, exist_ok=True)

    # feature backbone
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    tfm = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.isdir(args.frames_root):
        print(f"[error] frames_root not found: {args.frames_root}")
        sys.exit(1)

    raw_classes = [
        d
        for d in os.listdir(args.frames_root)
        if os.path.isdir(os.path.join(args.frames_root, d))
    ]
    raw_classes.sort(key=natural_key)

    # optional user filter
    if args.classes:
        want = {c.strip().lower() for c in args.classes.split(",") if c.strip()}
        raw_classes = [c for c in raw_classes if c.lower() in want]

    # map to fake/real and keep per-canonical buckets
    buckets = {"fake": [], "real": []}
    ignored = []
    for d in raw_classes:
        canon = norm_label(d)
        if canon is None:
            ignored.append(d)
            continue
        in_dir = os.path.join(args.frames_root, d)
        # collect items: subdirs (frames) or videos
        subdirs = [
            sd for sd in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, sd))
        ]
        videos = [
            vf
            for vf in os.listdir(in_dir)
            if os.path.isfile(os.path.join(in_dir, vf))
            and vf.lower().endswith(VIDEO_EXTS)
        ]
        items = [("dir", sd) for sd in subdirs] + [("video", vf) for vf in videos]
        items.sort(key=lambda x: natural_key(x[1]))
        buckets[canon].extend([(d, kind, name) for (kind, name) in items])

    n_fake, n_real = len(buckets["fake"]), len(buckets["real"])
    print(f"Discovered -> fake:{n_fake}  real:{n_real}  (ignored: {ignored})")
    if n_fake == 0 or n_real == 0:
        print(
            "❌ Need at least one item in BOTH fake and real. Fix your dataset layout."
        )
        sys.exit(1)

    rng = random.Random(42)
    if args.shuffle:
        rng.shuffle(buckets["fake"])
        rng.shuffle(buckets["real"])

    # per-class cap
    def cap_list(lst, cap):
        return lst[:cap] if cap and cap > 0 else lst

    fake_capped = cap_list(buckets["fake"], args.max_videos_per_class)
    real_capped = cap_list(buckets["real"], args.max_videos_per_class)

    if args.max_videos_total and args.max_videos_total > 0:
        half = max(1, args.max_videos_total // 2)
        fake_capped = fake_capped[:half]
        real_capped = real_capped[: args.max_videos_total - len(fake_capped)]

    plan = {"fake": fake_capped, "real": real_capped}
    print(
        f"Plan -> fake:{len(fake_capped)}  real:{len(real_capped)}  total:{len(fake_capped)+len(real_capped)}"
    )

    total_saved = 0
    for canon in ("fake", "real"):
        out_dir = os.path.join(args.features_out, canon)
        os.makedirs(out_dir, exist_ok=True)
        items = plan[canon]
        print(f"[class {canon}] processing {len(items)} items -> {out_dir}")

        for i, (orig_class, kind, name) in enumerate(items, 1):
            in_dir = os.path.join(args.frames_root, orig_class)
            if kind == "dir":
                src_folder = os.path.join(in_dir, name)
                dst_file = os.path.join(out_dir, f"{name}.npy")
                if args.skip_existing and os.path.isfile(dst_file):
                    if i % 10 == 0 or i == len(items):
                        print(f"  - {i}/{len(items)} (skip existing)")
                    continue
                frame_paths = select_frame_paths(src_folder, args.max_frames)
                feats = (
                    feats_for_images(frame_paths, model, tfm, args.batch_size, device)
                    if frame_paths
                    else None
                )
                if feats is None:
                    vids = [
                        vf
                        for vf in os.listdir(src_folder)
                        if os.path.isfile(os.path.join(src_folder, vf))
                        and vf.lower().endswith(VIDEO_EXTS)
                    ]
                    if vids:
                        feats = feats_for_video_file(
                            os.path.join(src_folder, vids[0]),
                            model,
                            tfm,
                            args.batch_size,
                            device,
                            args.max_frames,
                        )
            else:
                video_path = os.path.join(in_dir, name)
                stem, _ = os.path.splitext(name)
                dst_file = os.path.join(out_dir, f"{stem}.npy")
                if args.skip_existing and os.path.isfile(dst_file):
                    if i % 10 == 0 or i == len(items):
                        print(f"  - {i}/{len(items)} (skip existing)")
                    continue
                feats = feats_for_video_file(
                    video_path, model, tfm, args.batch_size, device, args.max_frames
                )

            if feats is None or (hasattr(feats, "size") and feats.size == 0):
                print(f"  [skip] {kind}:{name} (no usable frames)")
                continue

            try:
                np.save(dst_file, feats, allow_pickle=False)
                total_saved += 1
            except Exception as e:
                print(f"  [error] saving {dst_file}: {e}")

            if i % 10 == 0 or i == len(items):
                print(f"  - {i}/{len(items)} saved")

        print(f"[✓] {canon:<4} -> {out_dir}")

    print(f"\nAll done. Total features saved: {total_saved}  Device: {device}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    main()

import os
import random
import shutil
import pathlib
from typing import Dict, List


# --- Path portability (project-relative) ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "Dataset")

CLASS_SOURCES = {
    "real": os.path.join(DATASET_DIR, "Celeb-real"),
    "deepfakes": os.path.join(DATASET_DIR, "Celeb-synthesis"),
}

OUTPUT_BASE = os.path.join(PROJECT_DIR, "processed_dataset")
# LIMITS
PER_CLASS_LIMITS: Dict[str, int] = {
    "real": 120,
    "deepfakes": 120,
    "fake": 120,
}
DEFAULT_LIMIT = 120

# File types we’ll copy
EXTS = {".mp4", ".avi", ".mkv", ".mov"}
SEED = 42  # for reproducible sampling
SHUFFLE = True  # shuffle before selecting k videos


def list_videos(src_dir: str) -> List[str]:
    if not os.path.isdir(src_dir):
        print(f"[skip] {src_dir} (missing)")
        return []
    vids = [f for f in os.listdir(src_dir) if pathlib.Path(f).suffix.lower() in EXTS]
    vids.sort()
    return vids


def sample_and_copy(src_dir: str, dst_dir: str, k: int) -> List[str]:
    vids = list_videos(src_dir)
    if not vids:
        print(f"[warn] no videos in {src_dir}")
        return []

    os.makedirs(dst_dir, exist_ok=True)

    # shuffle
    if SHUFFLE:
        random.shuffle(vids)
    picked = vids[: min(k, len(vids))]

    copied = []
    for v in picked:
        src = os.path.join(src_dir, v)
        dst = os.path.join(dst_dir, v)
        try:
            shutil.copy(src, dst)
            copied.append(v)
        except Exception as e:
            print(f"[error] copying {src} -> {dst}: {e}")

    return copied


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    total = 0
    for cls, src in CLASS_SOURCES.items():
        cap = PER_CLASS_LIMITS.get(cls, DEFAULT_LIMIT)
        dst = os.path.join(OUTPUT_BASE, cls)
        copied = sample_and_copy(src, dst, cap)
        total += len(copied)
        print(f"[{cls:12}] copied {len(copied):3d}/{cap:3d} -> {dst}")

    print(f"\n[✓] Done. Total videos copied: {total}  -> {OUTPUT_BASE}")


if __name__ == "__main__":
    main()

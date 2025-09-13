# extract_frames_2.py — fast, resume-friendly, evenly spaced frame sampling

import os
import sys
import math
import cv2
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# ----------- Paths (project-relative, portable) -----------------
PROJECT_DIR = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_DIR / "processed_dataset"  # from sample_1.py
OUT_ROOT = PROJECT_DIR / "extracted_frames"

# ----------- Frame extraction knobs -------------------------
IM_SIZE = 112
FRAMES_PER_VIDEO = 24
EVENLY_SPACED = True
MAX_WORKERS = 4
SAVE_JPG_QUALITY = 95
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

# ----------- LIMITS (the important part) --------------------
# DEFAULT_CAP_PER_CLASS is used.
PER_CLASS_LIMITS: Dict[str, int] = {
    "real": 300,
    "deepfakes": 300,
    "fake": 300,
}
DEFAULT_CAP_PER_CLASS = 200

ALPHABETICAL = True

# ------------------------------------------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_class_dirs(src_root: Path) -> List[Path]:
    return sorted([d for d in src_root.iterdir() if d.is_dir()])


def list_videos_for_class(cls_dir: Path, cap: int) -> List[Path]:
    vids = [p for p in cls_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    vids.sort(key=lambda p: p.name)
    if not ALPHABETICAL:
        vids = list(reversed(vids))
    return vids[:cap]


def count_existing_frames(out_dir: Path) -> int:
    return len(list(out_dir.glob("img_*.jpg"))) + len(list(out_dir.glob("img_*.png")))


def sample_indices(n_total: int, n_take: int):
    """Evenly spaced indices in [0, n_total-1]."""
    if n_total <= 0:
        return []
    n_take = min(n_take, n_total)
    if n_take == 1:
        return [0]
    step = (n_total - 1) / (n_take - 1)
    return [int(round(i * step)) for i in range(n_take)]


def extract_evenly_spaced_frames(
    video_path: Path,
    out_dir: Path,
    n_frames: int = FRAMES_PER_VIDEO,
    im_size: int = IM_SIZE,
):

    ensure_dir(out_dir)

    # Resume-friendly: skip if already done
    existing = count_existing_frames(out_dir)
    if existing >= n_frames:
        return f"SKIP (exists {existing})"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return "ERROR: cannot open"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    collected = []

    try:
        if total <= 0:
            # fallback sequential read
            ok = True
            while ok and len(collected) < n_frames:
                ok, frame = cap.read()
                if ok:
                    collected.append(frame)
        else:
            idxs = (
                sample_indices(total, n_frames)
                if EVENLY_SPACED
                else list(range(n_frames))
            )
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    # try one sequential step
                    ok2, frame2 = cap.read()
                    if not ok2:
                        continue
                    frame = frame2
                collected.append(frame)
    finally:
        cap.release()

    if not collected:
        return "ERROR: no frames collected"

    # pad by repeating last frame if short
    while len(collected) < n_frames and collected:
        collected.append(collected[-1])

    # save
    for i, bgr in enumerate(collected[:n_frames], start=1):
        resized = cv2.resize(bgr, (im_size, im_size), interpolation=cv2.INTER_AREA)
        out_file = out_dir / f"img_{i:04d}.jpg"
        ok = cv2.imwrite(
            str(out_file), resized, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPG_QUALITY]
        )
        if not ok:
            print(f"[warn] failed to write frame {i} -> {out_file}")

    return f"OK ({len(collected[:n_frames])} saved)"


def process_one(class_name: str, video_path: Path) -> str:
    out_dir = OUT_ROOT / class_name / video_path.stem
    try:
        status = extract_evenly_spaced_frames(video_path, out_dir)
        return f"[{class_name:11}] {video_path.stem} -> {status}"
    except Exception:
        return f"[{class_name:11}] {video_path.stem} -> EXCEPTION:\n{traceback.format_exc()}"


def process_all():
    ensure_dir(OUT_ROOT)

    class_dirs = iter_class_dirs(SRC_ROOT)
    jobs: List[Tuple[str, Path]] = []

    for cls_dir in class_dirs:
        cls = cls_dir.name
        cap = PER_CLASS_LIMITS.get(cls, DEFAULT_CAP_PER_CLASS)
        vids = list_videos_for_class(cls_dir, cap)
        if not vids:
            print(f"[skip] {cls:11} (no videos)")
            continue
        for v in vids:
            jobs.append((cls, v))
        print(f"[plan] {cls:11} -> will process {len(vids):3d}/{cap:3d} videos")

    total = len(jobs)
    if total == 0:
        print(f"No videos found under: {SRC_ROOT}")
        return

    print(f"\nPlanned total: {total} videos from {SRC_ROOT}")
    print(
        f"Frames/video: {FRAMES_PER_VIDEO} | Output: {OUT_ROOT} | Workers: {MAX_WORKERS}"
    )
    print("-" * 60)

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut2job = {ex.submit(process_one, c, v): (c, v) for (c, v) in jobs}
            for fut in as_completed(fut2job):
                msg = fut.result()
                done += 1
                print(f"[{done}/{total}] {msg}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial progress is saved. Re-run to resume.")
        sys.exit(1)

    print("\n[✓] All done.")


if __name__ == "__main__":
    try:
        process_all()
    except Exception:
        print("FATAL ERROR:")
        traceback.print_exc()
        sys.exit(2)

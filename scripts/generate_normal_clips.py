"""
Generate Normal/Background training clips for the 3-class MoViNet model.

Strategy: Extract the first and last ~1.5 seconds from every existing
PedestrianLittering and VehicleLittering video. These frames capture
"before the event" and "after the event" footage — normal walking, normal
driving, empty roads, etc. — which is exactly what the Normal class needs.

Output: New_SawnDataset/NewTraining_Normal/  (~200 clips)
        New_SawnDataset/NewTest_Normal/       (~30 clips for evaluation)

Usage:
    python scripts/generate_normal_clips.py --data_dir "d:/final react/New_SawnDataset"
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# The training videos are short action clips (typically 50-100 frames at 30fps).
# We extract 16-frame clips (= the exact model input length) from the outer edges.
CLIP_DURATION_FRAMES = 16       # matches MoViNetClassifier.N_FRAMES exactly
MIN_SRC_FRAMES = 36             # at least 36 frames: 16 pre + 4 safety + 16 post
# Skip this fraction of the video from the centre so we don't accidentally
# capture the actual littering action in the Normal clips.
SAFETY_SKIP_FRACTION = 0.20     # skip inner 20% (narrowed for short clips)

SOURCE_FOLDERS = ["NewTraining_PedestrianLittering", "NewTraining_VehicleLittering"]
TEST_FRACTION  = 0.12  # ~12% of generated clips go to NewTest_Normal


def extract_clip(cap, start_frame: int, n_frames: int, out_path: str, fps: int) -> bool:
    """Extract n_frames starting at start_frame and save as MP4. Returns True on success."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return False

    written = 0
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
    writer.release()
    return written >= n_frames // 2  # accept if at least half were written


def process_video(src_path: str, out_dir: Path, base_name: str) -> list[str]:
    """Extract pre- and post-event clips from src_path. Returns list of saved paths."""
    cap = cv2.VideoCapture(src_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    saved = []

    if total < MIN_SRC_FRAMES:
        cap.release()
        return saved

    safety_skip = int(total * SAFETY_SKIP_FRACTION)
    safe_end_pre   = max(0, total // 2 - safety_skip)   # end of pre-event safe zone
    safe_start_post = min(total, total // 2 + safety_skip)  # start of post-event safe zone

    # ── Pre-event clip (beginning of video) ──────────────────────────────────
    pre_avail = safe_end_pre
    if pre_avail >= CLIP_DURATION_FRAMES:
        # Start randomly in the first pre-event block
        max_start = pre_avail - CLIP_DURATION_FRAMES
        start = random.randint(0, max(0, max_start))
        out_path = str(out_dir / f"{base_name}_pre.mp4")
        if extract_clip(cap, start, CLIP_DURATION_FRAMES, out_path, fps):
            saved.append(out_path)

    # ── Post-event clip (end of video) ────────────────────────────────────────
    post_avail = total - safe_start_post
    if post_avail >= CLIP_DURATION_FRAMES:
        max_start = total - CLIP_DURATION_FRAMES
        start = random.randint(safe_start_post, max(safe_start_post, max_start))
        out_path = str(out_dir / f"{base_name}_post.mp4")
        if extract_clip(cap, start, CLIP_DURATION_FRAMES, out_path, fps):
            saved.append(out_path)

    cap.release()
    return saved


def main(args):
    data_dir = Path(args.data_dir)
    train_out = data_dir / "NewTraining_Normal"
    test_out  = data_dir / "NewTest_Normal"
    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    # Collect all source videos
    all_src = []
    for folder in SOURCE_FOLDERS:
        src_dir = data_dir / folder
        if not src_dir.exists():
            print(f"[WARN] Not found: {src_dir}")
            continue
        vids = sorted(src_dir.glob("*.mp4")) + sorted(src_dir.glob("*.avi"))
        all_src.extend(vids)
        print(f"  [{folder}]  {len(vids)} source videos")

    print(f"\n[INFO] Total source videos: {len(all_src)}")
    print(f"[INFO] Extracting pre/post-event Normal clips (each up to 2 clips per video)...")

    all_clips: list[str] = []
    for i, src in enumerate(all_src):
        base = src.stem
        clips = process_video(str(src), train_out, f"Normal_{i:04d}_{base[:20]}")
        all_clips.extend(clips)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_src)} → {len(all_clips)} clips so far...")

    print(f"\n[INFO] Generated {len(all_clips)} Normal clips total")

    # ── Move a fraction to test set ───────────────────────────────────────────
    random.shuffle(all_clips)
    n_test = max(10, int(len(all_clips) * TEST_FRACTION))
    for clip_path in all_clips[:n_test]:
        dst = test_out / Path(clip_path).name
        shutil.move(clip_path, str(dst))

    remaining = len(list(train_out.glob("*.mp4")))
    test_size  = len(list(test_out.glob("*.mp4")))
    print(f"\n[DONE]  Training clips: {remaining}  →  {train_out}")
    print(f"        Test clips:     {test_size}   →  {test_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="d:/final react/New_SawnDataset",
        help="Root dataset directory containing NewTraining_* folders",
    )
    args = parser.parse_args()
    main(args)

"""
Video Augmentation Utilities
Matches paper: flip, multiply, Gaussian blur, brightness/darkness.

Dataset folder names:
  Source : NewTraining_PedestrianLittering/  NewTraining_VehicleLittering/
  Output : NewTraining_PedestrianLittering/  NewTraining_VehicleLittering/  (inside output_dir)

Usage:
    python utils/augmentation.py \
        --input  D:/New_SawnDataset-main \
        --output D:/New_SawnDataset-main/augmented \
        --n 3
"""

import cv2
import numpy as np
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# ── Exact training folder names ───────────────────────────────────────────────
TRAIN_FOLDERS = [
    "NewTraining_PedestrianLittering",
    "NewTraining_VehicleLittering",
]


# ── Frame-level augmentations ─────────────────────────────────────────────────

def flip_horizontal(f): return cv2.flip(f, 1)
def flip_vertical(f):   return cv2.flip(f, 0)
def flip_both(f):       return cv2.flip(f, -1)

def gaussian_blur(f, k=5):
    return cv2.GaussianBlur(f, (k, k), 0)

def brighten(f, factor=1.3):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def darken(f):  return brighten(f, 0.6)
def multiply(f, factor=1.2):
    return np.clip(f.astype(np.float32) * factor, 0, 255).astype(np.uint8)

AUGMENTATIONS = {
    "hflip":  flip_horizontal,
    "vflip":  flip_vertical,
    "bflip":  flip_both,
    "blur":   gaussian_blur,
    "bright": brighten,
    "dark":   darken,
    "mult":   multiply,
}


# ── Per-video augmentation ────────────────────────────────────────────────────

def augment_video(src: Path, dst: Path, aug_fn, target_size=(1920, 1080)):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, target_size)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = aug_fn(frame)
        out.write(frame)
    cap.release()
    out.release()


# ── Dataset-level augmentation ────────────────────────────────────────────────

def augment_dataset(input_dir: Path, output_dir: Path, n_augs: int = 3, seed: int = 42):
    """
    For each class folder in TRAIN_FOLDERS:
      1. Copy originals to output_dir/<class>/
      2. Apply n_augs random augmentations per video
    """
    random.seed(seed)
    aug_names = list(AUGMENTATIONS.keys())

    found_any = False
    for folder_name in TRAIN_FOLDERS:
        src_cls  = input_dir / folder_name
        dst_cls  = output_dir / folder_name

        if not src_cls.exists():
            print(f"[SKIP] Not found: {src_cls}")
            continue

        found_any = True
        dst_cls.mkdir(parents=True, exist_ok=True)
        videos = sorted(src_cls.glob("*.mp4")) + sorted(src_cls.glob("*.avi"))
        print(f"\n[AUG] {folder_name}  ({len(videos)} source videos × {n_augs} augs)")

        for vid in tqdm(videos, desc=folder_name):
            # 1. Copy original
            shutil.copy2(vid, dst_cls / vid.name)
            # 2. Apply N random augmentations
            chosen = random.sample(aug_names, min(n_augs, len(aug_names)))
            for aug_name in chosen:
                out_name = f"{vid.stem}_{aug_name}.mp4"
                augment_video(vid, dst_cls / out_name, AUGMENTATIONS[aug_name])

    if not found_any:
        print("[ERROR] No training folders found in:", input_dir)
        print("        Expected:", TRAIN_FOLDERS)
        return

    print("\n[DONE] Augmentation complete.")
    for folder_name in TRAIN_FOLDERS:
        dst_cls = output_dir / folder_name
        if dst_cls.exists():
            count = len(list(dst_cls.glob("*.mp4")))
            print(f"  {folder_name}: {count} total videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment SAWN video dataset")
    parser.add_argument("--input",  default="D:/New_SawnDataset-main",
                        help="Dataset root (containing NewTraining_* folders)")
    parser.add_argument("--output", default="D:/New_SawnDataset-main/augmented",
                        help="Output directory for augmented dataset")
    parser.add_argument("--n",      type=int, default=3,
                        help="Number of augmentations per video")
    args = parser.parse_args()
    augment_dataset(Path(args.input), Path(args.output), n_augs=args.n)

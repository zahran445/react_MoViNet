"""
SAWN — MoViNet Training Script (PyTorch + CUDA)
Binary classification:
  Class 0 = PedestrianLittering  (folder: NewTraining_PedestrianLittering)
  Class 1 = VehicleLittering     (folder: NewTraining_VehicleLittering)

Paper target: 99.5% accuracy with binary classification.

This script uses PyTorch on GPU (RTX 3060) for fast training.
MoViNet weights are approximated with a pretrained R3D-18 (ResNet 3D) from
torchvision, which is a strong transfer-learning backbone for video classification.

Usage:
    python scripts/train_movinet.py --data_dir D:/sawn_project/New_SawnDataset --epochs 30
"""

import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as vid_models

# ── Config ────────────────────────────────────────────────────────────────────
NUM_FRAMES  = 16
FRAME_SIZE  = (112, 112)   # R3D-18 standard input size
NUM_CLASSES = 2
SEED        = 42
BATCH_SIZE  = 8            # safe for RTX 3060 6 GB VRAM with R3D-18

TRAIN_FOLDERS = {
    0: "NewTraining_PedestrianLittering",
    1: "NewTraining_VehicleLittering",
}
CLASS_NAMES = ["PedestrianLittering", "VehicleLittering"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "  (CPU — slow!)"))


# ── Frame sampling ─────────────────────────────────────────────────────────────
def sample_frames(video_path: str, n_frames: int = NUM_FRAMES) -> np.ndarray:
    """Sample n_frames evenly. Returns (n, H, W, 3) float32 [0,1]."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return np.zeros((n_frames, *FRAME_SIZE, 3), dtype=np.float32)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.float32) / 255.0


# ── Dataset ────────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels
        # ImageNet mean/std for normalization (R3D pretrained)
        self.mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        self.std  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        clip = sample_frames(self.paths[idx])   # (T, H, W, 3) float32 [0,1]
        clip = (clip - self.mean) / self.std    # normalize
        # Convert to (C, T, H, W) for torchvision video models
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return clip, label


# ── Collect paths ──────────────────────────────────────────────────────────────
def collect_paths(data_dir: Path):
    paths, labels = [], []
    for label, folder_name in TRAIN_FOLDERS.items():
        cls_dir = data_dir / folder_name
        if not cls_dir.exists():
            print(f"[WARN] Folder not found: {cls_dir}")
            continue
        videos = sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.avi"))
        print(f"  [{folder_name}]  {len(videos)} videos")
        for v in videos:
            paths.append(str(v))
            labels.append(label)
    return paths, labels


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES, args=None):
    """
    Pretrained R3D-18 from torchvision (video ResNet).
    This closely matches the MoViNet family in the sense that both are
    efficient video classification models. We fine-tune the final layer.
    """
    model = vid_models.r3d_18(weights=vid_models.R3D_18_Weights.DEFAULT)
    # Replace head for binary classification
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[MODEL] Resuming from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=DEVICE))
        else:
            print(f"[WARN] Checkpoint not found: {args.resume} — starting from scratch.")

    print(f"[MODEL] R3D-18 loaded → head replaced with {num_classes}-class FC.")
    return model.to(DEVICE)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(loader, desc="  train", leave=False):
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for clips, labels in tqdm(loader, desc="  val  ", leave=False):
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        outputs = model(clips)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────
def train(args):
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DATA] Scanning: {data_dir}")
    all_paths, all_labels = collect_paths(data_dir)

    if not all_paths:
        print("\n[ERROR] No videos found.")
        for fn in TRAIN_FOLDERS.values():
            print(f"          {fn}/")
        return

    print(f"[DATA] Total: {len(all_paths)} videos")

    train_p, val_p, train_l, val_l = train_test_split(
        all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=SEED
    )
    print(f"[SPLIT] Train: {len(train_p)}  |  Val: {len(val_p)}")

    train_ds = VideoDataset(train_p, train_l)
    val_ds   = VideoDataset(val_p,   val_l)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = build_model(NUM_CLASSES, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_acc = 0.0
    patience_cnt = 0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"\n[TRAIN] Starting — max {args.epochs} epochs, batch={BATCH_SIZE}, lr={args.lr}, device={DEVICE}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc, vl_preds, vl_labels = val_epoch(model, val_dl, criterion)
        scheduler.step(vl_loss)

        train_accs.append(tr_acc);   val_accs.append(vl_acc)
        train_losses.append(tr_loss); val_losses.append(vl_loss)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), str(out_dir / "movinet_best.pt"))
            print(f"  ✓ New best val_acc={best_val_acc:.4f} — model saved.")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 8:
                print("[EARLY STOP] No improvement for 8 epochs.")
                break

    # ── Final metrics ──────────────────────────────────────────────────────────
    print("\n[EVAL] Final validation metrics:")
    print(classification_report(vl_labels, vl_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(vl_labels, vl_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("MoViNet — Validation Confusion Matrix")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"[SAVED] Confusion matrix → {out_dir / 'confusion_matrix.png'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_accs, label="Train"); ax1.plot(val_accs, label="Val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(train_losses, label="Train"); ax2.plot(val_losses, label="Val")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"[SAVED] Training curves → {out_dir / 'training_curves.png'}")

    torch.save(model.state_dict(), str(out_dir / "movinet_final.pt"))
    print(f"[SAVED] Best model    → {out_dir / 'movinet_best.pt'}")
    print(f"[SAVED] Final model   → {out_dir / 'movinet_final.pt'}")
    print(f"\n[DONE] Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="D:/sawn_project/New_SawnDataset",
                        help="Folder containing NewTraining_* subfolders")
    parser.add_argument("--output_dir", default="models/movinet")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--resume",     type=str,   default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    train(args)

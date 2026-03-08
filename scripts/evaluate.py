"""
SAWN Model Evaluation (PyTorch)
Evaluates trained R3D-18 on the test set using:
  NewTest_PedestrianLittering/
  NewTest_VehicleLittering/

Outputs: confusion matrix PNG, per-video CSV, summary TXT.

Usage:
    python scripts/evaluate.py \
        --model   models/movinet/movinet_best.pt \
        --test_dir D:/sawn_project/New_SawnDataset
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

import torch
import torch.nn as nn
import torchvision.models.video as vid_models

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FOLDERS = {
    0: "NewTest_PedestrianLittering",
    1: "NewTest_VehicleLittering",
}
CLASS_NAMES = ["PedestrianLittering", "VehicleLittering"]
NUM_FRAMES  = 16
FRAME_SIZE  = (112, 112)
NUM_CLASSES = 2

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((*FRAME_SIZE, 3), np.uint8)
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    clip = np.stack(frames).astype(np.float32) / 255.0
    clip = (clip - MEAN) / STD
    # (T, H, W, C) -> (1, C, T, H, W)
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float()
    return clip


def load_model(model_path: str) -> nn.Module:
    model = vid_models.r3d_18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"[MODEL] Loaded PyTorch R3D-18 from {model_path}")
    print(f"[DEVICE] Inference on: {DEVICE}")
    return model


def evaluate(args):
    model    = load_model(args.model)
    test_dir = Path(args.test_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, records = [], [], []

    with torch.no_grad():
        for label, folder_name in TEST_FOLDERS.items():
            cls_dir = test_dir / folder_name
            if not cls_dir.exists():
                print(f"[WARN] Not found: {cls_dir}  — skipping.")
                continue
            videos = sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.avi"))
            print(f"\n[TEST] {folder_name}  ({len(videos)} videos)")

            for vid in tqdm(videos, desc=folder_name):
                clip  = sample_frames(str(vid)).to(DEVICE)
                logits = model(clip)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred   = int(np.argmax(probs))
                conf   = float(probs[pred])
                y_true.append(label)
                y_pred.append(pred)
                records.append({
                    "video":      vid.name,
                    "true_label": CLASS_NAMES[label],
                    "pred_label": CLASS_NAMES[pred],
                    "confidence": round(conf * 100, 2),
                    "correct":    label == pred,
                })

    if not y_true:
        print("[ERROR] No test videos found. Check --test_dir.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("  Total test videos:", len(y_true))
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Test Confusion Matrix  (Acc: {acc*100:.2f}%)")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    cm_path = out_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"[SAVED] {cm_path}")

    csv_path = out_dir / "evaluation_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    summary_path = out_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model   : {args.model}\n")
        f.write(f"Test dir: {args.test_dir}\n")
        f.write(f"Device  : {DEVICE}\n")
        f.write(f"Videos  : {len(y_true)}\n\n")
        f.write(f"Accuracy  : {acc*100:.2f}%\n")
        f.write(f"Precision : {prec*100:.2f}%\n")
        f.write(f"Recall    : {rec*100:.2f}%\n")
        f.write(f"F1-Score  : {f1*100:.2f}%\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"[SAVED] {summary_path}")
    print(f"\n[DONE] Evaluation complete. All outputs in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="models/movinet/movinet_best.pt")
    parser.add_argument("--test_dir",   default="D:/sawn_project/New_SawnDataset",
                        help="Folder containing NewTest_* subfolders")
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()
    evaluate(args)

"""
YOLOv8 License Plate Detection - Training Script
Trains YOLOv8n on the KSA license plate dataset (or any YOLO-format dataset).
Paper result: 99.5% accuracy on KSA plates.

Dataset format expected (YOLO):
  data/plates/
    images/train/   *.jpg
    images/val/     *.jpg
    labels/train/   *.txt   (YOLO format: class cx cy w h)
    labels/val/     *.txt
    dataset.yaml

Usage:
    python scripts/train_yolov8.py --data data/plates/dataset.yaml --epochs 50
"""

import argparse
from pathlib import Path
import yaml


# ── Dataset YAML generator ─────────────────────────────────────────────────

def create_dataset_yaml(data_dir: Path, output_path: Path):
    """Generate a YOLO dataset.yaml if one doesn't exist."""
    cfg = {
        "path":  str(data_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": {0: "license_plate"},
    }
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[YAML] Dataset config saved → {output_path}")
    return str(output_path)


# ── Training ───────────────────────────────────────────────────────────────

def train(args):
    from ultralytics import YOLO

    data_dir  = Path(args.data).parent if args.data.endswith(".yaml") else Path(args.data)
    yaml_path = args.data if args.data.endswith(".yaml") else str(data_dir / "dataset.yaml")

    # Auto-generate YAML if it doesn't exist
    if not Path(yaml_path).exists():
        print(f"[WARN] dataset.yaml not found at {yaml_path}")
        print("[INFO] Generating dataset.yaml from directory structure...")
        yaml_path = create_dataset_yaml(data_dir, Path(yaml_path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[MODEL] Loading YOLOv8{args.model_size}...")
    model = YOLO(f"yolov8{args.model_size}.pt")  # auto-downloads pretrained weights

    print(f"[TRAIN] Starting training | epochs={args.epochs} | imgsz={args.imgsz}")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,                     # GPU 0 (RTX 3060)
        project=str(out_dir),
        name="plates_yolov8",
        patience=15,                  # early stopping
        save=True,
        save_period=5,
        plots=True,
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        augment=True,
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
    )

    # Validate
    print("\n[EVAL] Running validation...")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.mp:.4f}")
    print(f"  Recall:   {metrics.box.mr:.4f}")

    # Export to ONNX for faster inference
    export_path = out_dir / "plates_yolov8" / "weights" / "best.pt"
    if export_path.exists():
        print(f"\n[EXPORT] Exporting to ONNX...")
        export_model = YOLO(str(export_path))
        export_model.export(format="onnx", imgsz=args.imgsz)
        print(f"[SAVED] ONNX model → {export_path.with_suffix('.onnx')}")

    print("[DONE] YOLOv8 training complete.")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for license plate detection")
    parser.add_argument("--data",       default="data/plates/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--output_dir", default="models/yolov8",            help="Output directory")
    parser.add_argument("--epochs",     type=int,   default=50,             help="Training epochs")
    parser.add_argument("--imgsz",      type=int,   default=640,            help="Image size")
    parser.add_argument("--batch",      type=int,   default=16,             help="Batch size")
    parser.add_argument("--model_size", default="n",                        help="YOLOv8 size: n/s/m/l/x")
    args = parser.parse_args()
    train(args)

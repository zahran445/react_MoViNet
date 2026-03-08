# SAWN — Surveillance and Waste Notification System

> Reproduction of: *Alharbi et al., Scientific Reports (2025)*  
> [DOI: 10.1038/s41598-024-77118-x](https://doi.org/10.1038/s41598-024-77118-x)

---

## Dataset Structure (Your Actual Folders)

```
New_SawnDataset-main/
├── NewTraining_PedestrianLittering/   ← training  (class 0)
├── NewTraining_VehicleLittering/      ← training  (class 1)  [420 videos]
├── NewTest_PedestrianLittering/       ← test      (class 0)
└── NewTest_VehicleLittering/          ← test      (class 1)
```

All scripts use these **exact** folder names — no renaming needed.

---

## System Architecture

```
Camera / Video
      │
      ▼
 ┌─────────┐   "PedestrianLittering"    ┌──────────────────┐
 │ MoViNet │ ──────────────────────────▶│  Haar Cascade    │
 │  (A2)   │                            │  (Face detect)   │
 └─────────┘   "VehicleLittering"       └──────────────────┘
      │                                          │
      │        ┌──────────────────┐              │
      └───────▶│    YOLOv8n       │              │
               │  (Plate detect)  │              │
               └──────────────────┘              │
                        │                        │
                        ▼                        ▼
                   SQLite DB  ◀──── SAWN Dashboard (Flask)
                        │
                        ▼
                 Send Ticket / Fine
```

---

## Project Structure

```
sawn_project/
├── scripts/
│   ├── train_movinet.py       Train MoViNet binary classifier
│   ├── train_yolov8.py        Train YOLOv8 for license plate detection
│   ├── evaluate.py            Evaluate on NewTest_* folders
│   └── run_inference.py       Run detection on video / folder / webcam
├── utils/
│   ├── augmentation.py        Augment NewTraining_* videos
│   └── detector.py            SAWNDetector class (full pipeline)
├── web/
│   └── app.py                 Flask dashboard
├── models/                    Saved model weights (auto-created)
├── outputs/violations/        Saved violation snapshots (auto-created)
├── requirements.txt
└── README.md
```

---

## Setup (Windows — RTX 3060 Laptop)

### 1. Create virtual environment
```bat
cd sawn_project
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch with CUDA 11.8 (RTX 3060)
```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install all other dependencies
```bat
pip install -r requirements.txt
```

### 4. Install MoViNet (optional but recommended)
```bat
pip install tf-models-official
```
> If this fails, the training script automatically falls back to a lightweight 3D-CNN.

---

## Step-by-Step Workflow

### Step 1 — Augment Training Data

Applies flips, Gaussian blur, brightness/darkness to training videos.

```bat
python utils/augmentation.py ^
    --input  D:\New_SawnDataset-main ^
    --output D:\New_SawnDataset-main\augmented ^
    --n 3
```

Output:
```
D:\New_SawnDataset-main\augmented\
├── NewTraining_PedestrianLittering\   (originals + augmented)
└── NewTraining_VehicleLittering\      (originals + augmented)
```

---

### Step 2 — Train MoViNet

```bat
python scripts/train_movinet.py ^
    --data_dir   D:\New_SawnDataset-main\augmented ^
    --output_dir models\movinet ^
    --epochs 30 ^
    --lr 1e-4
```

**Or skip augmentation and train on raw data:**
```bat
python scripts/train_movinet.py --data_dir D:\New_SawnDataset-main
```

Outputs in `models/movinet/`:
- `movinet_best.keras` — best checkpoint (by val accuracy)
- `movinet_final.keras` — final epoch model
- `confusion_matrix.png`
- `training_curves.png`

**Target: 99.5% accuracy** (paper Experiment 3 — binary classification)

---

### Step 3 — Train YOLOv8 for License Plates

You need a KSA license plate dataset in YOLO format.  
Get it from [Roboflow](https://roboflow.com) or [Kaggle](https://kaggle.com) — search "Saudi Arabia license plate YOLO".

Place it as:
```
data\plates\
├── images\train\   *.jpg
├── images\val\     *.jpg
├── labels\train\   *.txt
├── labels\val\     *.txt
└── dataset.yaml
```

Then train:
```bat
python scripts/train_yolov8.py ^
    --data   data\plates\dataset.yaml ^
    --epochs 50 ^
    --batch  16
```

Best weights → `models\yolov8\plates_yolov8\weights\best.pt`

---

### Step 4 — Evaluate on Test Set

```bat
python scripts/evaluate.py ^
    --model    models\movinet\movinet_best.keras ^
    --test_dir D:\New_SawnDataset-main
```

Scans `NewTest_PedestrianLittering/` and `NewTest_VehicleLittering/` automatically.

Outputs in `outputs/evaluation/`:
- `test_confusion_matrix.png`
- `evaluation_results.csv` — per-video predictions
- `evaluation_summary.txt`

---

### Step 5 — Run Inference

**Single video:**
```bat
python scripts/run_inference.py --source D:\clips\vehicle_test.mp4
```

**Entire test folder:**
```bat
python scripts/run_inference.py --source D:\New_SawnDataset-main\NewTest_VehicleLittering
```

**Webcam:**
```bat
python scripts/run_inference.py --source 0
```

**With live dashboard:**
```bat
:: Terminal 1
python web\app.py

:: Terminal 2
python scripts/run_inference.py --source 0 --log-to-web
```

---

### Step 6 — Launch Web Dashboard

```bat
python web\app.py
```

Open **http://localhost:5000**

| Page | Description |
|------|-------------|
| Home `/` | Stats cards + recent violations table |
| Live `/live` | Real-time webcam feed |
| Violations `/violations` | Full log with type filter |
| Tickets `/tickets` | Sent tickets to authorities |

---

## Docker Deployment

You can run the full REACT/SAWN dashboard inside Docker (no local Python/venv needed).

### 1. Build the image

```bash
docker build -t sawn-react .
```

### 2. Run the container (simple)

```bash
docker run --rm -p 5000:5000 sawn-react
```

Then open `http://localhost:5000` in your browser.

### 3. Using docker-compose (recommended, with persistent volumes)

```bash
docker compose up --build
```

This:

- Builds the image from `Dockerfile`.
- Maps port `5000` on your machine to the Flask app in the container.
- Mounts `./outputs` and `./models` so:
  - New violation clips/snapshots and `sawn.db` are kept on your host.
  - Your trained models in `models/` are visible to the container.

### 4. GPU acceleration (optional)

The default image is **CPU-only** for portability. For GPU use:

- Enable NVIDIA GPU support for Docker (Docker Desktop + WSL2 + `nvidia-container-toolkit`).
- Replace the base image in `Dockerfile` with an NVIDIA / PyTorch image, for example:

```Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
```

- Uncomment the `deploy.resources.reservations.devices` section in `docker-compose.yml` to request a GPU.

All other usage inside the container remains the same (uploads, dashboard, etc.).


## Performance Targets (from paper)

| Metric | Paper Result | Notes |
|--------|-------------|-------|
| MoViNet Accuracy | **99.5%** | Binary classification (Exp. 3) |
| MoViNet Precision | 98.9% | |
| MoViNet Recall | 97.8% | |
| MoViNet F1-Score | 98.35% | |
| YOLOv8 Plate Accuracy | **99.5%** | KSA plate dataset |

---

## Troubleshooting

**CUDA out of memory:**
```bat
python scripts/train_movinet.py --data_dir D:\New_SawnDataset-main --batch 2
```

**"No videos found" error:**  
Make sure `--data_dir` points to the folder that **contains** `NewTraining_PedestrianLittering/` — not inside it.

**MoViNet import error:**  
Normal — the script falls back to a 3D-CNN automatically. Install `tf-models-official` for the full MoViNet.

**OpenCV window missing (headless):**
```bat
python scripts/run_inference.py --source video.mp4 --no-preview
```

---

## Citation

```bibtex
@article{alharbi2025sawn,
  title={Real-time detection and monitoring of public littering behavior
         using deep learning for a sustainable environment},
  author={Alharbi, Eaman and Alsulami, Ghadah and Aljohani, Sarah
          and Alharbi, Waad and Albaradei, Somayah},
  journal={Scientific Reports},
  volume={15}, pages={3000}, year={2025},
  doi={10.1038/s41598-024-77118-x}
}
```

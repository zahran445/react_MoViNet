import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models.video as vid_models
from pathlib import Path
import easyocr
import time

# ── Class mapping (model output index → display name) ─────────────────────────
CLASS_NAMES = ["PedestrianLittering", "VehicleLittering"]

@dataclass
class Violation:
    id: int
    timestamp: str
    violation_type: str
    confidence: float
    snapshot: np.ndarray
    face_crop: Optional[np.ndarray] = None
    plate_crop: Optional[np.ndarray] = None
    plate_text: str = ""
    video_path: str = ""

# ── MoViNet Classifier (using R3D-18 as proxy) ───────────────────────────────

class MoViNetClassifier:
    N_FRAMES   = 16
    FRAME_SIZE = (112, 112)  # R3D-18 standard size

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        print(f"[MoViNet] Loaded (PyTorch): {model_path} on {self.device}")

    def _load_model(self, model_path: str):
        model = vid_models.r3d_18(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, 2)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"[MoViNet] Warning: Could not load weights: {e}")
        model.to(self.device)
        model.eval()
        return model

    def preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocesses a list of 16 frames for the model."""
        processed = []
        for f in frames:
            f = cv2.resize(f, self.FRAME_SIZE)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            processed.append(f)
            
        clip = np.stack(processed).astype(np.float32) / 255.0
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)
        clip = (clip - mean) / std
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float()
        return clip.to(self.device)

    @torch.no_grad()
    def predict_segment(self, frames: List[np.ndarray]) -> tuple[str, float]:
        if len(frames) < self.N_FRAMES:
            return CLASS_NAMES[0], 0.0
        logits = self.model(self.preprocess_clip(frames))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx    = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])

# ── Plate & Face Detectors ───────────────────────────────────────────────────

class PlateDetector:
    def __init__(self, model_path: str, conf: float = 0.5):
        self.model = None
        self.conf  = conf
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[YOLOv8] Plate detector loaded")
            except Exception as e:
                print(f"[YOLOv8] Warning: {e}")

    def detect(self, frame) -> Optional[tuple[np.ndarray, str]]:
        if not self.model: return None
        for r in self.model.predict(frame, conf=self.conf, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2].copy()
                try:
                    ocr_res = self.reader.readtext(plate_crop)
                    plate_text = "".join([text[1] for text in ocr_res if text[2] > 0.3]).upper()
                    plate_text = "".join(filter(str.isalnum, plate_text))
                    return plate_crop, plate_text
                except:
                    return plate_crop, ""
        return None

class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)

    def detect(self, frame) -> Optional[np.ndarray]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if not len(faces): return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return frame[y:y+h, x:x+w].copy()

# ── SAWNDetector ─────────────────────────────────────────────────────────────

class SAWNDetector:
    THRESHOLD = 0.50

    def __init__(self, movinet_path: str, yolo_path: str, output_dir: str = "outputs/violations"):
        self.classifier = MoViNetClassifier(movinet_path)
        self.plate_det  = PlateDetector(yolo_path)
        self.face_det   = FaceDetector()
        self.out_dir    = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter   = self._get_last_id()

    def _get_last_id(self) -> int:
        """Heuristic to get the next ID based on files in out_dir."""
        files = list(self.out_dir.glob("violation_*_snapshot.jpg"))
        if not files: return 0
        try:
            return max(int(f.name.split('_')[1]) for f in files)
        except:
            return 0

    def process_video(self, video_path: str) -> Optional[Violation]:
        print(f"[SAWN] Scanning video (Streaming Mode): {Path(video_path).name}")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 16:
            cap.release()
            print("  [ERROR] Video too short")
            return None

        # --- Temporal Scan Phase (Memory Efficient) ---
        max_conf = 0.0
        best_label = ""
        peak_frame_idx = 0
        
        step = max(1, fps // 2)
        window = []
        frame_idx = 0
        
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            window.append(frame)
            if len(window) > 16:
                window.pop(0)
            
            if len(window) == 16 and frame_idx % step == 0:
                label, conf = self.classifier.predict_segment(window)
                if conf > max_conf:
                    max_conf = conf
                    best_label = label
                    peak_frame_idx = frame_idx - 8
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  → Scanned {frame_idx}/{total_frames} frames...")
        
        cap.release()
        print(f"  → Scan complete in {time.time() - start_time:.1f}s")
        
        if max_conf < self.THRESHOLD:
            print(f"  [SKIP] No violation found (Max Conf: {max_conf:.1%})")
            return None

        print(f"  [DETECTED] {best_label} at frame {peak_frame_idx} ({max_conf:.1%})")
        
        self._counter += 1
        vtype = "Pedestrian" if "Pedestrian" in best_label else "Vehicle"
        
        # --- Asset Extraction Phase (Second Pass) ---
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
        ret, snapshot = cap.read()
        
        # Extract clip segment (6 seconds total: 3s before, 3s after peak)
        start_f = max(0, peak_frame_idx - (fps * 3))
        
        clip_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(int(fps * 6)):
            ret_c, f_c = cap.read()
            if not ret_c: break
            clip_frames.append(f_c)
        cap.release()

        # OCR / Face
        plate_crop, plate_text = None, ""
        if vtype == "Vehicle":
            res = self.plate_det.detect(snapshot)
            if res:
                plate_crop, plate_text = res

        # Save Clip
        clip_name = f"violation_{self._counter:04d}_clip.mp4"
        clip_path = self.out_dir / clip_name
        self._save_clip_segment(clip_frames, fps, str(clip_path))

        v = Violation(
            id             = self._counter,
            timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            violation_type = vtype,
            confidence     = max_conf,
            snapshot       = snapshot,
            face_crop      = self.face_det.detect(snapshot),
            plate_crop     = plate_crop,
            plate_text     = plate_text,
            video_path     = str(clip_path)
        )
        self._save_assets(v)
        return v

    def _save_clip_segment(self, frames: List[np.ndarray], fps: int, dst_path: str):
        if not frames: return
        h, w = frames[0].shape[:2]
        target_h = 480
        target_w = int(target_h * (w / h))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(dst_path, fourcc, fps, (target_w, target_h))
        if not out.isOpened():
            out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
            
        for f in frames:
            out.write(cv2.resize(f, (target_w, target_h)))
        out.release()

    def _save_assets(self, v: Violation):
        cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_snapshot.jpg"), v.snapshot)
        if v.face_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_face.jpg"), v.face_crop)
        if v.plate_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_plate.jpg"), v.plate_crop)

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
    def __init__(self, model_path: str, conf: float = 0.05):
        self.model = None
        self.conf  = conf
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[YOLOv8] Plate detector loaded (conf threshold: {conf})")
            except Exception as e:
                print(f"[YOLOv8] Warning: {e}")

    def detect(self, frame) -> Optional[tuple[np.ndarray, str]]:
        """
        Detects the strongest plate, expands the crop slightly, and applies
        OCR with rotation robustness for portrait-oriented snapshots.
        """
        if not self.model:
            return None

        # Try original, then rotation (common in portrait snapshots)
        orientations = [
            ("Original", frame),
            ("Rotated CW", cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)),
            ("Rotated CCW", cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]

        best_crop = None
        best_text = ""
        best_conf = 0.0

        for name, img in orientations:
            h, w = img.shape[:2]
            results = self.model.predict(img, conf=self.conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Expand box by 15% in each direction to avoid cutting characters
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.15)
                    ex1 = max(0, x1 - pad_x)
                    ey1 = max(0, y1 - pad_y)
                    ex2 = min(w, x2 + pad_x)
                    ey2 = min(h, y2 + pad_y)

                    crop = img[ey1:ey2, ex1:ex2].copy()
                    if crop.size == 0:
                        continue

                    try:
                        # Upscale small crops a bit for better OCR
                        ch, cw = crop.shape[:2]
                        if max(ch, cw) < 120:
                            scale = 120.0 / max(ch, cw)
                            crop = cv2.resize(
                                crop,
                                (int(cw * scale), int(ch * scale)),
                                interpolation=cv2.INTER_CUBIC,
                            )

                        ocr_res = self.reader.readtext(crop)
                        # Keep characters with lower confidence floor (some fonts are tricky)
                        chars = [t[1] for t in ocr_res if t[2] >= 0.1]
                        text = "".join(chars).upper()
                        # Filter to alphanumeric only
                        text = "".join(filter(str.isalnum, text))
                        
                        # If empty after filtering, try raw OCR text as fallback
                        if not text and ocr_res:
                            text = "".join([t[1] for t in ocr_res]).strip().upper()

                        # Prefer boxes with non‑empty OCR; break ties by YOLO score
                        if text and (score >= best_conf or not best_text):
                            best_conf = score
                            best_text = text
                            best_crop = crop
                    except Exception:
                        continue

        if best_crop is not None:
            return best_crop, best_text
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

    def process_video(self, video_path: str, progress_callback=None) -> Optional[Violation]:
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

            # Update progress callback if provided
            if progress_callback and total_frames > 0 and frame_idx % max(step, 5) == 0:
                pct = min(100, int((frame_idx / total_frames) * 100))
                try:
                    progress_callback(pct)
                except Exception:
                    # Callback issues should not break processing
                    pass
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  -> Scanned {frame_idx}/{total_frames} frames...")
        
        cap.release()
        print(f"  -> Scan complete in {time.time() - start_time:.1f}s")

        # Ensure we report completion if callback is provided
        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass
        
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
            # Try multiple frames to find the best plate view
            res = self._find_best_plate(clip_frames, fps, peak_frame_idx - start_f)
            if res:
                plate_crop, plate_text = res
            else:
                # Fallback: try the single snapshot
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

    def run_live(self, source: int = 0, show_preview: bool = True, callback: Optional[callable] = None) -> List[Violation]:
        print(f"[SAWN] Starting Live Feed (Source: {source})")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [ERROR] Could not open video source {source}")
            return []

        window = []
        violations = []
        last_violation_time = 0
        cooldown = 5 # seconds

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                window.append(frame)
                if len(window) > 16:
                    window.pop(0)

                if len(window) == 16:
                    label, conf = self.classifier.predict_segment(window)
                    
                    if conf > self.THRESHOLD and (time.time() - last_violation_time) > cooldown:
                        print(f"  [LIVE DETECTED] {label} ({conf:.1%})")
                        vtype = "Pedestrian" if "Pedestrian" in label else "Vehicle"
                        
                        self._counter += 1
                        snapshot = frame.copy()
                        
                        plate_crop, plate_text = None, ""
                        if vtype == "Vehicle":
                            res = self.plate_det.detect(snapshot)
                            if res: plate_crop, plate_text = res
                        
                        v = Violation(
                            id             = self._counter,
                            timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            violation_type = vtype,
                            confidence     = conf,
                            snapshot       = snapshot,
                            face_crop      = self.face_det.detect(snapshot),
                            plate_crop     = plate_crop,
                            plate_text     = plate_text
                        )
                        self._save_assets(v)
                        violations.append(v)
                        if callback:
                            callback(v)
                        last_violation_time = time.time()

                if show_preview:
                    cv2.imshow("SAWN Live Dashboard", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return violations

    def _find_best_plate(self, clip_frames: List[np.ndarray], fps: int, peak_idx: int) -> Optional[tuple[np.ndarray, str]]:
        """
        Sample multiple frames from the clip to find the best plate detection.
        Tries frames around the peak to find clearest plate view.
        """
        if not clip_frames:
            return None
        
        # Sample frames: peak frame and nearby frames (0.5s before/after)
        sample_indices = []
        half_sec_frames = max(1, fps // 2)
        
        # Add peak frame first
        if 0 <= peak_idx < len(clip_frames):
            sample_indices.append(peak_idx)
        
        # Add frames before and after peak
        for offset in range(1, half_sec_frames + 1):
            before_idx = peak_idx - offset
            after_idx = peak_idx + offset
            if 0 <= before_idx < len(clip_frames):
                sample_indices.append(before_idx)
            if 0 <= after_idx < len(clip_frames):
                sample_indices.append(after_idx)
        
        best_result = None
        best_score = 0.0
        
        for idx in sample_indices:
            if idx < 0 or idx >= len(clip_frames):
                continue
            
            frame = clip_frames[idx]
            result = self.plate_det.detect(frame)
            
            if result:
                crop, text = result
                # Score based on text length and non-empty text
                if text:
                    # Longer text generally means better OCR detection
                    score = len(text)
                    if score > best_score:
                        best_score = score
                        best_result = result
        
        return best_result

    def _save_clip_segment(self, frames: List[np.ndarray], fps: int, dst_path: str):
        if not frames: return
        h, w = frames[0].shape[:2]
        # Preserve original resolution for higher clarity
        target_w, target_h = w, h

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(dst_path, fourcc, fps, (target_w, target_h))
        if not out.isOpened():
            out = cv2.VideoWriter(
                dst_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (target_w, target_h),
            )

        for f in frames:
            out.write(f)
        out.release()

    def _save_assets(self, v: Violation):
        cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_snapshot.jpg"), v.snapshot)
        if v.face_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_face.jpg"), v.face_crop)
        if v.plate_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{v.id:04d}_plate.jpg"), v.plate_crop)

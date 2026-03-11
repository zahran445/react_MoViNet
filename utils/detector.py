import cv2
import easyocr
import numpy as np
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models.video as vid_models

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


if not hasattr(np, "int"):
    np.int = int


CLASS_NAMES = ["PedestrianLittering", "VehicleLittering"]
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}


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


class MoViNetClassifier:
    N_FRAMES = 16
    FRAME_SIZE = (112, 112)

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
        except Exception as exc:
            print(f"[MoViNet] Warning: Could not load weights: {exc}")
        model.to(self.device)
        model.eval()
        return model

    def preprocess_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, self.FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed.append(frame)

        clip = np.stack(processed).astype(np.float32) / 255.0
        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        clip = (clip - mean) / std
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float()
        return clip.to(self.device)

    @torch.no_grad()
    def predict_segment(self, frames: List[np.ndarray]) -> tuple[str, float]:
        if len(frames) < self.N_FRAMES:
            return CLASS_NAMES[0], 0.0
        logits = self.model(self.preprocess_clip(frames))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])


class PlateDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.conf = conf
        self.model = None
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        self.last_read_conf = 0.0

        weight_path = Path(model_path)
        if YOLO is None:
            print("[PlateDetector] Warning: ultralytics is not available")
            return
        if not weight_path.exists():
            print(f"[PlateDetector] Warning: plate model not found - {weight_path}")
            return

        try:
            self.model = YOLO(str(weight_path))
            print(f"[PlateDetector] Loaded YOLO plate detector from {weight_path}")
        except Exception as exc:
            print(f"[PlateDetector] Warning: failed to load plate model - {exc}")

    def _plate_box_score(self, x1: float, y1: float, x2: float, y2: float, frame_shape: tuple[int, int, int], model_score: float) -> float:
        height, width = frame_shape[:2]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        area_ratio = (box_w * box_h) / float(max(1, width * height))
        aspect = box_w / box_h
        center_y = ((y1 + y2) * 0.5) / float(max(1, height))

        score = float(model_score) * 4.0
        if 1.5 <= aspect <= 7.0:
            score += 1.5
        else:
            score -= 1.5

        if 0.0004 <= area_ratio <= 0.12:
            score += 1.0
        else:
            score -= 1.5

        if center_y >= 0.2:
            score += 0.8

        return score

    def _plate_texture_score(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return -1e9
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        height, width = gray.shape[:2]
        if height < 10 or width < 20:
            return -1e9

        edges = cv2.Canny(gray, 60, 160)
        edge_ratio = float(np.count_nonzero(edges)) / float(max(1, height * width))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        score = 0.0
        if 0.02 <= edge_ratio <= 0.25:
            score += 1.4
        elif edge_ratio < 0.01:
            score -= 2.0

        if 30.0 <= lap_var <= 2200.0:
            score += 1.0
        elif lap_var < 15.0:
            score -= 1.5

        return score

    def _normalize_plate_text(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
        if len(cleaned) < 8:
            return ""
        if len(cleaned) >= 2 and cleaned[:2] not in INDIAN_STATE_CODES:
            return ""
        return cleaned[:10]

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        height, width = plate_crop.shape[:2]
        if height < 120 or width < 240:
            scale = max(120 / max(1, height), 240 / max(1, width))
            plate_crop = cv2.resize(
                plate_crop,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(enhanced, 7, 50, 50)

    def _read_plate_text(self, plate_crop: np.ndarray) -> str:
        processed = self._preprocess_plate(plate_crop)
        results = self.reader.readtext(processed)
        candidates = []
        best_conf = 0.0

        for _, text, conf in results:
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = 0.0
            if conf_val < 0.15 or not text:
                continue
            normalized = self._normalize_plate_text(text)
            if normalized:
                candidates.append(normalized)
                best_conf = max(best_conf, conf_val)

        if not candidates:
            self.last_read_conf = 0.0
            return ""

        self.last_read_conf = best_conf
        return max(candidates, key=len)

    def detect(self, frame: np.ndarray) -> Optional[tuple[np.ndarray, str]]:
        if self.model is None:
            return None

        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        except Exception as exc:
            print(f"[PlateDetector] Detection error: {exc}")
            return None

        best_box = None
        best_score = -1e9
        for result in results:
            if not result.boxes or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                conf_val = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                score = self._plate_box_score(x1, y1, x2, y2, frame.shape, conf_val)

                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    score += self._plate_texture_score(frame[iy1:iy2, ix1:ix2])

                if score > best_score:
                    best_score = score
                    best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        height, width = frame.shape[:2]
        pad_x = max(4, int((x2 - x1) * 0.06))
        pad_y = max(4, int((y2 - y1) * 0.10))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        plate_crop = frame[y1:y2, x1:x2].copy()
        if plate_crop.size == 0:
            return None

        plate_text = self._read_plate_text(plate_crop)
        if self.last_read_conf < 0.35:
            plate_text = ""
        return plate_crop, plate_text


class FaceDetector:
    def __init__(self):
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(xml)

    def detect(self, frame) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if not len(faces):
            return None
        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        return frame[y:y + height, x:x + width].copy()


class SAWNDetector:
    THRESHOLD = 0.50

    def __init__(self, movinet_path: str, plate_model_path: str = "models/yolo/plates_yolov8/weights/best.pt", output_dir: str = "outputs/violations"):
        self.classifier = MoViNetClassifier(movinet_path)
        self.plate_det = PlateDetector(plate_model_path)
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._get_last_id()

    def _get_last_id(self) -> int:
        files = list(self.out_dir.glob("violation_*_snapshot.jpg"))
        if not files:
            return 0
        try:
            return max(int(file.name.split("_")[1]) for file in files)
        except Exception:
            return 0

    def _is_plausible_plate_text(self, text: str) -> bool:
        text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
        if len(text) < 8 or len(text) > 10:
            return False
        if len(text) >= 2 and text[:2] not in INDIAN_STATE_CODES:
            return False
        if sum(char.isdigit() for char in text) < 3:
            return False
        return True

    def _pick_plate_from_frames(self, frames: List[np.ndarray]) -> tuple[Optional[np.ndarray], str]:
        if not frames:
            return None, ""

        votes: dict[str, int] = {}
        conf_sums: dict[str, float] = {}
        crops: dict[str, np.ndarray] = {}
        best_visual_crop: Optional[np.ndarray] = None
        best_visual_score = -1e9

        step = max(1, len(frames) // 10)
        sampled = frames[::step]
        if sampled[-1] is not frames[-1]:
            sampled.append(frames[-1])

        for frame in sampled:
            result = self.plate_det.detect(frame)
            if not result:
                continue

            plate_crop, plate_text = result
            visual_score = self.plate_det._plate_texture_score(plate_crop)
            if visual_score > best_visual_score:
                best_visual_score = visual_score
                best_visual_crop = plate_crop

            if not self._is_plausible_plate_text(plate_text):
                continue

            key = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
            votes[key] = votes.get(key, 0) + 1
            conf_sums[key] = conf_sums.get(key, 0.0) + float(getattr(self.plate_det, "last_read_conf", 0.0))
            if key not in crops or plate_crop.shape[0] * plate_crop.shape[1] > crops[key].shape[0] * crops[key].shape[1]:
                crops[key] = plate_crop

        if not votes:
            return best_visual_crop, ""

        best_text = max(votes, key=lambda key: (votes[key], conf_sums.get(key, 0.0) / max(1, votes[key])))
        best_votes = votes[best_text]
        avg_conf = conf_sums.get(best_text, 0.0) / max(1, best_votes)
        if best_votes < 2 or avg_conf < 0.35:
            return best_visual_crop, ""

        return crops.get(best_text), best_text

    def process_video(self, video_path: str, progress_callback=None) -> Optional[Violation]:
        print(f"[SAWN] Scanning video (Streaming Mode): {Path(video_path).name}")
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 16:
            cap.release()
            print("  [ERROR] Video too short")
            return None

        step = max(1, fps // 2)
        window = []
        frame_idx = 0
        eval_history = []

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            window.append(frame)
            if len(window) > 16:
                window.pop(0)

            if len(window) == 16 and frame_idx % step == 0:
                label, conf = self.classifier.predict_segment(window)
                eval_history.append((frame_idx, frame_idx - 8, label, conf))

            if progress_callback and total_frames > 0 and frame_idx % max(step, 5) == 0:
                pct = min(100, int((frame_idx / total_frames) * 100))
                try:
                    progress_callback(pct)
                except Exception:
                    pass

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  -> Scanned {frame_idx}/{total_frames} frames...")

        cap.release()
        print(f"  -> Scan complete in {time.time() - start_time:.1f}s")

        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

        if not eval_history:
            print("  [SKIP] No evaluations performed")
            return None

        best_idx = max(range(len(eval_history)), key=lambda index: eval_history[index][3])
        if best_idx == 0 and len(eval_history) > 2:
            first_conf = eval_history[0][3]
            next_conf = eval_history[1][3]
            if (first_conf - next_conf) > 0.15:
                best_idx = max(range(1, len(eval_history)), key=lambda index: eval_history[index][3])

        _, peak_frame_idx, best_label, max_conf = eval_history[best_idx]
        if max_conf < self.THRESHOLD:
            print(f"  [SKIP] No violation found (Max Conf: {max_conf:.1%})")
            return None

        print(f"  [DETECTED] {best_label} at frame {peak_frame_idx} ({max_conf:.1%})")

        self._counter += 1
        violation_type = "Pedestrian" if "Pedestrian" in best_label else "Vehicle"

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
        ret, snapshot = cap.read()

        start_frame = max(0, peak_frame_idx - (fps * 3))
        clip_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(int(fps * 6)):
            ret_clip, clip_frame = cap.read()
            if not ret_clip:
                break
            clip_frames.append(clip_frame)
        cap.release()

        plate_crop, plate_text = self._pick_plate_from_frames(clip_frames)
        if plate_crop is None and snapshot is not None:
            result = self.plate_det.detect(snapshot)
            if result:
                candidate_crop, candidate_text = result
                plate_crop = candidate_crop
                plate_text = candidate_text if self._is_plausible_plate_text(candidate_text) else ""

        clip_name = f"violation_{self._counter:04d}_clip.mp4"
        clip_path = self.out_dir / clip_name
        self._save_clip_segment(clip_frames, fps, str(clip_path))

        violation = Violation(
            id=self._counter,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            violation_type=violation_type,
            confidence=max_conf,
            snapshot=snapshot,
            plate_crop=plate_crop,
            plate_text=plate_text,
            video_path=str(clip_path),
        )
        self._save_assets(violation)
        return violation

    def run_live(self, source: int = 0, show_preview: bool = True, callback=None) -> List[Violation]:
        print(f"[SAWN] Starting Live Feed (Source: {source})")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [ERROR] Could not open video source {source}")
            return []

        window = []
        violations = []
        last_violation_time = 0
        cooldown = 5

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                window.append(frame)
                if len(window) > 16:
                    window.pop(0)

                if len(window) == 16:
                    label, conf = self.classifier.predict_segment(window)
                    if conf > self.THRESHOLD and (time.time() - last_violation_time) > cooldown:
                        print(f"  [LIVE DETECTED] {label} ({conf:.1%})")
                        violation_type = "Pedestrian" if "Pedestrian" in label else "Vehicle"

                        self._counter += 1
                        snapshot = frame.copy()
                        plate_crop, plate_text = None, ""
                        result = self.plate_det.detect(snapshot)
                        if result:
                            plate_crop, plate_text = result
                            if not self._is_plausible_plate_text(plate_text):
                                plate_text = ""

                        violation = Violation(
                            id=self._counter,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            violation_type=violation_type,
                            confidence=conf,
                            snapshot=snapshot,
                            plate_crop=plate_crop,
                            plate_text=plate_text,
                        )
                        self._save_assets(violation)
                        violations.append(violation)
                        if callback:
                            callback(violation)
                        last_violation_time = time.time()

                if show_preview:
                    cv2.imshow("SAWN Live Dashboard", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return violations

    def _save_clip_segment(self, frames: List[np.ndarray], fps: int, dst_path: str):
        if not frames:
            return
        height, width = frames[0].shape[:2]
        target_height = 480
        target_width = int(target_height * (width / height))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(dst_path, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
            out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))

        for frame in frames:
            out.write(cv2.resize(frame, (target_width, target_height)))
        out.release()

    def _save_assets(self, violation: Violation):
        cv2.imwrite(str(self.out_dir / f"violation_{violation.id:04d}_snapshot.jpg"), violation.snapshot)
        if violation.face_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{violation.id:04d}_face.jpg"), violation.face_crop)
        if violation.plate_crop is not None:
            cv2.imwrite(str(self.out_dir / f"violation_{violation.id:04d}_plate.jpg"), violation.plate_crop)

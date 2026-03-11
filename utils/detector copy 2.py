import cv2
import easyocr
import numpy as np
import os
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

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


if not hasattr(np, "int"):
    np.int = int


CLASS_NAMES = ["PedestrianLittering", "VehicleLittering"]
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH",
    "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}

LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
})

DIGIT_TO_LETTER = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
})

INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1}\d{1,4}$"),
    re.compile(r"^\d{2}[A-Z]{2}\d{4}$"),
]

LETTER_FIXES = {
    "0": "O", "1": "I", "8": "B", "6": "G", "5": "S", "2": "Z",
}
DIGIT_FIXES = {
    "O": "0", "I": "1", "B": "8", "G": "6", "S": "5", "Z": "2",
    "Q": "0", "D": "0", "L": "1", "T": "7",
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
        self.paddle = None
        if PaddleOCR is not None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            init_candidates = [
                {"use_angle_cls": True, "lang": "en", "show_log": False},
                {"use_textline_orientation": True, "lang": "en"},
                {"lang": "en"},
            ]
            try:
                for kwargs in init_candidates:
                    try:
                        self.paddle = PaddleOCR(**kwargs)
                        break
                    except Exception:
                        self.paddle = None
            except Exception:
                self.paddle = None
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

    def _clean_text(self, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

    def _position_aware_correction(self, text: str) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) < 6:
            return cleaned

        corrected = list(cleaned[:10])
        for idx, char in enumerate(corrected):
            if idx < 2:
                corrected[idx] = LETTER_FIXES.get(char, char)
            elif idx < 4:
                corrected[idx] = DIGIT_FIXES.get(char, char)
            elif idx < 6:
                corrected[idx] = LETTER_FIXES.get(char, char)
            else:
                corrected[idx] = DIGIT_FIXES.get(char, char)
        return "".join(corrected)

    def _validate_indian_plate(self, text: str) -> tuple[str, bool, str]:
        candidate = self._position_aware_correction(text)
        formats = {
            "Standard": r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$",
            "HSRP": r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$",
            "Short": r"^[A-Z]{2}\d{2}[A-Z]{1}\d{1,4}$",
            "Diplomatic": r"^\d{2}[A-Z]{2}\d{4}$",
        }
        for format_name, pattern in formats.items():
            if re.match(pattern, candidate):
                return candidate, True, format_name
        return candidate, False, "Unknown"

    def _is_valid_plate_regex(self, text: str) -> bool:
        candidate = self._clean_text(text)
        return any(pattern.match(candidate) for pattern in INDIAN_PLATE_PATTERNS)

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        points = np.array(points, dtype="float32")
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        return np.array([
            points[np.argmin(s)],
            points[np.argmin(diff)],
            points[np.argmax(s)],
            points[np.argmax(diff)],
        ], dtype="float32")

    def _four_point_warp(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = self._order_points(points)
        dst = np.array([[0, 0], [399, 0], [399, 119], [0, 119]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (400, 120))

    def _get_plate_corners(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype("float32")
                corners[:, 0] += x1
                corners[:, 1] += y1
                return corners

        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

    def _preprocess_plate(self, plate_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        upscaled = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)
        return cleaned, upscaled

    def _read_plate_text_paddle(self, color_img: np.ndarray) -> tuple[str, float, str]:
        if self.paddle is None:
            return "", 0.0, "PaddleOCR"
        try:
            result = self.paddle.ocr(color_img, cls=True)
        except Exception:
            return "", 0.0, "PaddleOCR"
        if not result or not result[0]:
            return "", 0.0, "PaddleOCR"

        best_text = ""
        best_conf = 0.0
        for line in result[0]:
            if len(line) < 2:
                continue
            text = str(line[1][0])
            conf = float(line[1][1])
            candidate, is_valid, _ = self._validate_indian_plate(text)
            if is_valid and conf > best_conf:
                best_text = candidate
                best_conf = conf
        return best_text, best_conf, "PaddleOCR"

    def _enhance_plate_for_display(self, plate_crop: np.ndarray) -> np.ndarray:
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        display = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        denoised = cv2.bilateralFilter(display, 7, 45, 45)
        gauss = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        return cv2.addWeighted(denoised, 1.5, gauss, -0.5, 0)

    def _read_plate_text_easy(self, binary_img: np.ndarray, color_img: np.ndarray) -> tuple[str, float, str]:
        best_text = ""
        best_conf = 0.0
        variants = [color_img, binary_img]
        for image in variants:
            try:
                results = self.reader.readtext(image)
            except Exception:
                continue
            for _, text, conf in results:
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    conf_val = 0.0
                candidate, is_valid, _ = self._validate_indian_plate(text)
                if is_valid and conf_val > best_conf:
                    best_text = candidate
                    best_conf = conf_val
        return best_text, best_conf, "EasyOCR"

    def _read_plate_text(self, plate_crop: np.ndarray) -> str:
        binary_img, color_img = self._preprocess_plate(plate_crop)
        results: list[tuple[str, float, str]] = []

        paddle_result = self._read_plate_text_paddle(color_img)
        if paddle_result[0]:
            results.append(paddle_result)

        easy_result = self._read_plate_text_easy(binary_img, color_img)
        if easy_result[0]:
            results.append(easy_result)

        if not results:
            self.last_read_conf = 0.0
            return ""

        best_text, best_conf, _ = max(results, key=lambda item: item[1])
        self.last_read_conf = best_conf
        return best_text

    def _accept_plate_text(self, text: str, conf: float) -> bool:
        if not text:
            return False
        cleaned = self._clean_text(text)
        if not self._is_valid_plate_regex(cleaned):
            return False
        return conf >= 0.32

    def detect(self, frame: np.ndarray) -> Optional[tuple[np.ndarray, str]]:
        if self.model is None:
            return None

        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        except Exception as exc:
            print(f"[PlateDetector] Detection error: {exc}")
            return None

        candidates: list[tuple[float, tuple[int, int, int, int], float]] = []
        for result in results:
            if not result.boxes or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                conf_val = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
                score = self._plate_box_score(x1, y1, x2, y2, frame.shape, conf_val)
                candidates.append((score, (x1, y1, x2, y2), conf_val))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_crop: Optional[np.ndarray] = None
        best_text = ""
        best_conf = 0.0
        best_score = -1e9

        for det_score, (x1, y1, x2, y2), det_conf in candidates[:3]:
            corners = self._get_plate_corners(frame, x1, y1, x2, y2)
            warped = self._four_point_warp(frame, corners)
            if warped is None or warped.size == 0:
                continue

            plate_text = self._read_plate_text(warped)
            ocr_conf = float(self.last_read_conf)
            valid_bonus = 1.5 if self._accept_plate_text(plate_text, ocr_conf) else 0.0
            score = det_score + (ocr_conf * 4.0) + valid_bonus + det_conf

            if score > best_score:
                best_score = score
                best_crop = self._enhance_plate_for_display(warped)
                best_text = plate_text if self._accept_plate_text(plate_text, ocr_conf) else ""
                best_conf = ocr_conf

        if best_crop is None:
            return None

        self.last_read_conf = best_conf
        return best_crop, best_text


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
        if len(text) < 7 or len(text) > 10:
            return False
        _, is_valid, _ = self.plate_det._validate_indian_plate(text)
        if is_valid:
            return True
        if len(text) >= 2 and text[:2] in INDIAN_STATE_CODES and sum(char.isdigit() for char in text) >= 2:
            return True
        if len(text) >= 2:
            state_fix = text[:2].translate(str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}))
            if state_fix in INDIAN_STATE_CODES and sum(char.isdigit() for char in text) >= 2:
                return True
            return False
        return False

    def _plate_distance(self, a: str, b: str) -> int:
        if not a or not b:
            return 99
        if len(a) != len(b):
            return 99
        if a[:2] != b[:2]:
            return 99
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def _pick_plate_from_frames(self, frames: List[np.ndarray]) -> tuple[Optional[np.ndarray], str]:
        if not frames:
            return None, ""

        votes: dict[str, float] = {}
        conf_sums: dict[str, float] = {}
        crops: dict[str, np.ndarray] = {}
        crop_scores: dict[str, float] = {}
        text_seen: dict[str, int] = {}
        best_visual_crop: Optional[np.ndarray] = None
        best_visual_score = -1e9
        top_visual_crops: list[tuple[float, np.ndarray]] = []

        # Accuracy mode: sample more frames for stronger temporal consensus.
        step = max(1, len(frames) // 20)
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

            if len(top_visual_crops) < 8:
                top_visual_crops.append((visual_score, plate_crop))
            else:
                min_idx = min(range(len(top_visual_crops)), key=lambda i: top_visual_crops[i][0])
                if visual_score > top_visual_crops[min_idx][0]:
                    top_visual_crops[min_idx] = (visual_score, plate_crop)

            if not self._is_plausible_plate_text(plate_text):
                continue

            key = re.sub(r"[^A-Z0-9]", "", plate_text.upper())
            ocr_conf = float(getattr(self.plate_det, "last_read_conf", 0.0))
            vote_weight = 1.0 + (ocr_conf * 1.5) + max(0.0, min(1.5, visual_score * 0.25))
            votes[key] = votes.get(key, 0.0) + vote_weight
            conf_sums[key] = conf_sums.get(key, 0.0) + ocr_conf
            text_seen[key] = text_seen.get(key, 0) + 1
            if key not in crops or visual_score > crop_scores.get(key, -1e9):
                crops[key] = plate_crop
                crop_scores[key] = visual_score

        # Merge near-identical OCR strings (single char noise) into one group.
        if votes:
            keys = sorted(votes.keys(), key=lambda k: (votes[k], conf_sums.get(k, 0.0), text_seen.get(k, 0)), reverse=True)
            merged_parent: dict[str, str] = {}
            for i, key_a in enumerate(keys):
                parent = merged_parent.get(key_a, key_a)
                for key_b in keys[i + 1:]:
                    if key_b in merged_parent:
                        continue
                    if self._plate_distance(parent, key_b) <= 1:
                        merged_parent[key_b] = parent

            if merged_parent:
                merged_votes: dict[str, float] = {}
                merged_conf: dict[str, float] = {}
                merged_count: dict[str, int] = {}
                for key in keys:
                    root = merged_parent.get(key, key)
                    merged_votes[root] = merged_votes.get(root, 0.0) + votes.get(key, 0.0)
                    merged_conf[root] = merged_conf.get(root, 0.0) + conf_sums.get(key, 0.0)
                    merged_count[root] = merged_count.get(root, 0) + text_seen.get(key, 0)
                    if key in crops and (root not in crops or crop_scores.get(key, -1e9) > crop_scores.get(root, -1e9)):
                        crops[root] = crops[key]
                        crop_scores[root] = crop_scores.get(key, -1e9)
                votes = merged_votes
                conf_sums = merged_conf
                text_seen = merged_count

        if not votes:
            if best_visual_crop is None:
                return None, ""
            # Retry OCR on top visual candidates to recover hard frames.
            for _, candidate_crop in sorted(top_visual_crops, key=lambda item: item[0], reverse=True):
                fallback_text = self.plate_det._read_plate_text(candidate_crop)
                if self._is_plausible_plate_text(fallback_text):
                    return candidate_crop, fallback_text
            return best_visual_crop, ""

        best_text = max(votes, key=lambda key: (votes[key], conf_sums.get(key, 0.0) / max(1, text_seen.get(key, 1))))
        best_votes = votes[best_text]
        best_count = text_seen.get(best_text, 1)
        avg_conf = conf_sums.get(best_text, 0.0) / max(1, best_count)
        # Prevent single noisy frame OCR from being accepted as final text.
        if (best_count < 2 and avg_conf < 0.45) or avg_conf < 0.22 or best_votes < 1.5:
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
        plate_scan_frames = []
        sample_every = max(1, fps // 4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(int(fps * 6)):
            try:
                ret_clip, clip_frame = cap.read()
            except cv2.error:
                break
            if not ret_clip:
                break

            # Keep video frames compact in memory to avoid OpenCV allocation crashes.
            h, w = clip_frame.shape[:2]
            if h > 480:
                new_w = max(1, int(w * (480.0 / float(h))))
                clip_for_save = cv2.resize(clip_frame, (new_w, 480), interpolation=cv2.INTER_AREA)
            else:
                clip_for_save = clip_frame
            clip_frames.append(clip_for_save)

            # For OCR, sample fewer frames and keep the original quality frame.
            if i % sample_every == 0:
                plate_scan_frames.append(clip_frame)
        cap.release()

        plate_crop, plate_text = self._pick_plate_from_frames(plate_scan_frames)
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
        cv2.imwrite(
            str(self.out_dir / f"violation_{violation.id:04d}_snapshot.jpg"),
            violation.snapshot,
            [cv2.IMWRITE_JPEG_QUALITY, 98],
        )
        if violation.face_crop is not None:
            cv2.imwrite(
                str(self.out_dir / f"violation_{violation.id:04d}_face.jpg"),
                violation.face_crop,
                [cv2.IMWRITE_JPEG_QUALITY, 98],
            )
        if violation.plate_crop is not None:
            cv2.imwrite(
                str(self.out_dir / f"violation_{violation.id:04d}_plate.jpg"),
                violation.plate_crop,
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

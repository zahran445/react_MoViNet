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
    plate_bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) for visualization


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
        model.fc = nn.Linear(in_feats, len(CLASS_NAMES))
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
            # Not enough frames
            return "Unknown", 0.0
        logits = self.model(self.preprocess_clip(frames))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        # Guard: if idx is out of range of CLASS_NAMES (e.g. old 2-class model
        # loaded while code expects 3+ classes) fall back to the first class.
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else CLASS_NAMES[0]
        return label, float(probs[idx])


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

    def _lower_plate_exposure(self, plate_img: np.ndarray) -> np.ndarray:
        if plate_img is None or plate_img.size == 0:
            return np.array([])

        lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        l_channel = cv2.convertScaleAbs(l_channel, alpha=1.18, beta=-24)
        l_channel = np.minimum(l_channel, 232).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _get_plate_corners(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        # Add padding to bbox to give contour detection more context
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if roi.size == 0:
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # More aggressive edge detection for better plate boundary
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 30, 150)
        
        # Dilate edges to connect broken contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        roi_h, roi_w = roi.shape[:2]
        min_area = (roi_w * roi_h) * 0.3  # At least 30% of ROI
        
        for cnt in contours[:10]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            peri = cv2.arcLength(cnt, True)
            # Try different epsilon values for better approximation
            for epsilon_factor in [0.02, 0.03, 0.015]:
                approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
                if len(approx) == 4:
                    corners = approx.reshape(4, 2).astype("float32")
                    corners[:, 0] += x1_pad
                    corners[:, 1] += y1_pad
                    
                    # Validate corners form a reasonable quadrilateral
                    rect = self._order_points(corners)
                    width = np.linalg.norm(rect[1] - rect[0])
                    height = np.linalg.norm(rect[3] - rect[0])
                    aspect = width / max(1, height)
                    
                    if 1.5 <= aspect <= 8.0 and width >= 40 and height >= 15:
                        return corners

        # Fallback: return original bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")

    def _preprocess_plate(self, plate_img: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if plate_img is None or plate_img.size == 0:
            return [], []
            
        # Upscale for better OCR (but not too much)
        h, w = plate_img.shape[:2]
        if h < 40 or w < 120:
            # Small plate, upscale 3x
            upscaled = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        elif h < 80 or w < 240:
            # Medium plate, upscale 2x
            upscaled = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            # Already large enough
            upscaled = plate_img.copy()

        lowered = self._lower_plate_exposure(upscaled)
        gray = cv2.cvtColor(lowered, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.12, beta=-12)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
        boosted = cv2.addWeighted(enhanced, 1.0, blackhat, 0.8, 0)

        gauss = cv2.GaussianBlur(boosted, (0, 0), 1.1)
        sharpened = cv2.addWeighted(boosted, 1.35, gauss, -0.35, 0)


        _, binary_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_adapt = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            27,
            6,
        )
        _, binary_inv = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel2)
        cleaned_adapt = cv2.morphologyEx(binary_adapt, cv2.MORPH_CLOSE, kernel2)
        cleaned_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel2)

        color_variants = [lowered, cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)]
        binary_variants = [cleaned_otsu, cleaned_adapt, cleaned_inv]
        return binary_variants, color_variants

    def _is_plausible_plate_candidate(self, text: str) -> bool:
        cleaned = self._clean_text(text)
        if len(cleaned) < 7 or len(cleaned) > 10:
            return False
        candidate, is_valid, _ = self._validate_indian_plate(cleaned)
        if is_valid:
            return True
        return len(candidate) >= 2 and candidate[:2] in INDIAN_STATE_CODES

    def _build_retry_plate_variants(self, plate_crop: np.ndarray) -> list[np.ndarray]:
        if plate_crop is None or plate_crop.size == 0:
            return []

        lowered = self._lower_plate_exposure(plate_crop)
        high_contrast = cv2.convertScaleAbs(lowered, alpha=1.38, beta=-26)
        low_contrast = cv2.convertScaleAbs(lowered, alpha=0.92, beta=6)
        darker_high_contrast = cv2.convertScaleAbs(plate_crop, alpha=1.55, beta=-44)
        brighter_soft = cv2.convertScaleAbs(plate_crop, alpha=1.02, beta=16)

        gamma_dark = np.clip(((plate_crop.astype(np.float32) / 255.0) ** 1.35) * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(lowered, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        gray_bgr = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

        return [
            plate_crop,
            lowered,
            high_contrast,
            low_contrast,
            darker_high_contrast,
            brighter_soft,
            gamma_dark,
            gray_bgr,
        ]

    def _retry_plate_text(self, plate_crop: np.ndarray) -> tuple[str, float]:
        best_text = ""
        best_conf = 0.0

        for idx, variant_crop in enumerate(self._build_retry_plate_variants(plate_crop), start=1):
            variant_text = self._read_plate_text(variant_crop)
            variant_conf = float(self.last_read_conf)
            print(f"      [OCR Retry] variant#{idx}: '{variant_text}' conf={variant_conf:.3f}")
            if self._accept_plate_text(variant_text, variant_conf) and variant_conf > best_conf:
                best_text = variant_text
                best_conf = variant_conf

        self.last_read_conf = best_conf
        return best_text, best_conf

    def _read_plate_text_paddle(self, color_variants: List[np.ndarray]) -> tuple[str, float, str]:
        if self.paddle is None:
            return "", 0.0, "PaddleOCR"

        best_valid_text = ""
        best_valid_conf = 0.0
        best_plausible_text = ""
        best_plausible_conf = 0.0

        for color_img in color_variants:
            if color_img is None or color_img.size == 0:
                continue
            try:
                result = self.paddle.ocr(color_img, cls=True)
            except Exception:
                continue
            if not result or not result[0]:
                continue

            for line in result[0]:
                if len(line) < 2:
                    continue
                text = str(line[1][0])
                conf = float(line[1][1])
                candidate, is_valid, _ = self._validate_indian_plate(text)
                if is_valid and conf > best_valid_conf:
                    best_valid_text = candidate
                    best_valid_conf = conf
                elif self._is_plausible_plate_candidate(candidate) and conf > best_plausible_conf:
                    best_plausible_text = candidate
                    best_plausible_conf = conf

        if best_valid_text:
            return best_valid_text, best_valid_conf, "PaddleOCR"
        return best_plausible_text, best_plausible_conf, "PaddleOCR"

    def _enhance_plate_for_display(self, plate_crop: np.ndarray, detected_text: str = "") -> np.ndarray:
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # ── Step 1: Upscale to a fixed display width ───────────────────────
        target_w = 480
        h0, w0 = plate_crop.shape[:2]
        scale = max(4.0, target_w / max(1, w0))
        upscaled = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # ── Step 2: Mild cleanup + contrast for readable display ───────────
        lowered = self._lower_plate_exposure(upscaled)
        denoised = cv2.GaussianBlur(lowered, (3, 3), 0)
        gauss = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        sharp = cv2.addWeighted(denoised, 1.2, gauss, -0.2, 0)

        # ── Step 3: Contrast / brightness normalisation per channel ────────
        result = np.zeros_like(sharp)
        for c in range(sharp.shape[2] if len(sharp.shape) == 3 else 1):
            ch = sharp[:, :, c] if len(sharp.shape) == 3 else sharp
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            eq = clahe.apply(ch)
            eq = cv2.convertScaleAbs(eq, alpha=1.08, beta=-6)
            if len(sharp.shape) == 3:
                result[:, :, c] = eq
            else:
                result = eq
        enhanced = result

        # ── Step 4: Subtle final sharpening pass ───────────────────────────
        kernel = np.array([[0, -0.25, 0], [-0.25, 2.0, -0.25], [0, -0.25, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # ── Step 5: Text label bar at the bottom ───────────────────────────
        label = detected_text if detected_text else "NO PLATE"
        h, w = enhanced.shape[:2]
        bar_h = 44
        canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
        canvas[:h] = enhanced if len(enhanced.shape) == 3 else cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        # dark bar
        canvas[h:] = (30, 30, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = max(0, (w - tw) // 2)
        ty = h + (bar_h + th) // 2 - 2
        color = (80, 220, 80) if detected_text else (80, 80, 220)
        cv2.putText(canvas, label, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(canvas, label, (tx, ty), font, font_scale, color, thickness)

        return canvas

    def _read_plate_text_easy(self, binary_variants: List[np.ndarray], color_variants: List[np.ndarray]) -> tuple[str, float, str]:
        best_valid_text = ""
        best_valid_conf = 0.0
        best_plausible_text = ""
        best_plausible_conf = 0.0
        variants = [*color_variants, *binary_variants]
        for image in variants:
            if image is None or image.size == 0:
                continue
            try:
                results = self.reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
            except Exception:
                continue
            for _, text, conf in results:
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    conf_val = 0.0
                candidate, is_valid, _ = self._validate_indian_plate(text)
                if is_valid and conf_val > best_valid_conf:
                    best_valid_text = candidate
                    best_valid_conf = conf_val
                elif self._is_plausible_plate_candidate(candidate) and conf_val > best_plausible_conf:
                    best_plausible_text = candidate
                    best_plausible_conf = conf_val

        if best_valid_text:
            return best_valid_text, best_valid_conf, "EasyOCR"
        return best_plausible_text, best_plausible_conf, "EasyOCR"

    def _read_plate_text(self, plate_crop: np.ndarray) -> str:
        binary_variants, color_variants = self._preprocess_plate(plate_crop)
        results: list[tuple[str, float, str]] = []

        # Debug: Check if preprocessing produced valid images
        if not color_variants:
            print("[PlateDetector] Warning: color variants are empty after preprocessing")
            self.last_read_conf = 0.0
            return ""
        if not binary_variants:
            print("[PlateDetector] Warning: binary variants are empty after preprocessing")
            self.last_read_conf = 0.0
            return ""

        paddle_result = self._read_plate_text_paddle(color_variants)
        if paddle_result[0]:
            results.append(paddle_result)
            print(f"      [OCR] PaddleOCR: '{paddle_result[0]}' conf={paddle_result[1]:.3f}")

        easy_result = self._read_plate_text_easy(binary_variants, color_variants)
        if easy_result[0]:
            results.append(easy_result)
            print(f"      [OCR] EasyOCR:   '{easy_result[0]}' conf={easy_result[1]:.3f}")

        if not results:
            self.last_read_conf = 0.0
            return ""

        best_text, best_conf, engine = max(results, key=lambda item: item[1])
        self.last_read_conf = best_conf
        print(f"      [OCR] Best ({engine}): '{best_text}' conf={best_conf:.3f}")
        return best_text

    def _accept_plate_text(self, text: str, conf: float) -> bool:
        if not text:
            return False
        cleaned = self._clean_text(text)
        if not self._is_valid_plate_regex(cleaned):
            return False
        return conf >= 0.20

    def detect(self, frame: np.ndarray) -> Optional[tuple[np.ndarray, str, tuple[int, int, int, int]]]:
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
            # print("[PlateDetector] No plate candidates detected")
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_crop: Optional[np.ndarray] = None
        best_text = ""
        best_conf = 0.0
        best_score = -1e9
        best_bbox = None

        print(f"  [PlateDetector] {len(candidates)} candidate(s) found, evaluating top 3...")
        for det_score, (x1, y1, x2, y2), det_conf in candidates[:3]:
            corners = self._get_plate_corners(frame, x1, y1, x2, y2)
            warped = self._four_point_warp(frame, corners)

            # Validate warped dimensions
            if warped is None or warped.size == 0:
                continue
            if warped.shape[0] < 20 or warped.shape[1] < 60:
                continue

            # Sharpness check — skip blurry crops before expensive OCR
            gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
            sharpness = float(cv2.Laplacian(gray_warp, cv2.CV_64F).var())
            if sharpness < 25.0:
                print(f"    [PlateDetector] bbox=({x1},{y1},{x2},{y2}) sharpness={sharpness:.1f} -> TOO BLURRY, skipping OCR")
                # Still record as a visual candidate so we have at least a crop
                if best_crop is None:
                    best_crop = self._enhance_plate_for_display(warped, "")
                    best_bbox = (x1, y1, x2, y2)
                continue

            # Skip expensive OCR on very weak detections to avoid long stalls on no-plate clips.
            if det_conf < 0.22 and det_score < 1.8:
                if best_crop is None:
                    best_crop = self._enhance_plate_for_display(warped, "")
                    best_bbox = (x1, y1, x2, y2)
                print(f"    [PlateDetector] bbox=({x1},{y1},{x2},{y2}) weak det_conf={det_conf:.2f} det_score={det_score:.2f} -> skipping OCR")
                continue

            print(f"    [PlateDetector] bbox=({x1},{y1},{x2},{y2}) det_conf={det_conf:.2f} sharpness={sharpness:.1f} -> running OCR")
            plate_text = self._read_plate_text(warped)
            ocr_conf = float(self.last_read_conf)
            if not self._accept_plate_text(plate_text, ocr_conf):
                # Retry OCR only for promising first-pass reads.
                should_retry = bool(plate_text) or ocr_conf >= 0.12
                if should_retry:
                    retry_text, retry_conf = self._retry_plate_text(warped)
                    if retry_text and retry_conf >= ocr_conf:
                        plate_text = retry_text
                        ocr_conf = retry_conf
            print(f"    [PlateDetector] OCR result: '{plate_text}' conf={ocr_conf:.3f} valid={self._accept_plate_text(plate_text, ocr_conf)}")
            valid_bonus = 1.5 if self._accept_plate_text(plate_text, ocr_conf) else 0.0
            score = det_score + (ocr_conf * 4.0) + valid_bonus + det_conf

            if score > best_score:
                best_score = score
                # Return raw OCR text so _pick_plate_from_frames can accumulate votes.
                # The final acceptance gate lives in the consensus logic, not here.
                best_text = plate_text
                best_crop = self._enhance_plate_for_display(warped, plate_text)
                best_conf = ocr_conf
                best_bbox = (x1, y1, x2, y2)

        if best_crop is None:
            print("  [PlateDetector] No usable crop produced")
            return None

        self.last_read_conf = best_conf
        print(f"  [PlateDetector] Final plate: '{best_text}' conf={best_conf:.3f}")
        return best_crop, best_text, best_bbox


class GeneralObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.40):
        self.conf = conf
        self.model = None
        self.face_det = FaceDetector() # NEW: Fallback face detector
        if YOLO is not None:
            try:
                # Use current directory or local file if possible
                self.model = YOLO(model_path)
                print(f"[GeneralDetector] Loaded YOLO model from {model_path}")
            except Exception as exc:
                print(f"[GeneralDetector] Warning: failed to load model - {exc}")

    def is_vehicle_present(self, frame: np.ndarray) -> bool:
        return self._is_present(frame, [2, 3, 5, 7])

    def is_person_present(self, frame: np.ndarray) -> bool:
        # First, try general YOLO (standard person detection)
        yolo_person = self._is_present(frame, [0])
        if yolo_person:
            return True
            
        # Second, fallback to face detection if close-up
        face_crop = self.face_det.detect(frame)
        if face_crop is not None:
            # print("  [DEBUG] YOLO missed body, but FaceDetector found a human face")
            return True
            
        return False

    def _is_present(self, frame: np.ndarray, target_classes: list[int]) -> bool:
        if self.model is None or frame is None:
            return True # Fallback

        try:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            for result in results:
                if not result.boxes:
                    continue
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls in target_classes:
                        return True
        except Exception as exc:
            print(f"[GeneralDetector] Error: {exc}")
            return True

        return False


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
    # Raised from 0.88 → 0.90: stricter gate compensated by lower motion threshold.
    THRESHOLD = 0.90
    # Raised from 0.93 → 0.96: vehicle class needs high confidence + presence validation.
    VEHICLE_THRESHOLD = 0.96
    # Raised from 4 → 5 and radius from 2 → 3: require more sustained temporal agreement.
    CONSENSUS_MIN_HITS = 5
    CONSENSUS_RADIUS = 3
    PLATE_SCAN_MAX_FALLBACK_RETRIES = 2

    def __init__(self, movinet_path: str, plate_model_path: str = "models/yolo/plates_yolov8/weights/best.pt", output_dir: str = "outputs/violations"):
        self.classifier = MoViNetClassifier(movinet_path)
        self.plate_det = PlateDetector(plate_model_path)
        self.obj_det = GeneralObjectDetector("yolov8n.pt")
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._get_last_id()

    def predict_segment(self, frames: List[np.ndarray]) -> tuple[str, float]:
        return self.classifier.predict_segment(frames)

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

    def _label_threshold(self, label: str) -> float:
        # Normal class should never trigger — threshold of 1.0 means unreachable.
        if "Normal" in (label or ""):
            return 1.0
        # Vehicle-only motion is a common false-positive source, so require stronger confidence.
        if "Vehicle" in label:
            return self.VEHICLE_THRESHOLD
        return self.THRESHOLD

    def _violation_label(self, label: str) -> str:
        if "Pedestrian" in (label or ""):
            return "PedestrianLittering"
        return "VehicleLittering"

    def _passes_temporal_consensus(self, eval_history: list[tuple[int, int, str, float]], best_idx: int) -> bool:
        if not eval_history:
            return False

        best_label = eval_history[best_idx][2]
        label_threshold = self._label_threshold(best_label)
        secondary_threshold = max(0.55, label_threshold - 0.12)

        start = max(0, best_idx - self.CONSENSUS_RADIUS)
        end = min(len(eval_history), best_idx + self.CONSENSUS_RADIUS + 1)
        neighborhood = eval_history[start:end]

        strong_hits = sum(1 for _, _, label, conf in neighborhood if label == best_label and conf >= secondary_threshold)
        if strong_hits < self.CONSENSUS_MIN_HITS:
            print(
                f"  [SKIP] Consensus failed for {best_label}: hits={strong_hits}/{self.CONSENSUS_MIN_HITS} "
                f"(threshold={secondary_threshold:.2f})"
            )
            return False
        return True

    def _motion_series(self, frames: List[np.ndarray]) -> list[float]:
        series: list[float] = []
        if len(frames) < 2:
            return series

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(gray, prev_gray)
            series.append(float(np.mean(diff)))
            prev_gray = gray
        return series

    def _passes_action_gate(self, frames: List[np.ndarray], label: str) -> bool:
        # This gate rejects static scenes and smooth pass-by motion that often cause false positives.
        series = self._motion_series(frames)
        if not series:
            return False

        mean_motion = float(np.mean(series))
        std_motion = float(np.std(series))
        peak_motion = float(np.max(series))
        min_motion = float(np.min(series))
        motion_span = peak_motion - min_motion
        burst_ratio = peak_motion / max(1e-6, (mean_motion + 1e-6))

        if "Vehicle" in label:
            # Reverting back to OR logic for vehicles – still requires a burst OR multiple peaks
            # to separate a throw gesture from a smooth drive-by.
            gradients = np.diff(np.array(series, dtype=np.float32))
            sign_changes = int(np.sum(np.diff(np.sign(gradients)) != 0)) if len(gradients) > 1 else 0
            peak_threshold = mean_motion + (0.8 * std_motion)
            peak_count = 0
            for i in range(1, len(series) - 1):
                if series[i] > series[i - 1] and series[i] >= series[i + 1] and series[i] >= peak_threshold:
                    peak_count += 1

            sharp_or_complex = (burst_ratio >= 1.8) or (peak_count >= 2) or (sign_changes >= 4)
            passes = (
                mean_motion >= 5.0
                and std_motion >= 2.5
                and motion_span >= 3.0
                and burst_ratio >= 1.5
                and sharp_or_complex
            )
        else:
            # Pedestrian littering - Refined for "purse throws" (lower total motion)
            # while still blocking head turns (higher timing precision).
            peak_idx = int(np.argmax(series))
            # Burst timing: peak should be in the middle of the window, not at edges.
            center_peak = 1 <= peak_idx <= max(1, len(series) - 2)
            
            # Sign changes: a throw has a transient speed-up, peak, then slow-down.
            gradients = np.diff(np.array(series, dtype=np.float32))
            sign_changes = int(np.sum(np.diff(np.sign(gradients)) != 0)) if len(gradients) > 1 else 0

            # CALIBRATION:
            # - False positive head turn: mean=3.3, peak=5.1, often smooth/slow.
            # - Real violation: purse throw can have mean as low as 3.7.
            conds = {
                "mean": mean_motion >= 3.6,
                "std": std_motion >= 1.0,
                "span": motion_span >= 2.0,
                "timing": center_peak,
                "burst": (sign_changes >= 2 or burst_ratio >= 1.3)
            }
            passes = all(conds.values()) or (peak_motion >= 12.0 and std_motion >= 2.5)
            
            if not passes:
                failed = [k for k,v in conds.items() if not v]
                print(f"  [DEBUG] Action Gate Refused Pedestrian: FAILED={failed} | "
                      f"mean={mean_motion:.1f} std={std_motion:.1f} span={motion_span:.1f} "
                      f"peak_idx={peak_idx} burst={burst_ratio:.1f}")

        if not passes:
            print(
                "  [SKIP] Action gate rejected "
                f"{label}: mean={mean_motion:.2f} std={std_motion:.2f} peak={peak_motion:.2f} "
                f"span={motion_span:.2f} burst={burst_ratio:.2f}"
            )
        return passes

    def _pick_plate_from_frames(self, frames: List[np.ndarray]) -> tuple[Optional[np.ndarray], str]:
        if not frames:
            print("  [PlateOCR] No frames provided for plate search")
            return None, ""

        votes: dict[str, float] = {}
        conf_sums: dict[str, float] = {}
        crops: dict[str, np.ndarray] = {}
        crop_scores: dict[str, float] = {}
        text_seen: dict[str, int] = {}
        best_visual_crop: Optional[np.ndarray] = None
        best_visual_score = -1e9
        top_visual_crops: list[tuple[float, np.ndarray]] = []

        # Scan every provided frame — caller is responsible for deciding sampling rate.
        print(f"  [PlateOCR] Scanning all {len(frames)} plate frames for consensus...")
        detected_count = 0
        valid_text_count = 0

        for frame_no, frame in enumerate(frames):
            # Fast path: when no plate has been seen yet, skip every other frame on long clips.
            if len(frames) > 24 and detected_count == 0 and (frame_no % 2 == 1):
                continue

            result = self.plate_det.detect(frame)
            if not result:
                continue
            detected_count += 1

            plate_crop, plate_text, plate_bbox = result
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
                if plate_text:
                    print(f"    [PlateOCR] Frame {frame_no}: implausible text '{plate_text}', skipping")
                continue

            valid_text_count += 1
            print(f"    [PlateOCR] Frame {frame_no}: plausible plate '{plate_text}'")
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

        print(f"  [PlateOCR] Summary: {detected_count}/{len(frames)} frames had plate bbox, {valid_text_count} had plausible text")

        if not votes:
            if best_visual_crop is None:
                print("  [PlateOCR] No plate found in any frame")
                return None, ""
            # Retry OCR on top visual candidates (sorted by sharpness) to recover text.
            print("  [PlateOCR] No votes — retrying OCR on sharpest visual crops...")
            retry_candidates = sorted(top_visual_crops, key=lambda item: item[0], reverse=True)[:self.PLATE_SCAN_MAX_FALLBACK_RETRIES]
            for rank, (vscore, candidate_crop) in enumerate(retry_candidates):
                fallback_text, fallback_conf = self.plate_det._retry_plate_text(candidate_crop)
                print(f"    [PlateOCR] Fallback crop #{rank+1} visual_score={vscore:.2f} -> '{fallback_text}'")
                if self._is_plausible_plate_text(fallback_text):
                    print(f"  [PlateOCR] Recovered plate text from fallback: '{fallback_text}'")
                    self.plate_det.last_read_conf = fallback_conf
                    return candidate_crop, fallback_text
            print("  [PlateOCR] All fallbacks failed; returning best visual crop with no text")
            return best_visual_crop, ""

        best_text = max(votes, key=lambda key: (votes[key], conf_sums.get(key, 0.0) / max(1, text_seen.get(key, 1))))
        best_votes = votes[best_text]
        best_count = text_seen.get(best_text, 1)
        avg_conf = conf_sums.get(best_text, 0.0) / max(1, best_count)
        # Accept if the plate is regex-valid and seen more than once, OR conf is high enough.
        is_regex_valid = self.plate_det._is_valid_plate_regex(best_text)
        rejected = False
        if is_regex_valid:
            # Valid format: require at least 2 detections across frames (outvotes noise)
            if best_count < 2:
                rejected = True
        else:
            # Unknown format: stricter — need good conf + votes
            if avg_conf < 0.28 or best_votes < 1.5:
                rejected = True

        if rejected:
            print(f"  [PlateOCR] Consensus REJECTED: text='{best_text}' valid={is_regex_valid} count={best_count} avg_conf={avg_conf:.3f} votes={best_votes:.1f}")
            retry_crop = crops.get(best_text)
            retry_text, retry_conf = self.plate_det._retry_plate_text(retry_crop) if retry_crop is not None else ("", 0.0)
            if self._is_plausible_plate_text(retry_text):
                self.plate_det.last_read_conf = retry_conf
                print(f"  [PlateOCR] Recovered plate text after consensus retry: '{retry_text}'")
                return retry_crop, retry_text
            return crops.get(best_text), ""

        print(f"  [PlateOCR] Consensus ACCEPTED: text='{best_text}' valid={is_regex_valid} count={best_count} avg_conf={avg_conf:.3f} votes={best_votes:.1f}")
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
        
        # --- SMART AGENT MAPPING ---
        # Get frame at peak for presence check.
        cap_check = cv2.VideoCapture(video_path)
        cap_check.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
        ret_check, frame_check = cap_check.read()
        cap_check.release()
        
        is_person = self.obj_det.is_person_present(frame_check) if ret_check else False
        is_vehicle = self.obj_det.is_vehicle_present(frame_check) if ret_check else False
        
        final_label = best_label
        if is_person and not is_vehicle:
            final_label = "PedestrianLittering"
        elif is_vehicle and not is_person:
            final_label = "VehicleLittering"
        elif not is_person and not is_vehicle:
            print(f"  [SKIP] No agent (person/vehicle) detected at peak frame {peak_frame_idx}")
            return None

        # Apply thresholds based on the CORRECTED agent label.
        label_threshold = self._label_threshold(final_label)
        effective_threshold = min(label_threshold, 0.85) # Lower if presence confirmed
        
        if max_conf < effective_threshold:
            print(f"  [SKIP] Conf {max_conf:.1%} below effective threshold {effective_threshold:.1%} for {final_label}")
            return None

        if "Vehicle" in final_label:
            vehicle_support_hits = sum(
                1 for _, _, label, conf in eval_history
                if "Vehicle" in label and conf >= 0.82
            )
            if vehicle_support_hits < 2:
                print(f"  [SKIP] Vehicle support too weak: hits={vehicle_support_hits}")
                return None

        if not self._passes_temporal_consensus(eval_history, best_idx):
            return None

        best_window_start = max(0, peak_frame_idx - 8)
        best_window_end = min(total_frames - 1, peak_frame_idx + 8)
        action_frames = []
        cap_action = cv2.VideoCapture(video_path)
        cap_action.set(cv2.CAP_PROP_POS_FRAMES, best_window_start)
        for _ in range(max(2, best_window_end - best_window_start + 1)):
            ret_action, frame_action = cap_action.read()
            if not ret_action:
                break
            action_frames.append(frame_action)
        cap_action.release()

        action_gate_passed = self._passes_action_gate(action_frames, best_label)
        allow_override = "Pedestrian" in best_label
        high_conf_override = allow_override and (max_conf >= min(0.94, label_threshold + 0.10))
        if not action_gate_passed and not high_conf_override:
            return None
        if not action_gate_passed and high_conf_override:
            print(f"  [INFO] Action gate overridden due to very high confidence ({max_conf:.1%})")

        print(f"  [DETECTED] {best_label} at frame {peak_frame_idx} ({max_conf:.1%})")

        self._counter += 1
        violation_type = self._violation_label(best_label)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame_idx)
        ret, snapshot = cap.read()

        start_frame = max(0, peak_frame_idx - (fps * 3))
        clip_frames = []
        plate_scan_frames = []
        # Sample every ~5 frames to reduce no-plate latency while retaining consensus coverage.
        sample_every = max(1, fps // 6)
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

            # Sample more densely so blurry frames are outvoted by sharp ones.
            if i % sample_every == 0:
                plate_scan_frames.append(clip_frame)
        cap.release()
        print(f"  [SAWN] Clip collected: {len(clip_frames)} frames, {len(plate_scan_frames)} sampled for plate scan")

        plate_crop, plate_text = self._pick_plate_from_frames(plate_scan_frames)
        plate_bbox = None
        if plate_crop is None and snapshot is not None:
            print("  [SAWN] No plate from clip; retrying on peak snapshot frame")
            result = self.plate_det.detect(snapshot)
            if result:
                candidate_crop, candidate_text, candidate_bbox = result
                plate_crop = candidate_crop
                if self._is_plausible_plate_text(candidate_text):
                    plate_text = candidate_text
                else:
                    retry_text, retry_conf = self.plate_det._retry_plate_text(candidate_crop)
                    plate_text = retry_text if self._is_plausible_plate_text(retry_text) else ""
                    self.plate_det.last_read_conf = retry_conf
                plate_bbox = candidate_bbox
                print(f"  [SAWN] Snapshot plate: '{plate_text or '<empty>'}'")
            else:
                print("  [SAWN] Snapshot plate detection also missed")
        _plate_display = plate_text or "<empty>"
        _crop_display = "yes" if plate_crop is not None else "no"
        print(f"  [SAWN] Final plate text: '{_plate_display}', crop={_crop_display}")

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
            plate_bbox=plate_bbox,
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
        frame_idx = 0
        consecutive_label = ""
        consecutive_hits = 0

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
                    
                    # --- SMART AGENT MAPPING ---
                    is_person = self.obj_det.is_person_present(frame)
                    is_vehicle = self.obj_det.is_vehicle_present(frame)
                    
                    final_label = label
                    if is_person and not is_vehicle:
                        final_label = "PedestrianLittering"
                    elif is_vehicle and not is_person:
                        final_label = "VehicleLittering"
                    elif not is_person and not is_vehicle:
                        # No agent? Skip window.
                        consecutive_hits = 0
                        continue

                    label_threshold = self._label_threshold(final_label)
                    effective_threshold = min(label_threshold, 0.85)
                    
                    if conf >= effective_threshold:
                        if final_label == consecutive_label:
                            consecutive_hits += 1
                        else:
                            consecutive_label = final_label
                            consecutive_hits = 1
                    else:
                        consecutive_label = ""
                        consecutive_hits = 0

                    action_gate_passed = self._passes_action_gate(window, final_label)
                    high_conf_override = conf >= min(0.97, label_threshold + 0.10)

                    if (
                        consecutive_hits >= self.CONSENSUS_MIN_HITS
                        and (time.time() - last_violation_time) > cooldown
                        and (action_gate_passed or high_conf_override)
                    ):
                        print(f"  [LIVE DETECTED] {final_label} ({conf:.1%})")
                        violation_type = self._violation_label(final_label)

                        self._counter += 1
                        snapshot = frame.copy()
                        plate_crop, plate_text, plate_bbox = None, "", None
                        result = self.plate_det.detect(snapshot)
                        if result:
                            plate_crop, plate_text, plate_bbox = result
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
                            plate_bbox=plate_bbox,
                        )
                        self._save_assets(violation)
                        violations.append(violation)
                        if callback:
                            callback(violation)
                        last_violation_time = time.time()
                        consecutive_label = ""
                        consecutive_hits = 0

                frame_idx += 1

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
        # Save enhanced snapshot with optional plate detection overlay
        snapshot_viz = violation.snapshot.copy()
        
        # If we detected a plate, add visual indicator on snapshot
        if violation.plate_text and violation.plate_bbox is not None:
            x1, y1, x2, y2 = violation.plate_bbox
            # Draw green box around detected plate area
            cv2.rectangle(snapshot_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Add label above box
            label = f"Plate: {violation.plate_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_size = cv2.getTextSize(label, font, 0.7, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(snapshot_viz, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, label_y + 5), (0, 255, 0), -1)
            cv2.putText(snapshot_viz, label, (x1 + 5, label_y), font, 0.7, (0, 0, 0), 2)
        
        cv2.imwrite(
            str(self.out_dir / f"violation_{violation.id:04d}_snapshot.jpg"),
            snapshot_viz,
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

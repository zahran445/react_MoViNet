import os
import sys
import json
import base64
import threading
import time
from collections import deque
from pathlib import Path
from datetime import datetime
import traceback

# CRITICAL: Set FFMPEG options BEFORE importing cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import urllib.parse

from flask import (
    Flask, render_template_string, jsonify,
    request, redirect, url_for, send_from_directory, Response,
)
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import func

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'sawn.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "outputs" / "violations")
app.config["UPLOAD_VIDEO_FOLDER"] = str(BASE_DIR / "outputs" / "uploads")
app.config["SECRET_KEY"] = "react-secret-2025"

db = SQLAlchemy(app)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
Path(app.config["UPLOAD_VIDEO_FOLDER"]).mkdir(parents=True, exist_ok=True)

processing_jobs = {} # Global job tracker
live_stream_state = {
    "running": False,
    "lock": threading.Lock(),
}
live_detector = None
live_face_detector = None
live_sessions = {}
LITTERING_LABELS = {"PedestrianLittering", "VehicleLittering"}


def to_littering_label(label: str) -> str:
    if "Pedestrian" in (label or ""):
        return "PedestrianLittering"
    return "VehicleLittering"




def live_detection_decision(detector, window, raw_label: str, confidence: float):
    """Decide whether the current live window is a genuine littering event.

    Strategy:
    - Both the model confidence AND the motion gate must pass.  There is
      no fallback path so that idle standing / looking at the camera cannot
      trigger a detection even at high confidence (the model has no
      background class, so it always assigns the frame to one of the two
      littering classes).
    - For pedestrian events the motion must also resemble a throw-like burst
      (sharp transient rise-then-fall), ruling out sustained walking motion.
    """
    # --- SMART AGENT MAPPING ---
    # We use YOLO to tell us what the agent actually is, regardless of MoViNet's guess.
    # This fixes cases where the model confuses Pedestrian vs Vehicle actions.
    is_person = detector.obj_det.is_person_present(window[-1])
    is_vehicle = detector.obj_det.is_vehicle_present(window[-1])
    
    final_label = raw_label
    if is_person and not is_vehicle:
        final_label = "PedestrianLittering"
    elif is_vehicle and not is_person:
        final_label = "VehicleLittering"
    elif not is_person and not is_vehicle:
        # No agent detected? Likely a false positive from lights/branch movement.
        return False, "None"
    
    # Now that we've mapped the agent correctly, apply the corresponding threshold and action gate.
    label_threshold = detector._label_threshold(final_label)
    # Important: we use a slightly lower threshold (0.85) if we have strong YOLO presence confirmation.
    effective_threshold = min(label_threshold, 0.85) 
    
    action_ok = detector._passes_action_gate(window, final_label)

    if confidence > effective_threshold and action_ok:
        return True, to_littering_label(final_label)

    # No fallback path — require both confidence threshold AND motion gate.
    return False, "None"


def get_live_face_detector():
    global live_face_detector
    if live_face_detector is None:
        from utils.detector import FaceDetector
        live_face_detector = FaceDetector()
    return live_face_detector


def get_live_detector():
    global live_detector
    if live_detector is None:
        from utils.detector import SAWNDetector
        live_detector = SAWNDetector(
            str(BASE_DIR / "models" / "movinet" / "movinet_best.pt"),
            str(BASE_DIR / "models" / "yolo" / "plates_yolov8" / "weights" / "best.pt"),
        )
    return live_detector


def save_video_clip(frames, fps, dst_path):
    if not frames:
        return False

    height, width = frames[0].shape[:2]
    target_height = 480
    target_width = int(target_height * (width / max(1, height)))
    fps = int(fps or 30)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (target_width, target_height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))

    if not writer.isOpened():
        return False

    for frame in frames:
        writer.write(cv2.resize(frame, (target_width, target_height)))
    writer.release()
    return True


def decode_data_url_image(data_url):
    if not data_url or not isinstance(data_url, str):
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


    except Exception:
        return data


def crypt_url(data: str, key: str = "abcd1234") -> str:
    """Simple XOR encryption/decryption."""
    if not data: return ""
    res = "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
    return res


def decode_url(data: str) -> str:
    """Base64 decode + XOR decrypt."""
    if not data: return ""
    try:
        # Check if it looks like base64
        raw = base64.b64decode(data).decode()
        return crypt_url(raw)
    except Exception:
        return data


def _safe_rtsp_url(url: str) -> str:
    """Ensures password in RTSP URL is properly escaped for OpenCV."""
    if not url or not url.startswith("rtsp://"):
        return url
    try:
        # Standard parser might struggle with multiple '@', so we do a quick check
        if url.count("@") > 1:
            # Find the last '@' which separates credentials from host
            creds_part, host_part = url[7:].rsplit("@", 1)
            if ":" in creds_part:
                user, passwd = creds_part.split(":", 1)
                # Unquote once before quoting to prevent double-encoding (%40 -> %2540)
                passwd = urllib.parse.unquote(passwd)
                return f"rtsp://{user}:{urllib.parse.quote(passwd)}@{host_part}"
        else:
            parsed = urllib.parse.urlparse(url)
            if parsed.password:
                user = parsed.username
                pw = urllib.parse.unquote(parsed.password) # Fix double encoding
                pw = urllib.parse.quote(pw)
                new_netloc = f"{user}:{pw}@{parsed.hostname}"
                if parsed.port: new_netloc += f":{parsed.port}"
                return parsed._replace(netloc=new_netloc).geturl()
    except Exception:
        pass
    return url


def get_live_session(session_id: str):
    with live_stream_state["lock"]:
        session = live_sessions.get(session_id)
        if session is None:
            session = {
                "recent_frames": deque(maxlen=45),
                "clip_active": False,
                "clip_frames": [],
                "clip_target_frames": 0,
                "last_violation_time": 0.0,
                "last_analysis_time": 0.0,
                "last_label": "None",
                "last_conf": 0.0,
                "analysis_count": 0,
                "analysis_stride": 3,
                "fps_estimate": 5,
            }
            live_sessions[session_id] = session
        return session


def persist_live_violation(detector, clip_frames, fps, snapshot, plate_crop, face_crop, plate_text, plate_bbox, violation_type, confidence):
    if snapshot is None or not clip_frames:
        return None

    if violation_type not in LITTERING_LABELS:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_viz = snapshot.copy()
    if plate_bbox is not None:
        x1, y1, x2, y2 = plate_bbox
        cv2.rectangle(snapshot_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if plate_text:
            label = f"Plate: {plate_text}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(snapshot_viz, (x1, label_y - label_size[1] - 10), (x1 + label_size[0] + 10, label_y + 5), (0, 255, 0), -1)
            cv2.putText(snapshot_viz, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    with app.app_context():
        record = ViolationRecord(
            timestamp=timestamp,
            violation_type=violation_type,
            confidence=float(confidence),
            snapshot_path="",
            face_path="",
            plate_path="",
            video_path="",
            plate_text=plate_text or "",
            status="PENDING",
        )
        db.session.add(record)
        db.session.commit()

        base_name = f"violation_{record.id:04d}"
        snapshot_path = Path(app.config["UPLOAD_FOLDER"]) / f"{base_name}_snapshot.jpg"
        face_path = Path(app.config["UPLOAD_FOLDER"]) / f"{base_name}_face.jpg"
        plate_path = Path(app.config["UPLOAD_FOLDER"]) / f"{base_name}_plate.jpg"
        clip_path = Path(app.config["UPLOAD_FOLDER"]) / f"{base_name}_clip.mp4"

        cv2.imwrite(str(snapshot_path), snapshot_viz, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if face_crop is not None:
            cv2.imwrite(str(face_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if plate_crop is not None:
            cv2.imwrite(str(plate_path), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

        save_video_clip(clip_frames, fps, str(clip_path))

        record.snapshot_path = str(snapshot_path)
        record.face_path = str(face_path) if face_crop is not None else ""
        record.plate_path = str(plate_path) if plate_crop is not None else ""
        record.video_path = str(clip_path)
        db.session.commit()

    return record

# ── Database Models ────────────────────────────────────────────────────────

class ViolationRecord(db.Model):
    __tablename__ = "violations"
    id             = db.Column(db.Integer, primary_key=True)
    timestamp      = db.Column(db.String(30), nullable=False)
    violation_type = db.Column(db.String(20), nullable=False)
    confidence     = db.Column(db.Float, default=0.0)
    snapshot_path  = db.Column(db.String(200), default="")
    face_path      = db.Column(db.String(200), default="")
    plate_path     = db.Column(db.String(200), default="")
    video_path     = db.Column(db.String(200), default="")
    plate_text     = db.Column(db.String(50),  default="")
    ticket_sent    = db.Column(db.Boolean, default=False)
    status         = db.Column(db.String(20), default="PENDING")

    def to_dict(self):
        return {
            "id":             self.id,
            "timestamp":      self.timestamp,
            "violation_type": self.violation_type,
            "confidence":     round(self.confidence * 100, 1),
            "plate_text":     self.plate_text or "Unknown",
            "status":         self.status,
            "severity":       "HIGH" if self.confidence > 0.8 else "MEDIUM",
            "snapshot_url":   f"/assets/{Path(self.snapshot_path).name}" if self.snapshot_path else "",
            "face_url":       f"/assets/{Path(self.face_path).name}" if self.face_path else "",
            "plate_url":      f"/assets/{Path(self.plate_path).name}" if self.plate_path else "",
            "video_url":      f"/assets/{Path(self.video_path).name}" if self.video_path else "",
        }

with app.app_context():
    db.create_all()

# ── HTML Templates (REACT Rebranding) ──────────────────────────────────────

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REACT — AI Monitoring System</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0c; --card-bg: #121214; --sidebar-bg: #0d0d0f;
            --accent: #10b981; --accent-glow: rgba(16, 185, 129, 0.2);
            --text-main: #ffffff; --text-dim: #94a3b8; --border: #1f2937;
            --red: #ef4444; --yellow: #f59e0b; --blue: #3b82f6;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--bg); color: var(--text-main); font-family: 'Outfit', sans-serif; display: flex; min-height: 100vh; overflow-x: hidden; }

        .sidebar { width: 260px; background: var(--sidebar-bg); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 32px 0; position: fixed; height: 100vh; }
        .logo { padding: 0 24px 40px; display: flex; align-items: center; gap: 12px; }
        .logo-box { width: 32px; height: 32px; background: var(--accent); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #000; font-weight: 700; }
        .logo h2 { font-size: 20px; letter-spacing: -0.5px; }

        .nav-item { display: flex; align-items: center; gap: 12px; padding: 14px 24px; color: var(--text-dim); text-decoration: none; font-size: 15px; border-left: 3px solid transparent; }
        .nav-item.active { color: var(--accent); background: rgba(16, 185, 129, 0.05); border-left-color: var(--accent); font-weight: 600; }

        .main { margin-left: 260px; flex: 1; display: flex; flex-direction: column; }
        .header { padding: 24px 40px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); background: rgba(10, 10, 12, 0.8); backdrop-filter: blur(10px); position: sticky; top: 0; z-index: 100; }
        
        .btn { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .btn-accent { background: var(--accent); color: #000; }
        .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text-main); }
        .btn-red { background: var(--red); color: #fff; }
        .btn-blue { background: var(--blue); color: #fff; }

        .content { padding: 40px; max-width: 1400px; }
        .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 24px; margin-bottom: 40px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }
        .stat-label { color: var(--text-dim); font-size: 14px; margin-bottom: 12px; }
        .stat-value { font-size: 32px; font-weight: 700; }

        .table-container { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; }
        th { background: rgba(255,255,255,0.02); padding: 16px 24px; text-align: left; font-size: 13px; color: var(--text-dim); border-bottom: 1px solid var(--border); }
        td { padding: 20px 24px; border-bottom: 1px solid var(--border); font-size: 14px; }

        .status-badge { padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
        .status-accepted { background: rgba(16, 185, 129, 0.1); color: var(--accent); border: 1px solid var(--accent); }
        .status-rejected { background: rgba(239, 68, 68, 0.1); color: var(--red); border: 1px solid var(--red); }
        .status-pending { background: rgba(245, 158, 11, 0.1); color: var(--yellow); border: 1px solid var(--yellow); }

        .modal-overlay { display: none; position: fixed; inset: 0; padding: 24px; background: rgba(0,0,0,0.85); z-index: 1000; align-items: center; justify-content: center; backdrop-filter: blur(4px); overflow-y: auto; }
        .modal-overlay.open { display: flex; }
        .modal { background: var(--card-bg); border: 1px solid var(--border); border-radius: 24px; padding: 32px; width: min(960px, 100%); max-width: 100%; max-height: calc(100vh - 48px); overflow-y: auto; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 24px; }
        .modal-actions-row { display: flex; justify-content: flex-end; gap: 16px; margin-top: 32px; flex-wrap: wrap; }
        .video-player { width: 100%; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 24px; background: #000; overflow: hidden; aspect-ratio: 16 / 9; max-height: min(42vh, 350px); }
        .asset-grid { display: flex; gap: 16px; margin-bottom: 24px; align-items: stretch; }
        .asset-box { background: #000; border-radius: 12px; height: 220px; overflow: hidden; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; color: var(--text-dim); min-width: 0; }
        .asset-box img { width: 100%; height: 100%; object-fit: contain; background: #000; }
        .asset-box-snap { flex: 0.9; }
        .asset-box-plate { flex: 1.8; height: 220px; padding: 10px 16px; }
        .asset-box-plate img { object-position: center; }
        .detail-card { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .detail-item label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; display: block; }
        .detail-item span { font-weight: 600; font-size: 15px; }

        @media (max-width: 900px) {
            .modal-overlay { padding: 16px; align-items: flex-start; }
            .modal { padding: 24px; max-height: calc(100vh - 32px); }
            .asset-grid { gap: 12px; }
            .asset-box { height: 190px; }
            .asset-box-plate { height: 190px; }
        }

        @media (max-width: 640px) {
            .modal-overlay { padding: 8px; }
            .modal { padding: 16px; border-radius: 16px; max-height: calc(100vh - 16px); }
            .modal-header { flex-direction: column; align-items: stretch; }
            .modal-header .btn { width: 100%; justify-content: center; }
            .video-player { margin-bottom: 16px; max-height: min(34vh, 220px); }
            .asset-grid { flex-direction: column; gap: 12px; }
            .asset-box { height: 140px; }
            .asset-box-snap { flex: none; }
            .asset-box-plate { flex: none; height: 180px; padding: 8px; }
            .detail-card { grid-template-columns: 1fr; gap: 12px; padding: 14px; }
            .modal-actions-row { margin-top: 20px; }
            .modal-actions-row .btn { width: 100%; justify-content: center; }
            #modal-actions { width: 100%; display: flex; gap: 12px; flex-wrap: wrap; }
            #modal-actions > * { flex: 1 1 100%; text-align: center; justify-content: center; }
        }

        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
        .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .active-job-card { background: rgba(16, 185, 129, 0.05); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }
        .job-info { display: flex; align-items: center; justify-content: space-between; width: 100%; }
        .spinner { width: 14px; height: 14px; border: 2px solid rgba(16, 185, 129, 0.3); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Dashed Progress Bar */
        .progress-container { height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; position: relative; }
        .progress-bar { height: 100%; width: 100%; background: linear-gradient(90deg, transparent, var(--accent), transparent); background-size: 200% 100%; animation: moveGradient 2s linear infinite; position: relative; }
        .progress-bar::after { content: ''; position: absolute; inset: 0; background-image: linear-gradient(90deg, var(--card-bg) 2px, transparent 2px); background-size: 8px 100%; }
        @keyframes moveGradient { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

        .live-shell { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.6fr); gap: 24px; align-items: start; }
        .live-panel, .live-side { background: var(--card-bg); border: 1px solid var(--border); border-radius: 20px; overflow: hidden; }
        .live-panel { padding: 16px; }
        .live-frame { width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 14px; overflow: hidden; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; }
        .live-frame img { width: 100%; height: 100%; object-fit: contain; }
        .live-side { padding: 24px; display: flex; flex-direction: column; gap: 16px; }
        .live-metric { padding: 16px; border-radius: 14px; background: rgba(255,255,255,0.03); border: 1px solid var(--border); }
        .live-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 6px; }
        .live-value { font-size: 28px; font-weight: 700; }
        .live-note { color: var(--text-dim); font-size: 14px; line-height: 1.5; }

        @media (max-width: 980px) {
            .live-shell { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="logo"><div class="logo-box">R</div><h2>REACT</h2></div>
        <a href="/" class="nav-item {{ 'active' if active_page == 'home' else '' }}">🏠 Dashboard</a>
        <a href="/live" class="nav-item {{ 'active' if active_page == 'live' else '' }}">📷 Live Stream</a>
        <a href="/upload" class="nav-item">📤 Upload Video</a>
    </nav>

    <div class="main">
        <header class="header">
            <h1 id="page-title">{{ title }}</h1>
            <div style="display:flex; gap:12px">
                <a href="/live" class="btn btn-blue">📷 Live Stream</a>
                <a href="/upload" class="btn btn-accent">📤 Upload</a>
                <button class="btn btn-outline">Logout</button>
            </div>
        </header>

        <main class="content">
            {% block content %}{% endblock %}
        </main>
    </div>

    <!-- UI Modal -->
    <div class="modal-overlay" id="detailModal">
        <div class="modal">
            <div class="modal-header">
                <h3 id="modal-title" style="font-size:20px">Evidence</h3>
                <button class="btn btn-outline" onclick="closeModal()">← Back to Log</button>
            </div>
            
            <div id="modal-v-container" class="video-player"></div>
            
            <div id="modal-assets" class="asset-grid">
                <div class="asset-box asset-box-snap" id="m-snap"></div>
                <div class="asset-box asset-box-plate" id="m-plate"></div>
            </div>

            <div id="modal-details" class="detail-card"></div>

            <div class="modal-actions-row">
                <button class="btn btn-outline" onclick="closeModal()">Close</button>
                <div id="modal-actions" style="display:flex; gap:12px"></div>
            </div>
        </div>
    </div>

    <script>
        let curId = null;
        function openModal(d, forReview){
            curId = d.id;
            const status = (d.status || "").toUpperCase().trim();
            const vWrap = document.getElementById('modal-v-container');
            const assetWrap = document.getElementById('modal-assets');
            const detailWrap = document.getElementById('modal-details');
            const actionWrap = document.getElementById('modal-actions');
            const title = document.getElementById('modal-title');

            title.textContent = forReview ? "Violation Review" : "Evidence Playback";
            vWrap.innerHTML = d.video_url ? `<video controls autoplay loop muted width="100%" height="100%"><source src="${d.video_url}" type="video/mp4"></video>` : 'No Video';

            if(forReview){
                assetWrap.style.display = 'flex';
                detailWrap.style.display = 'grid';
                document.getElementById('m-snap').innerHTML = d.snapshot_url ? `<img src="${d.snapshot_url}">` : '';
                document.getElementById('m-plate').innerHTML = d.plate_url ? `<img src="${d.plate_url}">` : 'No Plate';
                detailWrap.innerHTML = `
                    <div class="detail-item"><label>Reg No.</label><span>${d.plate_text}</span></div>
                    <div class="detail-item"><label>AI Conf.</label><span>${d.confidence}%</span></div>
                `;
                if(status === 'PENDING'){
                    actionWrap.innerHTML = `<button class="btn btn-red" onclick="rev('REJECTED')">✖ Reject</button><button class="btn btn-accent" onclick="rev('ACCEPTED')">✔ Accept</button>`;
                } else {
                    actionWrap.innerHTML = `<span class="status-badge status-${status.toLowerCase()}">${status}</span>`;
                }
            } else {
                assetWrap.style.display = 'none';
                detailWrap.style.display = 'none';
                actionWrap.innerHTML = status !== 'PENDING' ? `<span class="status-badge status-${status.toLowerCase()}">${status}</span>` : '';
            }
            document.getElementById('detailModal').classList.add('open');
        }
        function closeModal(){ document.getElementById('detailModal').classList.remove('open'); document.getElementById('modal-v-container').innerHTML = ''; }
        function rev(s){
            const ep = s === 'ACCEPTED' ? '/api/accept_violation/' : '/api/reject_violation/';
            fetch(ep+curId, {method:'POST'}).then(()=>location.reload());
        }
    </script>
</body>
</html>
"""

HOME_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="stats-row" id="s-row"></div>

<div id="active-jobs-container"></div>

<div class="table-container">
    <div style="padding: 20px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
        <h3 style="font-size: 16px; font-weight: 600;">Detection Log</h3>
        <div style="display:flex; align-items:center; gap:12px">
            <span id="log-count" style="font-size: 12px; color: var(--text-dim);"></span>
            <button class="btn btn-red" style="padding:6px 14px; font-size:12px" onclick="clearAllLogs()">🗑 Clear All</button>
        </div>
    </div>
    <table>
        <thead><tr><th>Timestamp</th><th>Reg No.</th><th>Conf.</th><th>Status</th><th>Play</th><th>Review</th><th>Delete</th></tr></thead>
        <tbody id="rows"></tbody>
    </table>
</div>
<script>
    async function up(){
        const s = await (await fetch('/api/stats')).json();
        const jobs = await (await fetch('/api/processing_status')).json();
        
        document.getElementById('s-row').innerHTML = `
            <div class="stat-card"><div class="stat-label">Total</div><div class="stat-value">${s.total}</div></div>
            <div class="stat-card" style="color:var(--accent)"><div class="stat-label">Accepted</div><div class="stat-value">${s.accepted}</div></div>
            <div class="stat-card" style="color:var(--red)"><div class="stat-label">Rejected</div><div class="stat-value">${s.rejected}</div></div>
            <div class="stat-card"><div class="stat-label">Avg Conf.</div><div class="stat-value">${s.avg_confidence}%</div></div>
            <div class="stat-card"><div class="stat-label">Last Detection</div><div class="stat-value" style="font-size:14px; padding-top:6px">${s.last_detection || '—'}</div></div>
        `;

        const activeJobsHtml = jobs.jobs.map(j => {
            const elapsed = Math.floor(Date.now()/1000 - j.start_time);
            const m = Math.floor(elapsed/60).toString().padStart(2, '0');
            const s = (elapsed%60).toString().padStart(2, '0');
            const pct = j.progress || 0;
            return `
                <div class="active-job-card">
                    <div class="job-info">
                        <div style="display:flex; align-items:center; gap:10px">
                            <div class="spinner"></div>
                            <div style="font-size: 14px; font-weight: 600; color: var(--accent);">AI Processing (${pct}%)</div>
                            <div style="font-size: 12px; color: var(--text-dim); max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${j.filename}</div>
                        </div>
                        <div style="font-size: 13px; font-family: monospace; color: var(--accent); font-weight:700">${m}:${s}s elapsed</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: ${pct}%; background: var(--accent); animation: none"></div>
                    </div>
                </div>
            `;
        }).join('');
        document.getElementById('active-jobs-container').innerHTML = activeJobsHtml;

        const vs = await (await fetch('/api/violations?limit=50')).json();
        document.getElementById('log-count').textContent = `${vs.length} entries`;
        document.getElementById('rows').innerHTML = vs.map(v => `
            <tr>
                <td>${v.timestamp}</td>
                <td style="color:var(--accent); font-weight:700">${v.plate_text}</td>
                <td>${v.confidence}%</td>
                <td><span class="status-badge status-${v.status.toLowerCase()}">${v.status}</span></td>
                <td><button class="btn btn-outline" style="padding:6px" onclick='openModal(${JSON.stringify(v)}, false)'>▶ Play</button></td>
                <td><button class="btn btn-accent" style="padding:6px 12px; font-size:12px" onclick='openModal(${JSON.stringify(v)}, true)'>${v.status.toUpperCase()=='PENDING'?'Review':'View'}</button></td>
                <td><button class="btn btn-red" style="padding:6px 10px; font-size:12px" onclick='deleteLog(${v.id})'>🗑</button></td>
            </tr>
        `).join('') || '<tr><td colspan="7" style="text-align:center; padding:50px">No detections</td></tr>';
    }
    async function deleteLog(id) {
        if (!confirm('Delete this entry?')) return;
        await fetch('/api/violations/' + id, {method: 'DELETE'});
        up();
    }
    async function clearAllLogs() {
        if (!confirm('Delete ALL detection log entries? This cannot be undone.')) return;
        await fetch('/api/violations/all', {method: 'DELETE'});
        up();
    }
    up(); setInterval(up, 3000);
</script>
""")

@app.route("/")
def home(): return render_template_string(HOME_HTML, title="Dashboard", active_page="home")


LIVE_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="live-shell">
    <div class="live-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
            <div>
                <h2 style="font-size:24px; margin-bottom:4px;">Live Webcam Stream</h2>
                <p style="color: var(--text-dim); font-size:14px;">Choose a browser webcam or an RTSP connected camera and run the live models on the selected source.</p>
            </div>
            <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
                <select id="stream-mode" class="btn btn-outline" style="min-width:180px; padding:10px 14px; color:var(--text-main); background:transparent;">
                    <option value="webcam">Browser Webcam</option>
                    <option value="rtsp">RTSP Camera</option>
                </select>
                <select id="camera-select" class="btn btn-outline" style="min-width:220px; padding:10px 14px; color:var(--text-main); background:transparent;"></select>
                <input id="rtsp-url" type="text" placeholder="rtsp://user:pass@host:554/stream" style="min-width:320px; padding:10px 14px; border-radius:8px; border:1px solid var(--border); background:transparent; color:var(--text-main); display:none;">
                <button class="btn btn-blue" onclick="startLive()">▶ Start Live</button>
                <button class="btn btn-outline" onclick="stopLive()">■ Stop</button>
            </div>
        </div>

        <div class="live-frame">
            <video id="live-video" autoplay muted playsinline style="width:100%; height:100%; object-fit:contain; background:#000;"></video>
            <img id="rtsp-video" alt="RTSP live stream" style="width:100%; height:100%; object-fit:contain; background:#000; display:none;">
            <canvas id="live-canvas" style="display:none;"></canvas>
        </div>
    </div>

    <aside class="live-side">
        <div class="live-metric">
            <div class="live-label">Status</div>
            <div class="live-value" id="live-status">Idle</div>
        </div>
        <div class="live-metric">
            <div class="live-label">Latest Label</div>
            <div class="live-value" id="live-label">--</div>
        </div>
        <div class="live-metric">
            <div class="live-label">Confidence</div>
            <div class="live-value" id="live-conf">0%</div>
        </div>
        <div class="live-metric">
            <div class="live-label">Plate</div>
            <div class="live-value" id="live-plate" style="font-size:18px;">--</div>
        </div>
        <div class="live-metric">
            <div class="live-label">How it works</div>
            <div class="live-note">The browser shows the camera feed directly. Every sampled frame is analyzed on the server with the SAWN classifier, plate detector, and face detector. If a violation is detected, a 5-second clip is saved and appears in the dashboard list.</div>
        </div>
    </aside>
</div>

<script>
    const liveStatus = document.getElementById('live-status');
    const liveLabel = document.getElementById('live-label');
    const liveConf = document.getElementById('live-conf');
    const livePlate = document.getElementById('live-plate');
    const liveVideo = document.getElementById('live-video');
    const rtspVideo = document.getElementById('rtsp-video');
    const liveCanvas = document.getElementById('live-canvas');
    const liveCtx = liveCanvas.getContext('2d');
    const streamMode = document.getElementById('stream-mode');
    const cameraSelect = document.getElementById('camera-select');
    const rtspUrlInput = document.getElementById('rtsp-url');

    let liveStream = null;
    let liveTimer = null;
    let sessionId = crypto.randomUUID();
    let sending = false;
    let selectedDeviceId = '';
    let currentRtspUrl = '';

    function updateModeUi(){
        const mode = streamMode.value;
        cameraSelect.style.display = mode === 'webcam' ? 'inline-flex' : 'none';
        rtspUrlInput.style.display = mode === 'rtsp' ? 'inline-block' : 'none';
        liveVideo.style.display = mode === 'webcam' ? 'block' : 'none';
        rtspVideo.style.display = mode === 'rtsp' ? 'block' : 'none';
    }

    async function loadCameraDevices(){
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(device => device.kind === 'videoinput');
            cameraSelect.innerHTML = cameras.map((device, index) => {
                const label = device.label || `Camera ${index + 1}`;
                return `<option value="${device.deviceId}">${label}</option>`;
            }).join('');
            if (cameras.length) {
                selectedDeviceId = cameras[0].deviceId;
                cameraSelect.value = selectedDeviceId;
            }
        } catch (err) {
            console.error(err);
        }
    }

    function cryptUrl(data, key = "abcd1234") {
        let out = "";
        for (let i = 0; i < data.length; i++) {
            out += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
        }
        return btoa(out);
    }

    async function startLive(){
        stopLive();
        const mode = streamMode.value;
        try {
            if (mode === 'webcam') {
                const constraints = selectedDeviceId
                    ? { video: { deviceId: { exact: selectedDeviceId } }, audio: false }
                    : { video: true, audio: false };
                liveStream = await navigator.mediaDevices.getUserMedia(constraints);
                liveVideo.srcObject = liveStream;
                liveStatus.textContent = 'Running (webcam)';
                liveTimer = setInterval(sendFrame, 200);
            } else {
                const rtspUrl = rtspUrlInput.value.trim();
                if (!rtspUrl) {
                    liveStatus.textContent = 'RTSP URL required';
                    return;
                }
                currentRtspUrl = rtspUrl;
                // Camera networking begins here: the browser opens the RTSP live feed served by Flask.
                const encUrl = cryptUrl(rtspUrl);
                console.log("[RTSP] Feed request initiated: (encrypted)", encUrl);
                
                // Add an error handler to retry if the browser fails to load the MJPEG stream
                rtspVideo.onerror = () => {
                    console.warn("[RTSP] Browser stream error. Retrying in 2s...");
                    setTimeout(() => {
                        if (currentRtspUrl) rtspVideo.src = `/live_feed_rtsp?url=${encodeURIComponent(encUrl)}&ts=${Date.now()}`;
                    }, 2000);
                };
                
                rtspVideo.src = `/live_feed_rtsp?url=${encodeURIComponent(encUrl)}&ts=${Date.now()}`;
                liveStatus.textContent = 'Running (RTSP)';
            }
        } catch (err) {
            liveStatus.textContent = 'Camera blocked';
            console.error(err);
        }
    }

    function stopLive(){
        if (liveTimer) {
            clearInterval(liveTimer);
            liveTimer = null;
        }
        if (liveStream) {
            liveStream.getTracks().forEach(track => track.stop());
            liveStream = null;
        }
        liveVideo.srcObject = null;
        rtspVideo.src = '';
        liveStatus.textContent = 'Stopped';
        liveLabel.textContent = '--';
        liveConf.textContent = '0%';
        livePlate.textContent = '--';
    }

    async function sendFrame(){
        if (!liveVideo.videoWidth || !liveVideo.videoHeight || sending) return;
        sending = true;
        try {
            // Camera networking continues here: a sampled webcam frame is encoded and sent to the server.
            liveCanvas.width = liveVideo.videoWidth;
            liveCanvas.height = liveVideo.videoHeight;
            liveCtx.drawImage(liveVideo, 0, 0, liveCanvas.width, liveCanvas.height);
            const dataUrl = liveCanvas.toDataURL('image/jpeg', 0.75);
            // Camera networking continues here: the sampled browser frame is uploaded to the server for inference.
            const res = await fetch('/api/live_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ session_id: sessionId, image: dataUrl })
            });
            if (!res.ok) return;
            const data = await res.json();
            if (data.status) liveStatus.textContent = data.status;
            if (data.label) liveLabel.textContent = data.label;
            if (typeof data.confidence === 'number') liveConf.textContent = `${data.confidence.toFixed(1)}%`;
            if (data.plate_text !== undefined) livePlate.textContent = data.plate_text || '--';
        } catch (err) {
            console.error(err);
        } finally {
            sending = false;
        }
    }

    cameraSelect.addEventListener('change', async () => {
        selectedDeviceId = cameraSelect.value;
        if (streamMode.value === 'webcam') {
            stopLive();
            await startLive();
        }
    });

    streamMode.addEventListener('change', () => {
        updateModeUi();
        stopLive();
    });

    navigator.mediaDevices?.addEventListener?.('devicechange', loadCameraDevices);
    updateModeUi();
    loadCameraDevices();

</script>
""")


@app.route("/live")
def live():
    return render_template_string(LIVE_HTML, title="Live Stream", active_page="live")


def _encode_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buffer.tobytes()


def _rtsp_cap(url: str):
    # Camera networking on the server side happens here: OpenCV opens the RTSP URL as a live source.
    print(f"\n[RTSP] >>> Connecting to: {url[:15]}... (password hidden)")
    safe_url = _safe_rtsp_url(url)
    
    if hasattr(cv2, "CAP_FFMPEG"):
        print(f"[RTSP] Using FFMPEG backend with TCP transport...")
        cap = cv2.VideoCapture(safe_url, cv2.CAP_FFMPEG)
        # 5 second timeout for opening
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        # Reduce buffer size to minimum to avoid lag/black frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Stabilization period: wait for the encoder to sync
        time.sleep(1.0) 
    else:
        print(f"[RTSP] Using default backend...")
        cap = cv2.VideoCapture(safe_url)
        
    if cap.isOpened():
        print(f"[RTSP] SUCCESS: Stream opened! Processing frames...")
    else:
        print(f"[RTSP] ERROR: Failed to open stream at {safe_url[:20]}...")
        
    return cap


def _stream_live_source(cap, source_name: str):
    detector = get_live_detector()
    face_detector = get_live_face_detector()

    if not cap.isOpened():
        print(f"[RTSP] Failed to open: {source_name}")
        frame = 255 * np.ones((480, 854, 3), dtype='uint8')
        cv2.putText(frame, f'Could not open {source_name}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        encoded = _encode_frame(frame)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        return

    # Initial Sync: Discard the first 200 frames of "junk" buffer data to catch a clean Keyframe
    print(f"[RTSP] Super-Syncing with camera... discarding 200 frames.")
    for _ in range(200): cap.grab()

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    analysis_stride = max(1, fps // 2)
    recent_frames = deque(maxlen=max(int(fps * 2), 30))
    clip_session = None
    last_label = 'None'
    last_conf = 0.0
    last_violation_time = 0.0
    cooldown = 5.0
    frame_index = 0

    try:
        error_count = 0
        while True:
            # Buffer clearing: grab many frames to reach the most recent one (low latency)
            for _ in range(10): 
                cap.grab() 
            
            success, frame = cap.read()
            if not success:
                error_count += 1
                if error_count > 60: # Stop if we can't read for ~3 seconds
                    print(f"[RTSP] Fatal: Persistent read error. Closing stream.")
                    break
                time.sleep(0.01) # Small delay to let the camera buffer refill
                continue
            
            error_count = 0 # Reset on every successful frame
            recent_frames.append(frame.copy())

            if clip_session is not None:
                clip_session["frames"].append(frame.copy())
                if len(clip_session["frames"]) >= clip_session["target_frames"]:
                    persist_live_violation(
                        detector=detector,
                        clip_frames=clip_session["frames"],
                        fps=fps,
                        snapshot=clip_session["snapshot"],
                        plate_crop=clip_session["plate_crop"],
                        face_crop=clip_session["face_crop"],
                        plate_text=clip_session["plate_text"],
                        plate_bbox=clip_session["plate_bbox"],
                        violation_type=clip_session["violation_type"],
                        confidence=clip_session["confidence"],
                    )
                    clip_session = None
                    last_violation_time = time.time()

            display = frame.copy()
            if len(recent_frames) >= 16 and frame_index % analysis_stride == 0:
                window = list(recent_frames)[-16:]
                label, conf = detector.classifier.predict_segment(window)
                is_detected, detected_label = live_detection_decision(detector, window, label, conf)

                if is_detected:
                    last_label = detected_label
                    last_conf = conf
                    cv2.putText(display, f'{last_label} ({conf:.1%})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    last_label = 'None'
                    last_conf = 0.0
                    cv2.putText(display, 'None (0.0%)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

                if clip_session is None and is_detected and (time.time() - last_violation_time) > cooldown:
                    snapshot = frame.copy()
                    plate_crop = None
                    plate_text = ""
                    plate_bbox = None
                    face_crop = face_detector.detect(snapshot)
                    result = detector.plate_det.detect(snapshot)
                    if result:
                        plate_crop, plate_text, plate_bbox = result
                        if plate_bbox is not None:
                            x1, y1, x2, y2 = plate_bbox
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if plate_text:
                                cv2.putText(display, plate_text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    clip_session = {
                        "frames": list(recent_frames),
                        "target_frames": max(int(fps * 5), len(recent_frames)),
                        "snapshot": snapshot,
                        "face_crop": face_crop,
                        "plate_crop": plate_crop,
                        "plate_text": plate_text if detector._is_plausible_plate_text(plate_text) else "",
                        "plate_bbox": plate_bbox,
                        "violation_type": detected_label,
                        "confidence": conf,
                    }
                    last_violation_time = time.time()
                    cv2.putText(display, 'Saving 5s clip...', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(display, source_name, (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if clip_session is not None:
                cv2.putText(display, f"REC {len(clip_session['frames'])}/{clip_session['target_frames']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            encoded = _encode_frame(display)
            if not encoded:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
            frame_index += 1
    finally:
        cap.release()


def live_frame_generator():
    detector = get_live_detector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        frame = 255 * np.ones((480, 854, 3), dtype='uint8')
        cv2.putText(frame, 'Could not open webcam', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        encoded = _encode_frame(frame)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    analysis_stride = max(1, fps // 2)
    recent_frames = deque(maxlen=max(int(fps * 2), 30))
    clip_session = None
    last_label = 'None'
    last_conf = 0.0
    last_violation_time = 0.0
    cooldown = 5.0
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            recent_frames.append(frame.copy())

            if clip_session is not None:
                clip_session["frames"].append(frame.copy())
                if len(clip_session["frames"]) >= clip_session["target_frames"]:
                    persist_live_violation(
                        detector=detector,
                        clip_frames=clip_session["frames"],
                        fps=fps,
                        snapshot=clip_session["snapshot"],
                        plate_crop=clip_session["plate_crop"],
                        plate_text=clip_session["plate_text"],
                        plate_bbox=clip_session["plate_bbox"],
                        violation_type=clip_session["violation_type"],
                        confidence=clip_session["confidence"],
                    )
                    clip_session = None
                    last_violation_time = time.time()

            display = frame.copy()
            if len(recent_frames) >= 16 and frame_index % analysis_stride == 0:
                window = list(recent_frames)[-16:]
                label, conf = detector.classifier.predict_segment(window)
                is_detected, detected_label = live_detection_decision(detector, window, label, conf)

                if is_detected:
                    last_label = detected_label
                    last_conf = conf
                    cv2.putText(display, f'{last_label} ({conf:.1%})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    last_label = 'None'
                    last_conf = 0.0
                    cv2.putText(display, 'None (0.0%)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

                if clip_session is None and is_detected and (time.time() - last_violation_time) > cooldown:
                    snapshot = frame.copy()
                    plate_crop = None
                    plate_text = ""
                    plate_bbox = None
                    result = detector.plate_det.detect(snapshot)
                    if result:
                        plate_crop, plate_text, plate_bbox = result
                        if plate_bbox is not None:
                            x1, y1, x2, y2 = plate_bbox
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if plate_text:
                                cv2.putText(display, plate_text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    clip_session = {
                        "frames": list(recent_frames),
                        "target_frames": max(int(fps * 5), len(recent_frames)),
                        "snapshot": snapshot,
                        "plate_crop": plate_crop,
                        "plate_text": plate_text if detector._is_plausible_plate_text(plate_text) else "",
                        "plate_bbox": plate_bbox,
                        "violation_type": detected_label,
                        "confidence": conf,
                    }
                    last_violation_time = time.time()
                    cv2.putText(display, 'Saving 5s clip...', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(display, 'LIVE CAMERA', (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if clip_session is not None:
                cv2.putText(display, f"REC {len(clip_session['frames'])}/{clip_session['target_frames']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            encoded = _encode_frame(display)
            if not encoded:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
            frame_index += 1
    finally:
        cap.release()


@app.route("/live_feed")
def live_feed():
    return Response(live_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/live_feed_rtsp")
def live_feed_rtsp():
    raw_url = request.args.get("url", "").strip()
    if not raw_url:
        return jsonify({"ok": False, "error": "Missing RTSP URL"}), 400
    
    # Decrypt if encrypted (user provided code abcd1234)
    rtsp_url = decode_url(raw_url)
    cap = _rtsp_cap(rtsp_url)
    return Response(_stream_live_source(cap, "RTSP camera"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/live_frame", methods=["POST"])
def api_live_frame():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or "default"
    image = decode_data_url_image(payload.get("image"))
    if image is None:
        return jsonify({"ok": False, "status": "No frame"}), 400

    detector = get_live_detector()
    face_detector = get_live_face_detector()
    session = get_live_session(session_id)

    now = time.time()
    if session["last_analysis_time"]:
        elapsed = max(0.05, now - session["last_analysis_time"])
        estimated_fps = 1.0 / elapsed
        session["fps_estimate"] = max(2, min(10, int(round(estimated_fps))))
    session["last_analysis_time"] = now

    session["recent_frames"].append(image.copy())
    if session["clip_active"]:
        session["clip_frames"].append(image.copy())

    label = "None"
    confidence = 0.0
    plate_text = ""
    face_detected = False
    saved = False

    if len(session["recent_frames"]) >= 16:
        window = list(session["recent_frames"])[-16:]
        raw_label, confidence = detector.classifier.predict_segment(window)

        is_detected, detected_label = live_detection_decision(detector, window, raw_label, confidence)

        if is_detected:
            label = detected_label
            session["last_label"] = label
            session["last_conf"] = confidence
            cv2.putText(image, f"{label} ({confidence:.1%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if session["clip_active"] is False and (now - session["last_violation_time"]) > 5.0:
                snapshot = image.copy()
                face_crop = face_detector.detect(snapshot)
                face_detected = face_crop is not None

                plate_crop = None
                plate_bbox = None
                plate_result = detector.plate_det.detect(snapshot)
                if plate_result:
                    plate_crop, plate_text, plate_bbox = plate_result
                    if not detector._is_plausible_plate_text(plate_text):
                        plate_text = ""
                else:
                    plate_text = ""

                session["clip_active"] = True
                session["clip_frames"] = list(session["recent_frames"])
                session["clip_target_frames"] = max(10, int(session["fps_estimate"] * 5))
                session["snapshot"] = snapshot
                session["face_crop"] = face_crop
                session["plate_crop"] = plate_crop
                session["plate_text"] = plate_text
                session["plate_bbox"] = plate_bbox
                session["violation_type"] = label
                session["confidence"] = session["last_conf"]
        else:
            label = "None"
            confidence = 0.0
            session["last_label"] = "None"
            session["last_conf"] = 0.0

        if session["clip_active"] and len(session["clip_frames"]) >= session["clip_target_frames"]:
            record = persist_live_violation(
                detector=detector,
                clip_frames=session["clip_frames"],
                fps=session["fps_estimate"],
                snapshot=session.get("snapshot"),
                plate_crop=session.get("plate_crop"),
                face_crop=session.get("face_crop"),
                plate_text=session.get("plate_text", ""),
                plate_bbox=session.get("plate_bbox"),
                violation_type=session.get("violation_type", "VehicleLittering"),
                confidence=session.get("confidence", confidence),
            )
            session["clip_active"] = False
            session["clip_frames"] = []
            session["last_violation_time"] = now
            saved = record is not None

    face_result = face_detector.detect(image)
    face_detected = face_detected or (face_result is not None)

    return jsonify({
        "ok": True,
        "status": "Saving clip" if session["clip_active"] else "Running",
        "label": label,
        "confidence": round(float(confidence) * 100, 1),
        "plate_text": plate_text,
        "face_detected": face_detected,
        "clip_active": session["clip_active"],
        "clip_saved": saved,
    })

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("video")
        if not file: return jsonify({"ok":False})
        fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        p = Path(app.config["UPLOAD_VIDEO_FOLDER"]) / fn
        file.save(str(p))
        processing_jobs[fn] = {"start_time": datetime.now().timestamp(), "status": "Processing", "progress": 0}
        def task():
            with app.app_context():
                try:
                    from utils.detector import SAWNDetector
                    det = SAWNDetector(
                        str(BASE_DIR / "models" / "movinet" / "movinet_best.pt"),
                        str(BASE_DIR / "models" / "yolo" / "plates_yolov8" / "weights" / "best.pt")
                    )
                    
                    def progress_cb(pct):
                        if fn in processing_jobs:
                            processing_jobs[fn]["progress"] = pct
                            
                    v = det.process_video(str(p), progress_callback=progress_cb)
                    if v:
                        rec = ViolationRecord(
                            timestamp=v.timestamp, violation_type=v.violation_type, confidence=float(v.confidence),
                            snapshot_path=str(Path(app.config["UPLOAD_FOLDER"])/f"violation_{v.id:04d}_snapshot.jpg"),
                            face_path="",
                            plate_path=str(Path(app.config["UPLOAD_FOLDER"])/f"violation_{v.id:04d}_plate.jpg") if v.plate_crop is not None else "",
                            video_path=v.video_path, plate_text=v.plate_text, status="PENDING"
                        )
                        db.session.add(rec); db.session.commit()
                except Exception as e:
                    print(f"Background processing error: {e}")
                    traceback.print_exc()
                finally:
                    processing_jobs.pop(fn, None)
        threading.Thread(target=task).start()
        return jsonify({"ok":True})
    return render_template_string(BASE_HTML.replace("{% block content %}{% endblock %}", """
    <div style="max-width: 600px; margin: 0 auto;">
        <div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: 24px; padding: 40px; text-align: center;">
            <div style="width: 64px; height: 64px; background: rgba(16, 185, 129, 0.1); border-radius: 20px; display: flex; align-items: center; justify-content: center; margin: 0 auto 24px;">
                <span style="font-size: 32px;">🎥</span>
            </div>
            <h2 style="font-size: 24px; margin-bottom: 8px;">Upload Evidence</h2>
            <p style="color: var(--text-dim); font-size: 14px; margin-bottom: 32px;">Select a video file (MP4/AVI) for AI processing and violation analysis.</p>
            
            <form id="f">
                <div style="border: 2px dashed var(--border); border-radius: 16px; padding: 40px 20px; margin-bottom: 32px; cursor: pointer; transition: all 0.3s;" 
                     onclick="document.getElementById('i').click()" 
                     onmouseover="this.style.borderColor=var(--accent); this.style.background='rgba(16,185,129,0.02)'" 
                     onmouseout="this.style.borderColor=var(--border); this.style.background='transparent'">
                    <input type="file" id="i" style="display:none" accept="video/*" onchange="updateFn(this)">
                    <div id="file-info">
                        <div style="font-size: 15px; font-weight: 600; margin-bottom: 4px;">Click to browse files</div>
                        <div style="font-size: 12px; color: var(--text-dim);">Max file size: 50MB</div>
                    </div>
                </div>
                
                <button class="btn btn-accent" id="s-btn" type="submit" style="width: 100%; justify-content: center; height: 50px; font-size: 16px;" disabled>
                    <span>Upload & Start Analysis</span>
                </button>
            </form>
            
            <div id="status-msg" style="margin-top: 24px; font-size: 14px; display: none;" class="pulse">
                <span style="color: var(--accent); font-weight: 600;">⚡ Analysis in progress...</span>
                <p style="color: var(--text-dim); font-size: 12px; margin-top: 4px;">You will be redirected shortly.</p>
            </div>
        </div>
    </div>
    
    <script>
        function updateFn(el){
            const btn = document.getElementById('s-btn');
            const info = document.getElementById('file-info');
            if(el.files && el.files[0]){
                info.innerHTML = `
                    <div style="color: var(--accent); font-weight: 600; font-size: 15px;">${el.files[0].name}</div>
                    <div style="font-size: 12px; color: var(--text-dim);">${(el.files[0].size/1024/1024).toFixed(2)} MB</div>
                `;
                btn.disabled = false;
            } else {
                info.innerHTML = `<div style="font-size: 15px; font-weight: 600;">Click to browse files</div><div style="font-size: 12px; color: var(--text-dim);">Max file size: 50MB</div>`;
                btn.disabled = true;
            }
        }
        
        document.getElementById('f').onsubmit = async e => {
            e.preventDefault();
            const btn = document.getElementById('s-btn');
            const msg = document.getElementById('status-msg');
            const fileInput = document.getElementById('i');
            
            if(!fileInput.files[0]) return;
            
            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div><span style="margin-left:12px">Uploading Evidence...</span>';
            msg.style.display = 'block';
            
            const fd = new FormData();
            fd.append('video', fileInput.files[0]);
            
            try {
                const res = await fetch('/upload', {method: 'POST', body: fd});
                if(res.ok) {
                    btn.innerHTML = '✔ Uploaded Successfully';
                    setTimeout(() => { location.href = '/'; }, 1000);
                } else {
                    throw new Error('Upload failed');
                }
            } catch(err) {
                btn.innerHTML = '❌ Upload Failed';
                btn.classList.remove('btn-accent');
                btn.classList.add('btn-red');
                setTimeout(() => { location.reload(); }, 2000);
            }
        };
    </script>
    """), title="Upload", active_page="upload")

@app.route("/api/processing_status")
def api_status():
    return jsonify({"jobs": [
        {"filename": k, "start_time": v["start_time"], "progress": v.get("progress", 0)} 
        for k, v in processing_jobs.items()
    ]})

@app.route("/api/violations")
def api_vs():
    q = ViolationRecord.query.order_by(ViolationRecord.id.desc())
    return jsonify([v.to_dict() for v in q.limit(50).all()])

@app.route("/api/stats")
def api_stats():
    total = ViolationRecord.query.count()
    acc = ViolationRecord.query.filter_by(status="ACCEPTED").count()
    rej = ViolationRecord.query.filter_by(status="REJECTED").count()
    avg = db.session.query(func.avg(ViolationRecord.confidence)).scalar() or 0
    last = ViolationRecord.query.order_by(ViolationRecord.id.desc()).first()
    last_ts = last.timestamp if last else None
    return jsonify({"total":total, "accepted":acc, "rejected":rej, "avg_confidence":round(float(avg)*100,1), "last_detection":last_ts})

@app.route("/api/accept_violation/<int:vid>", methods=["POST"])
def api_acc(vid):
    v = ViolationRecord.query.get_or_404(vid)
    v.status = "ACCEPTED"; db.session.commit(); return jsonify({"ok":True})

@app.route("/api/reject_violation/<int:vid>", methods=["POST"])
def api_rej(vid):
    v = ViolationRecord.query.get_or_404(vid)
    v.status = "REJECTED"; db.session.commit(); return jsonify({"ok":True})

@app.route("/api/violations/<int:vid>", methods=["DELETE"])
def api_delete_violation(vid):
    v = ViolationRecord.query.get_or_404(vid)
    db.session.delete(v)
    db.session.commit()
    return jsonify({"ok": True})

@app.route("/api/violations/all", methods=["DELETE"])
def api_clear_all_violations():
    ViolationRecord.query.delete()
    db.session.commit()
    return jsonify({"ok": True})
    ViolationRecord.query.delete()
    db.session.commit()
    return jsonify({"ok": True})

@app.route("/assets/<path:fn>")
def serve_asset(fn): return send_from_directory(app.config["UPLOAD_FOLDER"], fn)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

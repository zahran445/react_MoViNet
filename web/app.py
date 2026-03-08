import os
import sys
import json
import base64
import threading
from pathlib import Path
from datetime import datetime
import traceback

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

        .content { padding: 40px; max-width: 1400px; }
        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin-bottom: 40px; }
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

        .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 1000; align-items: center; justify-content: center; backdrop-filter: blur(4px); }
        .modal-overlay.open { display: flex; }
        .modal { background: var(--card-bg); border: 1px solid var(--border); border-radius: 24px; padding: 32px; width: 800px; max-width: 95vw; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
        .video-player { width: 100%; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 24px; background: #000; overflow: hidden; height: 350px; }
        .asset-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
        .asset-box { background: #000; border-radius: 12px; height: 150px; overflow: hidden; border: 1px solid var(--border); }
        .asset-box img { width: 100%; height: 100%; object-fit: cover; }
        .detail-card { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .detail-item label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; display: block; }
        .detail-item span { font-weight: 600; font-size: 15px; }
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="logo"><div class="logo-box">R</div><h2>REACT</h2></div>
        <a href="/" class="nav-item active">🏠 Dashboard</a>
        <a href="/upload" class="nav-item">📤 Upload Video</a>
    </nav>

    <div class="main">
        <header class="header">
            <h1 id="page-title">{{ title }}</h1>
            <div style="display:flex; gap:12px">
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
            <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                <h3 id="modal-title" style="font-size:20px">Evidence</h3>
                <button class="btn btn-outline" onclick="closeModal()">← Back to Log</button>
            </div>
            
            <div id="modal-v-container" class="video-player"></div>
            
            <div id="modal-assets" class="asset-grid">
                <div class="asset-box" id="m-snap"></div>
                <div class="asset-box" id="m-face"></div>
                <div class="asset-box" id="m-plate"></div>
            </div>

            <div id="modal-details" class="detail-card"></div>

            <div style="display:flex; justify-content: flex-end; gap:16px; margin-top:32px">
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
                assetWrap.style.display = 'grid';
                detailWrap.style.display = 'grid';
                document.getElementById('m-snap').innerHTML = d.snapshot_url ? `<img src="${d.snapshot_url}">` : '';
                document.getElementById('m-face').innerHTML = d.face_url ? `<img src="${d.face_url}">` : 'No Face';
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
<div class="table-container">
    <table>
        <thead><tr><th>Timestamp</th><th>Reg No.</th><th>Conf.</th><th>Status</th><th>Play</th><th>Review</th></tr></thead>
        <tbody id="rows"></tbody>
    </table>
</div>
<script>
    async function up(){
        const s = await (await fetch('/api/stats')).json();
        document.getElementById('s-row').innerHTML = `
            <div class="stat-card"><div class="stat-label">Total</div><div class="stat-value">${s.total}</div></div>
            <div class="stat-card" style="color:var(--accent)"><div class="stat-label">Accepted</div><div class="stat-value">${s.accepted}</div></div>
            <div class="stat-card" style="color:var(--red)"><div class="stat-label">Rejected</div><div class="stat-value">${s.rejected}</div></div>
            <div class="stat-card"><div class="stat-label">Avg Conf.</div><div class="stat-value">${s.avg_confidence}%</div></div>
        `;
        const vs = await (await fetch('/api/violations?limit=50')).json();
        document.getElementById('rows').innerHTML = vs.map(v => `
            <tr>
                <td>${v.timestamp}</td>
                <td style="color:var(--accent); font-weight:700">${v.plate_text}</td>
                <td>${v.confidence}%</td>
                <td><span class="status-badge status-${v.status.toLowerCase()}">${v.status}</span></td>
                <td><button class="btn btn-outline" style="padding:6px" onclick='openModal(${JSON.stringify(v)}, false)'>▶ Play</button></td>
                <td><button class="btn btn-accent" style="padding:6px 12px; font-size:12px" onclick='openModal(${JSON.stringify(v)}, true)'>${v.status.toUpperCase()=='PENDING'?'Review':'View'}</button></td>
            </tr>
        `).join('') || '<tr><td colspan="6" style="text-align:center; padding:50px">No detections</td></tr>';
    }
    up(); setInterval(up, 5000);
</script>
""")

@app.route("/")
def home(): return render_template_string(HOME_HTML, title="Dashboard")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("video")
        if not file: return jsonify({"ok":False})
        fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        p = Path(app.config["UPLOAD_VIDEO_FOLDER"]) / fn
        file.save(str(p))
        processing_jobs[fn] = "Processing"
        def task():
            with app.app_context():
                try:
                    from utils.detector import SAWNDetector
                    det = SAWNDetector(str(BASE_DIR/"models"/"movinet"/"movinet_best.pt"), str(BASE_DIR/"models"/"yolov8"/"plates_yolov8"/"weights"/"best.pt"))
                    v = det.process_video(str(p))
                    if v:
                        rec = ViolationRecord(
                            timestamp=v.timestamp, violation_type=v.violation_type, confidence=float(v.confidence),
                            snapshot_path=str(Path(app.config["UPLOAD_FOLDER"])/f"violation_{v.id:04d}_snapshot.jpg"),
                            face_path=str(Path(app.config["UPLOAD_FOLDER"])/f"violation_{v.id:04d}_face.jpg") if v.face_crop is not None else "",
                            plate_path=str(Path(app.config["UPLOAD_FOLDER"])/f"violation_{v.id:04d}_plate.jpg") if v.plate_crop is not None else "",
                            video_path=v.video_path, plate_text=v.plate_text, status="PENDING"
                        )
                        db.session.add(rec); db.session.commit()
                except: traceback.print_exc()
                finally: processing_jobs.pop(fn, None)
        threading.Thread(target=task).start()
        return jsonify({"ok":True})
    return render_template_string(BASE_HTML.replace("{% block content %}{% endblock %}", "<h2>Upload Video</h2><form id='f' style='margin-top:20px'><input type='file' id='i'><button class='btn btn-accent' type='submit' style='margin-top:20px'>Upload</button></form><script>document.getElementById('f').onsubmit=async e=>{e.preventDefault(); const fd=new FormData(); fd.append('video', document.getElementById('i').files[0]); await fetch('/upload',{method:'POST',body:fd}); location.href='/';}</script>"), title="Upload")

@app.route("/api/processing_status")
def api_status(): return jsonify({"jobs": list(processing_jobs.keys())})

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
    return jsonify({"total":total, "accepted":acc, "rejected":rej, "avg_confidence":round(float(avg)*100,1)})

@app.route("/api/accept_violation/<int:vid>", methods=["POST"])
def api_acc(vid):
    v = ViolationRecord.query.get_or_404(vid)
    v.status = "ACCEPTED"; db.session.commit(); return jsonify({"ok":True})

@app.route("/api/reject_violation/<int:vid>", methods=["POST"])
def api_rej(vid):
    v = ViolationRecord.query.get_or_404(vid)
    v.status = "REJECTED"; db.session.commit(); return jsonify({"ok":True})

@app.route("/assets/<path:fn>")
def serve_asset(fn): return send_from_directory(app.config["UPLOAD_FOLDER"], fn)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

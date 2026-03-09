"""
SAWN Inference Runner
Run the full detection pipeline on video files, a folder, or webcam.

Examples:
    # Single video
    python scripts/run_inference.py --source D:/clips/test.mp4

    # Full test folder (both classes)
    python scripts/run_inference.py --source D:/New_SawnDataset-main/NewTest_VehicleLittering

    # Webcam
    python scripts/run_inference.py --source 0

    # Log violations to running Flask dashboard
    python scripts/run_inference.py --source 0 --log-to-web
"""

import argparse
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.detector import SAWNDetector


def log_to_web(v, web_url="http://localhost:5000"):
    def p(attr):
        crop = getattr(v, attr, None)
        if crop is None:
            return ""
        fp = Path(f"outputs/violations/violation_{v.id:04d}_{attr.replace('_crop','')}.jpg")
        return str(fp) if fp.exists() else ""

    payload = {
        "timestamp":      v.timestamp,
        "violation_type": v.violation_type,
        "confidence":     v.confidence,
        "snapshot_path":  str(Path(f"outputs/violations/violation_{v.id:04d}_snapshot.jpg")),
        "face_path":      p("face_crop"),
        "plate_path":     p("plate_crop"),
        "plate_text":     v.plate_text,
    }
    try:
        r = requests.post(f"{web_url}/api/add_violation", json=payload, timeout=5)
        if r.ok:
            print(f"  [WEB] Logged violation #{v.id}")
    except requests.exceptions.ConnectionError:
        print("  [WEB] Dashboard not reachable — saved locally only.")


def run(args):
    print("=" * 60)
    print("  SAWN Inference Runner")
    print("=" * 60)

    det = SAWNDetector(
        movinet_path=args.movinet,
        yolo_path=args.yolo,
        output_dir=args.output,
    )

    source = args.source

    # ── Webcam ────────────────────────────────────────────────────────────────
    if source.isdigit():
        def on_detect(v):
            if args.log_to_web:
                log_to_web(v, args.web_url)
        
        # We modify detector.py to accept a callback for real-time logging
        violations = det.run_live(int(source), show_preview=not args.no_preview, callback=on_detect)
        return


    src_path = Path(source)

    # ── Single video ──────────────────────────────────────────────────────────
    if src_path.is_file():
        videos = [src_path]

    # ── Folder (e.g. NewTest_VehicleLittering/) ───────────────────────────────
    elif src_path.is_dir():
        videos = sorted(src_path.glob("**/*.mp4")) + sorted(src_path.glob("**/*.avi"))
        print(f"[FOLDER] {src_path}  →  {len(videos)} videos found")
    else:
        print(f"[ERROR] Source not found: {source}")
        return

    total_violations = 0
    for i, vid in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vid.name}")
        v = det.process_video(str(vid))
        if v:
            total_violations += 1
            print(f"  ✅  VIOLATION #{v.id} | {v.violation_type} | {v.confidence:.1%}")
            if args.log_to_web:
                log_to_web(v, args.web_url)
        else:
            print("  ✔  No violation.")

    print(f"\n{'='*60}")
    print(f"  {total_violations} / {len(videos)} videos flagged as violations")
    print(f"  Outputs → {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     required=True,
                        help="Video path, folder (e.g. NewTest_VehicleLittering/), or webcam index")
    parser.add_argument("--movinet",    default="models/movinet/movinet_best.keras")
    parser.add_argument("--yolo",       default="models/yolov8/plates_yolov8/weights/best.pt")
    parser.add_argument("--output",     default="outputs/violations")
    parser.add_argument("--log-to-web", action="store_true")
    parser.add_argument("--web-url",    default="http://localhost:5000")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    run(args)

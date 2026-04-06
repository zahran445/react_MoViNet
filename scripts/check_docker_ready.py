import os
import sys
import torch
import tensorflow as tf
from pathlib import Path

def check_requirements():
    print("--- SAWN Docker Verification ---")
    
    # 1. Check Python version
    print(f"Python version: {sys.version}")
    
    # 2. Check DL Frameworks
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch GPU available: {torch.cuda.is_available()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # 3. Check Weights
    base_dir = Path(__file__).resolve().parent.parent
    weights = [
        base_dir / "models/movinet/movinet_best.pt",
        base_dir / "models/yolo/plates_yolov8/weights/best.pt"
    ]
    
    all_present = True
    for w in weights:
        if w.exists():
            print(f"  [OK] Found: {w.relative_to(base_dir)}")
        else:
            print(f"  [MISSING] {w.relative_to(base_dir)}")
            all_present = False
            
    if not all_present:
        print("\nERROR: Model weights are missing from the container!")
        sys.exit(1)
        
    print("\n--- Dry Run: Loading Models ---")
    try:
        # Check if we can import the detector
        sys.path.append(str(base_dir))
        from utils.detector import SAWNDetector
        
        print("Initializing SAWNDetector (CPU fallback mode)...")
        # Initialize detector - this will load the weights
        # We use CPU to ensure it works even without GPU during verification
        detector = SAWNDetector(
            str(weights[0]),
            str(weights[1])
        )
        print("  [SUCCESS] SAWNDetector initialized and weights loaded.")
    except Exception as e:
        print(f"  [FAILED] Could not load models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\nVERIFICATION COMPLETE: Docker image is ready for production.")

if __name__ == "__main__":
    check_requirements()

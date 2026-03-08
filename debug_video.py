import cv2
cap = cv2.VideoCapture(r"d:\sawn_project\outputs\uploads\20260308_141646_violation_vid.mp4")
if not cap.isOpened():
    print("Could not open video")
else:
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
cap.release()

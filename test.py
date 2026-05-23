import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/isac_run_1-7/weights/best.pt')

# 1. Update this to match your EXACT file name
video_path = '2026-05-23 19-38-38.mp4' 
cap = cv2.VideoCapture(video_path)

# 2. THE FIX: Force Python to scream if the video is missing
if not cap.isOpened():
    print(f"\n[ERROR] OpenCV could not find or open: {video_path}")
    print("Check that the file is in the same folder as test.py!\n")
    exit()

# 3. Suppress the terminal spam so you can actually see what's happening
print("Video loaded! Launching preview window...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 

    # Added verbose=False to stop the terminal from printing every single frame
    results = model(frame, conf=0.4, verbose=False) 

    annotated_frame = results[0].plot()
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow('ISAC Vision Test', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
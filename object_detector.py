from ultralytics import YOLO
import cv2

model_general = YOLO("yolo11n.pt")       # general objects (people, cups, etc.)
model_custom = YOLO("adapter_detector.pt")  # custom device class

cap = cv2.VideoCapture(0)  # 0 = webcam
while True:
    ret, frame = cap.read()
    # Draw general detections first, then overlay custom detections on top
    annotated = model_general(frame)[0].plot()
    annotated = model_custom(annotated)[0].plot()
    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
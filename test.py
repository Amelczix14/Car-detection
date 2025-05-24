import cv2
import os
from ultralytics import YOLO

# === KONFIGURACJA ===
INPUT_PATH = "app/videos/madzia.mp4"
MODEL_PATH = "models/color_model/color_model.pt"
CONFIDENCE_THRESHOLD = 0.80  # Detekcje tylko powyżej 85%
# =====================

ext = os.path.splitext(INPUT_PATH)[1].lower()
is_image = ext in [".jpg", ".jpeg", ".png"]

model = YOLO(MODEL_PATH)

def draw_detections(frame, results):
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_text = f'{label} {conf:.2f}'

                # Ramka wokół obiektu
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Tło pod napisem
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)

                # Tekst
                cv2.putText(frame, label_text, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

if is_image:
    img = cv2.imread(INPUT_PATH)
    if img is not None:
        results = model(img)
        annotated = draw_detections(img.copy(), results)
        cv2.imshow("Detekcja koloru samochodu", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(INPUT_PATH)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            results = model(frame, verbose=False)
            annotated = draw_detections(frame.copy(), results)
            cv2.imshow("Detekcja koloru samochodu", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

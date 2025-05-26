import cv2
from detection.plate_detection import PlateDetection
from detection.car_detection import CarDetection
from detection.color_detection import ColorDetection

# === Ścieżka do pliku wideo ===
VIDEO_PATH = "app/videos/amcia.mp4"
LICENSE_MODEL_PATH = "models/my_model/my_model.pt"
CAR_MODEL_PATH = "models/car_model/car_model.pt"

plate_detector = PlateDetection(LICENSE_MODEL_PATH)
car_detector = CarDetection(CAR_MODEL_PATH)
color_detector = ColorDetection()

# === Otwórz wideo ===
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Nie udało się otworzyć pliku wideo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    plate_results = plate_detector.model(frame, verbose=False)
    plate_detected = False

    for det in plate_results[0].boxes:
        if det.conf > plate_detector.min_thresh:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            label = plate_detector.model.names[int(det.cls)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            plate_detected = True

    # detekcja samochodu tylko jeśli wykryto tablicę
    if plate_detected:
        car_results = car_detector.model(frame, verbose=False)
        for det in car_results[0].boxes:
            if det.conf > car_detector.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                label = car_detector.model.names[int(det.cls)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                car_crop = frame[y1:y2, x1:x2]

                car_color = color_detector.classify_color(car_crop)
                print("Kolor samochodu:", car_color)

                if car_color:
                    cv2.putText(frame, f"Kolor: {car_color}", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                break

    cv2.imshow("Detekcja tablicy i samochodu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
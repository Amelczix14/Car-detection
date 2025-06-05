import cv2
from detection.plate_detection import PlateDetection
from detection.car_detection import CarDetection
from detection.brand_detection import BrandDetection

# === Ścieżki ===
VIDEO_PATH = "app/videos/amcia4.mp4"
LICENSE_MODEL_PATH = "models/my_model/my_model.pt"
CAR_MODEL_PATH = "models/car_model/car_model.pt"
BRAND_MODEL_PATH = "models/brand_model/best.pt"

# === Inicjalizacja modeli ===
plate_detector = PlateDetection(LICENSE_MODEL_PATH)
car_detector = CarDetection(CAR_MODEL_PATH)
brand_detector = BrandDetection(BRAND_MODEL_PATH)

# === Otwórz wideo ===
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Nie udało się otworzyć pliku wideo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # === Detekcja tablicy ===
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

    # === Detekcja samochodu i marki tylko jeśli wykryto tablicę ===
    if plate_detected:

        # === Samochód ===
        car_results = car_detector.model(frame, verbose=False)
        for det in car_results[0].boxes:
            if det.conf > car_detector.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                car_label = car_detector.model.names[int(det.cls)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{car_label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                break  # jeden samochód

        # === Marka samochodu ===
        brand_results = brand_detector.model(frame, verbose=False)
        for det in brand_results[0].boxes:
            if det.conf > brand_detector.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                brand_label = brand_detector.model.names[int(det.cls)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{brand_label}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print("Wykryta marka:", brand_label)
                break  # jedna marka

    cv2.imshow("Detekcja marki i samochodu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

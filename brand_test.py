import cv2
from detection.plate_detection import PlateDetection
from detection.car_detection import CarDetection
from detection.brand_detection import BrandDetection

VIDEO_PATH = "app/videos/madzia.mp4"
LICENSE_MODEL_PATH = "models/my_model/my_model.pt"
CAR_MODEL_PATH = "models/car_model/car_model.pt"
BRAND_MODEL_PATH = "models/brand_model/brand_model.pt"

plate_detector = PlateDetection(LICENSE_MODEL_PATH)
car_detector = CarDetection(CAR_MODEL_PATH)
brand_detector = BrandDetection(BRAND_MODEL_PATH)

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

                # === Detekcja marki na wyciętym samochodzie ===
                brand_label, brand_crop, brand_coords = brand_detector.detect_brand(car_crop)
                if brand_label:
                    print("Marka:", brand_label)
                    cv2.putText(frame, f"Marka: {brand_label}", (x1, y2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                break  # Zakładamy tylko jeden samochód na raz

    cv2.imshow("Detekcja marki samochodu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

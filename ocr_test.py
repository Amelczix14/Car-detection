import cv2
import numpy as np
from detection.plate_detection import PlateDetection
from detection.ocr_detection import read_licence_plate

VIDEO_PATH = "app/videos/amcia.mp4"
LICENSE_MODEL_PATH = "models/my_model/my_model.pt"

plate_detector = PlateDetection(LICENSE_MODEL_PATH)

def preprocess_for_ocr(cropped_plate):
    """ identyczne przetwarzanie jak w OCR — pokazujemy co dokładnie widzi Tesseract """
    scale_factor = 2
    enlarged = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)

    return dilated


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Nie udało się otworzyć pliku wideo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OBRÓCENIE O 90° W LEWO
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # --- DETEKCJA TABLICY ---
    plate_results = plate_detector.model(frame, verbose=False)

    ocr_preview = None

    for det in plate_results[0].boxes:
        if det.conf > plate_detector.min_thresh:
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] > 10 and crop.shape[1] > 10:
                ocr_preview = preprocess_for_ocr(crop)

                text, conf = read_licence_plate(crop)
                cv2.putText(frame, f"OCR: {text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

    cv2.imshow("Oryginalna klatka", frame)

    if ocr_preview is not None:
        cv2.imshow("OCR input (po przetwarzaniu)", ocr_preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

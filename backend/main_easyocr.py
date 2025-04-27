import cv2
import easyocr
import numpy as np

allowed_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

def ensure_easyocr_models():
    try:
        easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR models are ready.")
    except Exception as e:
        print(f"⚠️ Error initializing EasyOCR models: {e}")

# Automatyczne pobranie modeli przy starcie, jeśli ich brak
ensure_easyocr_models()

# Właściwy OCR reader (możesz zmienić gpu=True jeśli masz wsparcie GPU)
reader = easyocr.Reader(['en'], gpu=False)

def read_licence_plate(cropped_plate):
    """
    Funkcja do przetwarzania wyciętej tablicy rejestracyjnej i odczytu jej tekstu za pomocą OCR.
    """
    scale_factor = 2
    enlarged_plate = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(enlarged_plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)

    result = reader.readtext(dilated, allowlist=allowed_characters)
    detected_text = "".join([detection[1] + " " for detection in result])

    print(f"odczytana tablica: {detected_text.strip()}")

    return detected_text.strip()

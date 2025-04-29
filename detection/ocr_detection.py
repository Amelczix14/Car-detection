# import cv2
# import easyocr
# import numpy as np
#
#
# # TODO: ocr nie czyta poprawnie wszystkich znaków
# def read_licence_plate(cropped_plate):
#     """
#     Funkcja do przetwarzania wyciętej tablicy rejestracyjnej i odczytu jej tekstu za pomocą OCR.
#     """
#     # powiększenie wyciętej ramki
#     scale_factor = 2
#     enlarged_plate = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
#
#     allowed_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#     reader = easyocr.Reader(['en'], gpu=True)
#
#     gray = cv2.cvtColor(enlarged_plate, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#
#     _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#
#     denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
#
#     kernel = np.ones((3, 3), np.uint8)
#     dilated = cv2.dilate(denoised, kernel, iterations=1)
#
#     result = reader.readtext(dilated, allowlist=allowed_characters)
#
#     detected_text = ""
#     for detection in result:
#         detected_text += detection[1] + " "
#
#     print(f"odczytana tablica: {detected_text.strip()}")
#
#     # TODO: usunąć w wersji webowej
#     cv2.imshow("processed frame", dilated)
#     cv2.waitKey(0)
#
#     return detected_text.strip()



import cv2
import pytesseract
import numpy as np

# TODO: OCR nie czyta poprawnie wszystkich znaków
def read_licence_plate(cropped_plate):
    """
    Funkcja do przetwarzania wyciętej tablicy rejestracyjnej i odczytu jej tekstu za pomocą Tesseract OCR.
    """
    # Powiększenie wyciętej ramki
    scale_factor = 2
    enlarged_plate = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    allowed_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Konwersja obrazu na szarość
    gray = cv2.cvtColor(enlarged_plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Thresholding
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)

    # Dylatacja
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)

    # Konfiguracja Tesseract: wybór języka i zestawu dozwolonych znaków
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=' + allowed_characters

    # Rozpoznanie tekstu z obrazu
    detected_text = pytesseract.image_to_string(dilated, config=custom_config)

    # Wyświetlenie wyników
    print(f"odczytana tablica: {detected_text.strip()}")

    return detected_text.strip()

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Wczytanie modelu YOLO
model = YOLO("my_model/my_model.pt")

# Ścieżka do obrazu
image_path = "images/Cars97.png"
image = cv2.imread(image_path)

# Tworzenie obiektu OCR EasyOCR
reader = easyocr.Reader(['en'])  # Możesz dodać inne języki, np. ['pl'] dla polskiego

# Przetwarzanie obrazu za pomocą YOLO
results = model(image)

# Dla każdego wykrytego obiektu (tablica rejestracyjna)
for r in results:
    for box in r.boxes:
        # Pobieranie współrzędnych ramki
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]  # Wycięcie fragmentu obrazu (tablica)

        # Przeskalowanie obrazu
        scaled_cropped = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # Przekształcenie na skalę szarości
        gray = cv2.cvtColor(scaled_cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Zastosowanie progu binarnego (czarne obiekty na czarno, reszta biała)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # # Dodatkowe wyostrzanie obrazu (opcjonalne)
        # sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # sharpened = cv2.filter2D(binary, -1, sharpen_kernel)

        # Preprocessing obrazu - usuwanie szumów (opcjonalne)
        denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)

        # Odczyt tekstu za pomocą EasyOCR
        result = reader.readtext(denoised)

        # Przetwarzanie wyników z EasyOCR i wyświetlanie odczytanych tekstów
        detected_text = ""
        for detection in result:
            detected_text += detection[1] + " "  # Łączenie wyników w jeden ciąg tekstowy

        print(f"Odczytana tablica: {detected_text.strip()}")

        # Wyświetlanie wyciętego obrazu (tablica rejestracyjna)
        cv2.imshow("Tablica", cropped)
        cv2.imshow('Processed', denoised)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


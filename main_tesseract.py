import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Wczytanie modelu YOLO
model = YOLO("my_model/my_model.pt")

# Ścieżka do obrazu
image_path = "images/Cars1.png"
image = cv2.imread(image_path)

# Przetwarzanie obrazu za pomocą YOLO
results = model(image)

# Dla każdego wykrytego obiektu (tablica rejestracyjna)
for r in results:
    for box in r.boxes:
        # Pobieranie współrzędnych ramki
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]  # Wycięcie fragmentu obrazu (tablica)

        scaled_cropped = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # Przekształcenie na skalę szarości
        gray = cv2.cvtColor(scaled_cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Zastosowanie progu binarnego (czarne obiekty na czarno, reszta biała)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Dodatkowe wyostrzanie obrazu (opcjonalne)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(binary, -1, sharpen_kernel)

        # Preprocessing obrazu - usuwanie szumów (opcjonalne)
        denoised = cv2.fastNlMeansDenoising(sharpened, None, 30, 7, 21)

        # Dostosowanie parametrów pytesseract - tylko znaki i cyfry
        config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Odczyt tekstu z obrazu
        text = pytesseract.image_to_string(denoised, config=config)
        # Wyświetlenie odczytanej tablicy
        print(f"Odczytana tablica: {text.strip()}")

        # Wyświetlanie wyciętego obrazu (tablica rejestracyjna)
        cv2.imshow("Tablica", cropped)
        cv2.imshow('Processed', denoised)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




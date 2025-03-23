import cv2
import pytesseract
from ultralytics import YOLO

model = YOLO("my_model/my_model.pt")  

image_path = "images/Cars1.png"  
image = cv2.imread(image_path)

results = model(image)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        cropped = image[y1:y2, x1:x2]  

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
        
        config = r'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        
        cv2.imshow("Tablica", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Odczytana tablica: {text.strip()}")


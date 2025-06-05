from ultralytics import YOLO


class BrandDetection:
    def __init__(self, model_path, min_thresh=0.5):
        self.model = YOLO(model_path)
        self.min_thresh = min_thresh

    def detect_brand(self, frame):
        """
        Wykrywa markę samochodu na klatce.
        Zwraca: nazwa marki, fragment obrazu z marką, współrzędne.
        """
        results = self.model(frame, verbose=False)
        best_brand = None
        best_crop = None
        best_coords = None

        for det in results[0].boxes:
            conf = det.conf
            if conf > self.min_thresh:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                label = self.model.names[int(det.cls)]

                best_crop = frame[y1:y2, x1:x2]
                best_brand = label
                best_coords = (x1, y1, x2, y2)
                break  # Tylko pierwsze wykrycie

        return best_brand, best_crop, best_coords

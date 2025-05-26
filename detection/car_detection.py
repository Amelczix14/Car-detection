from ultralytics import YOLO


class CarDetection:
    def __init__(self, model_path, min_thresh=0.5):
        self.model = YOLO(model_path)
        self.min_thresh = min_thresh

    def detect_car(self, frame):
        """
        Wykrywa samochód na danej klatce (np. wcześniej wyciętej z tablicą).
        Zwraca fragment obrazu z samochodem oraz współrzędne (x1, y1, x2, y2).
        """

        results = self.model(frame, verbose=False)
        best_car = None
        best_coords = None

        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = det.conf

            if conf > self.min_thresh:
                best_car = frame[y1:y2, x1:x2]
                best_coords = (x1, y1, x2, y2)

                break

        return best_car, best_coords
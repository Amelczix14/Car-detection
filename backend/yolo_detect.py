import cv2
from ultralytics import YOLO
import numpy as np

class YOLODetection:
    def __init__(self, model_path="my_model/my_model.pt", min_thresh=0.85):
        self.model = YOLO(model_path)
        self.min_thresh = min_thresh

    def detect_plate(self, source):
        frame = cv2.imread(source)
        if frame is None:
            raise ValueError(f"Nie można załadować obrazu: {source}")
        return self._process_frame(frame)

    def _process_frame(self, frame):
        results = self.model(frame, verbose=False)
        best_plate = None
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            if det.conf > self.min_thresh:
                cropped_plate = frame[y1:y2, x1:x2]
                best_plate = cropped_plate
                cv2.imwrite("capture.png", cropped_plate)
        return best_plate

import cv2
from ultralytics import YOLO
import numpy as np


class YOLODetection:
    def __init__(self, model_path, min_thresh=0.85):
        self.model = YOLO(model_path)
        self.min_thresh = min_thresh

    # def detect_plate(self, source):
    #     """
    #     Funkcja do detekcji tablicy rejestracyjnej z podanego źródła (obraz, wideo, kamera).
    #     Zwraca najlepszą wykrytą ramkę z tablicą rejestracyjną.
    #     """
    #
    #     # dla obrazu
    #     is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.png'))
    #
    #     if is_image:
    #         frame = cv2.imread(source)
    #         return self._process_frame(frame)
    #     else:
    #         # dla wideo lub kamery
    #         cap = None if is_image else cv2.VideoCapture(source)
    #
    #         while cap and cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #
    #             # TODO: usunąć jeśli obraz jest obrócony
    #             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #
    #             cv2.imshow('yolo detection', frame)
    #
    #             key = cv2.waitKey(1 if cap else 0) & 0xFF
    #             if key == 27:
    #                 break
    #
    #             plate = self._process_frame(frame)
    #             if plate is not None:
    #                 return plate
    #
    #         cap.release() if cap else None
    #         cv2.destroyAllWindows()
    #
    #     return None



    def detect_plate(self, source):
        """
        Funkcja do detekcji tablicy rejestracyjnej z podanego źródła (obraz, wideo, kamera).
        Zwraca pierwszą wykrytą ramkę z tablicą rejestracyjną, ale nie przerywa wyświetlania wideo.
        """

        # dla obrazu
        is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.png'))

        if is_image:
            frame = cv2.imread(source)
            return self._process_frame(frame)
        else:
            # dla wideo lub kamery
            cap = None if is_image else cv2.VideoCapture(source)

            best_plate = None

            while cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # TODO: usunąć jeśli obraz jest obrócony
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Przetwarzanie klatki
                results = self.model(frame, verbose=False)

                for det in results[0].boxes:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    conf = det.conf
                    label = self.model.names[int(det.cls)]

                    if conf > self.min_thresh:
                        # Rysowanie ramki wokół wykrytej tablicy
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, label, (x1, y1 - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                        # Jeśli jeszcze nie przypisano 'best_plate', przypisujemy pierwszą wykrytą tablicę
                        if best_plate is None:
                            best_plate = frame[y1:y2, x1:x2]

                # Wyświetlanie wideo w czasie rzeczywistym
                cv2.imshow("Detection", frame)

                # Jeśli użytkownik naciśnie 'q', zatrzymujemy detekcję
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release() if cap else None
            cv2.destroyAllWindows()

            return best_plate




    def _process_frame(self, frame):
        """
        Pomocnicza funkcja do detekcji tablicy na pojedynczym obrazie/wideo.
        Zwraca najlepszą wykrytą ramkę z tablicą rejestracyjną.
        """
        results = self.model(frame, verbose=False)

        # przechodzimy przez wszystkie detekcje
        best_plate = None
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            if det.conf > self.min_thresh:
                # wycięcie tablicy
                cropped_plate = frame[y1:y2, x1:x2]
                best_plate = cropped_plate

        return best_plate
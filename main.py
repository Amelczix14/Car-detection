from detection.yolo_detection import YOLODetection
from detection.ocr_detection import read_licence_plate
import cv2


def main(source):
    detector = YOLODetection(model_path='my_model/my_model.pt')
    best_plate = detector.detect_plate(source)

    if best_plate is not None:
        # TODO: odkomentować żeby zobaczyć wynik yolo
        # cv2.imshow("cropped plate", best_plate)
        # cv2.waitKey(0)

        read_licence_plate(best_plate)
    else:
        print("Nie wykryto tablicy rejestracyjnej.")


if __name__ == "__main__":
    # TODO:
    #  można zmeniać źródło na ścieżli do plików, nr kamery (np 0)
    #  lub adres ip telefonu (np. source = "http://192.168.1.100:4747/video")
    source = "IMG_6581.mp4"
    main(source)

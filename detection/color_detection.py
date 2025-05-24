import cv2
import numpy as np


class ColorDetection:
    def __init__(self):
        pass

    def classify_color(self, image):
        if image is None or image.size == 0:
            return None

        img = cv2.resize(image, (100, 100))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(img_hsv)
        mask = (s < 60) & (v > 40) & (v < 220)

        if np.count_nonzero(mask) == 0:
            return 'unknown'

        median_v = np.median(v[mask])

        if median_v > 180:
            return 'white'
        elif median_v < 70:
            return 'black'
        else:
            return 'grey'

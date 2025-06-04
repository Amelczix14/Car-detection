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

        # czerwony najwyższy priorytet
        red_mask1 = cv2.inRange(img_hsv, (0, 70, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(img_hsv, (160, 70, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        if cv2.countNonZero(red_mask) > (0.05 * red_mask.size):  # minimum 5% pikseli
            return 'czerwony'

        # neutralne kolory: biały, szary, czarny
        neutral_mask = (s < 60) & (v > 40) & (v < 230)
        neutral_pixels = v[neutral_mask]

        if len(neutral_pixels) > 0:
            median_v = np.median(neutral_pixels)
            if median_v > 180:
                return 'biały'
            elif median_v < 70:
                return 'czarny'
            else:
                return 'szary'

        return 'unknown'

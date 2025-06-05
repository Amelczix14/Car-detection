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

        # czerwony – najwyższy priorytet
        red_mask1 = cv2.inRange(img_hsv, (0, 70, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(img_hsv, (160, 70, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        if cv2.countNonZero(red_mask) > (0.05 * red_mask.size):
            return 'czerwony'

        # neutralne – czarny, biały, szary
        neutral_mask = (s < 60)
        s_neutral = s[neutral_mask]
        v_neutral = v[neutral_mask]

        if len(v_neutral) > 0:
            mean_v = np.mean(v_neutral)
            std_v = np.std(v_neutral)
            median_v = np.median(v_neutral)

            # Czarny: niska jasność i bardzo niskie nasycenie
            if mean_v < 90 and np.mean(s_neutral) < 40:
                return 'czarny'

            # Biały: bardzo wysoka jasność
            if mean_v > 200:
                return 'biały'

            # Szary: średnia jasność + niskie nasycenie
            if 110 < mean_v < 200:
                return 'szary'

        return 'unknown'


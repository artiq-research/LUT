import numpy as np
from cv2 import cv2


def exposure(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


if __name__ == "__main__":
    pass
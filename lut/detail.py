import math
import numpy as np
from cv2 import cv2
from PIL import Image
from PIL import ImageEnhance
from lut.transforms import RGBTransform


def sharpness(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(factor)
    img = np.asarray(img)
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res

if __name__ == "__main__":
    image = cv2.imread('img/50343084823_48a0274a6a_c.jpg')
    cv2.imshow('sharpness', sharpness(image, 0.0))
    cv2.waitKey()

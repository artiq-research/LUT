import math
import numpy as np
from cv2 import cv2
from PIL import Image
from PIL import ImageEnhance
from lut.transforms import RGBTransform


def clamp(x, min, max):
    if x < min:
        return min
    if x > max:
        return max
    return x

def get_temp(kelvin):
    e = 1e-7
    temperature = kelvin / 100

    red, green, blue = 0, 0, 0

    if temperature <= 66:
        red = 255
        green = temperature
        green = 99.4708025861 * math.log(green+e) - 161.1195681661
        if temperature <= 19:
            blue = 0
        else:
            blue = temperature - 10
            blue = 138.5177312231 * math.log(blue+e) - 305.0447927307
    else:
        red = temperature - 60
        red = 329.698727446 * math.pow(red, -0.1332047592)
        green = temperature - 60
        green = 288.1221695283 * math.pow(green, 0.0755148492)
        blue = 255
    
    res_tuple = (
        clamp(red, 0, 255),
        clamp(green, 0, 255),
        clamp(blue, 0, 255)
    )

    return res_tuple

def temperature(img, kelvin):
    rgb = get_temp(kelvin=kelvin)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    mixed = RGBTransform().mix_with(base_color=rgb, factor=.30).applied_to(img)
    mixed = np.asarray(mixed)
    res = cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR)
    return res

def get_tint(kelvin):
    e = 1e-7
    temperature = kelvin / 100

    red, green, blue = 0, 0, 0

    if temperature <= 66:
        red = 255
        green = temperature
        green = 99.4708025861 * math.log(green+e) - 161.1195681661
        if temperature <= 19:
            blue = 0
        else:
            blue = temperature - 10
            blue = 138.5177312231 * math.log(blue+e) - 305.0447927307
    else:
        red = temperature - 60
        red = 329.698727446 * math.pow(red, -0.1332047592)
        green = temperature - 60
        green = 288.1221695283 * math.pow(green, 0.0755148492)
        blue = 255
    
    res_tuple = (
        clamp(green, 0, 255),
        clamp(red, 0, 255),
        clamp(blue, 0, 255)
    )

    return res_tuple

def tint(img, kelvin):
    rgb = get_tint(kelvin=kelvin)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    mixed = RGBTransform().mix_with(base_color=rgb, factor=.30).applied_to(img)
    mixed = np.asarray(mixed)
    res = cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR)
    return res


def color_balance(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageEnhance.Color(img)
    img = img.enhance(factor)
    img = np.asarray(img)
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

if __name__ == "__main__":
    image = cv2.imread('img/50343084823_48a0274a6a_c.jpg')
    # for i in range(0, 10):
    #     cv2.imshow('asdf', tint(image, 1000*i))
    #     cv2.waitKey()
    # cv2.imshow('color balance', color_balance(image, 2.0))
    # cv2.waitKey()
    cv2.imshow('white balance', white_balance(image))
    cv2.waitKey()

import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import cv2
import onnxruntime as rt
import time

class Preprocessor():

    SKY_RATIO = 330/720
    CAR_RATIO = 163/720

    def run(self, img):
        imgh, imgw = len(img), len(img[0])

        sky_crop = int(imgh * Preprocessor.SKY_RATIO)
        car_crop = int(imgh * Preprocessor.CAR_RATIO)

        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

        brightness = np.sum(img, axis=-1)
        brightness = np.repeat(brightness[..., np.newaxis], 3, axis=-1)
        img = np.where(brightness < 100, 60, img)

        #img = img[sky_crop:imgh-car_crop]
        cv2.rectangle(img, (0,0), (imgw, sky_crop), (0, 0, 0), -1)
        cv2.rectangle(img, (0,imgh - car_crop), (imgw, imgh), (0, 0, 0), -1)

        return img
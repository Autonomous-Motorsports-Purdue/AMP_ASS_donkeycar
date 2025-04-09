import numpy as np
import cv2
class Viewer():
    def __init__(self):
        pass
    def run(self, image):
        if image is not None:
            cv2.imshow("image", np.array(image))
            cv2.waitKey(1)
        else:
            print("image is None")
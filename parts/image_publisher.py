import cv2

class Image_Publisher():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
    def run(self):
        print("running")
        ret, self.frame = self.cap.read()
        if self.frame is not None:
            # cv2.imshow("Image", self.frame)
            # cv2.waitKey(1)
            return self.frame
import cv2
class Camera():
    def __init__(self) -> None:
        self.img_num = 200
    def run(self):
        if 1 <= self.img_num <= 1543:
            self.img_num += 1
        if self.img_num > 1543:
            self.img_num = 1
        file_name = file_name = "./images/img (" + str(self.img_num) + ").jpg"
        img = cv2.imread(file_name)
        return img
import cv2

class Test_Publisher():
    def __init__(self):
        self.video_path = ' path '
        self.cap = cv2.VideoCapture(self.video_path)
    def run(self,vid_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,vid_frame)

        ret, frame = self.cap.read()

        if ret:
            cv2.imshow("Frame",frame)
        else:
            print("failed to retrieve frame")

        self.cap.release()

        return frame
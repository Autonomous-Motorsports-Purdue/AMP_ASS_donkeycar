import cv2
import numpy as np
class Frame_Publisher():
    def grab_middle_section(self, image, width_ratio, height_ratio):
        """
        Grabs the middle section of an image based on given ratios.

        Args:
            image (numpy.ndarray): The input image.
            width_ratio (float): Ratio for the width of the middle section (0 < ratio < 1).
            height_ratio (float): Ratio for the height of the middle section (0 < ratio < 1).

        Returns:
            numpy.ndarray: The cropped middle section of the image.
        """

        height, width = image.shape[:2]

        start_x = int((1 - width_ratio) / 2 * width)
        end_x = int(start_x + width_ratio * width)
        start_y = int((1 - height_ratio) / 2 * height)
        end_y = int(start_y + height_ratio * height)

        return image[start_y:end_y, start_x:end_x]
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
    def run(self):
        """
        Read in and return the current left and right frames from the ZED.
        """
        ret, self.frame = self.cap.read()   
        # cv2.imshow("test", self.frame)
        if self.frame is not None:
            height, width = self.frame.shape[:2]

            # Split the self.frame in half
            # Assuming we want to split vertically
            left_half = self.frame[:, :width // 2]
            right_half = self.frame[:, width // 2:]
            # left_half = self.grab_middle_section(left_half, 0.9, 0.9)
            # right_half = self.grab_middle_section(right_half, 0.9, 0.9)
            print("left_half: ", left_half.shape)
            return np.array(left_half), np.array(right_half)
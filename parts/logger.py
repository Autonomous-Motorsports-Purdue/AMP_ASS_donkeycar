import datetime
import cv2
import os
import numpy as np
class Logger():
    def __init__(self):
        self.image_directory = "data/images/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        self.point_directory = "data/points/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        # self.depth_directory = "data/depth/"  + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        if not os.path.exists(self.point_directory):
            os.makedirs(self.point_directory)
        # if not os.path.exists(self.depth_directory):
            # os.makedirs(self.depth_directory)

    def run(self, image, depth, points):
        if image is not None and points.size != 0:
            image_file = self.image_directory + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + ".jpg"
            point_file = self.point_directory + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + ".t"

            np.savetxt(point_file, points)
            # depth_file = self.depth_directory + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + ".jpg"
            print(f"Attempting to save file: {image_file}")
            # Save the image and check if it was successful
            success = cv2.imwrite(image_file, image)
            if success:
                print(f"File saved successfully: {image_file}")
            else:
                print("Failed to save the file. Check if the image is valid.")

            # Save the depth image and check if it was successful
            # success = cv2.imwrite(depth_file, depth)
            # if success:
                # print(f"File saved successfully: {depth_file}")
            # else:
                # print("Failed to save the file. Check if the depth image is valid.")

            # cv2.imshow("image", image)
            # cv2.imshow("depth", depth)
            # cv2.waitKey(1)

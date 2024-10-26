import datetime
import cv2
import os
import numpy as np
import csv
class Logger():
    def __init__(self):
        start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.image_directory = "data/images/" + start_time
        self.segmented_directory = "data/segmented_images/" +start_time
        # self.depth_directory = "data/depth/"  + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        if not os.path.exists(self.segmented_directory):
            os.makedirs(self.segmented_directory)
        self.info_csv = "data/information/" + start_time + ".csv"
        # if not os.path.exists(self.depth_directory):
            # os.makedirs(self.depth_directory)
        # writing to csv file
        with open(self.info_csv, 'w') as self.csvfile:
            # creating a csv writer object
            self.csvwriter = csv.writer(self.csvfile)

            fields = ['timestamp', 'image', 'segmented', 'centroid', 'steering', 'throttle']
            # writing the fields
            self.csvwriter.writerow(fields)
        
    def run(self, image, segmentedImage, centroid, steering, throttle):
        if image is not None and segmentedImage is not None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
            image_file = self.image_directory +"/"+ timestamp + ".jpg"
            segmented_file = self.point_directory + "/" + timestamp + ".jpg"            

            # Save the images if written is successful
            success_image = cv2.imwrite(image_file, image)
            success_segmented = cv2.imwrite(segmented_file, segmentedImage)
            if success_image and success_segmented:
                rows = [timestamp, image_file, segmented_file, centroid, steering, throttle]    
                self.csvwriter.writerow(rows)
            
            
        


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
        self.information_directory= "data/information/"
        # self.depth_directory = "data/depth/"  + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        if not os.path.exists(self.segmented_directory):
            os.makedirs(self.segmented_directory)
        if not os.path.exists(self.information_directory):
            os.makedirs(self.information_directory)
        self.info_csv = "data/information/" + start_time + ".csv"
        # Writing to csv file
        self.csvfile = open(self.info_csv, 'w') 
        # Creating a csv writer object
        self.csvwriter = csv.writer(self.csvfile)

        fields = ['timestamp', 'image', 'segmented', 'centroid', 'steering', 'throttle']
        # Writing the fields
        self.csvwriter.writerow(fields)
        
    def run(self, image, segmentedImage, centroid, steering, throttle):
        """
        Logs the current image, segmented Image, centroid, steering, and throttle values.
        Saves the images in their respective directory and logs the image paths and other data into a CSV.
        """
        if image is not None and segmentedImage is not None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
            image_file = self.image_directory +"/"+ timestamp + ".jpg"
            segmented_file = self.segmented_directory + "/" + timestamp + ".jpg"            

            # Save the images if written is successful
            success_image = cv2.imwrite(image_file, image.copy())
            success_segmented = cv2.imwrite(segmented_file, segmentedImage.copy())
            if success_image and success_segmented:
                rows = [timestamp, image_file, segmented_file, centroid, steering, throttle]    
                self.csvwriter.writerow(rows)
            
            
        


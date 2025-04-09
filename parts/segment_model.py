import numpy as np
import cv2
import onnxruntime as rt
from simple_pid import PID
import math
from constants import *

class Segment_Model():
        def __init__(self):
            self.sess = rt.InferenceSession( "q_model.onnx", providers=[('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"})])
            self.pid = PID(K_P, 0.00, 0, setpoint=0)
            self.prev_steer = 0
            print('model loaded')

        def run(self,img):
            """
            Run segmentation model on the inputted image and calculate steering and throttle values based on the centroid in the detected track.
            Uses the PID to follow the centroid of the track

            Args:
                img (numpy.ndarray): Inputted image from frame publisher
            Returns: 
                img_rs(numpy.ndarray): Segmented image from the model after cropping and countour detection
                (contour_center_x,contour_center_y) (int, int): Tuple containing the X and Y pixel values of the detected centroid
                steering (float): Calculated steering value based on the PID output from the location of the centroid
                throttle (float): Calcualted throttle based on the offset of the current centroid value
            """

            print(f"begin proc")
            #st = time.time()
            img = cv2.resize(img, (640,360))
            img_rs=img.copy()

            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = np.expand_dims(img, 0)  # add a batch dimension
            img = img.astype(np.float32)
            img=img / 255.0

            # Run segmentation model on inputted image
            img_out = self.sess.run(None, {'input.1': img})

            #end = time.time()

            x0=img_out[0]
            x1=img_out[1]

            # Detect driveable area and lane lines

            # da = driveable area
            # ll = lane lines
            da_predict=np.argmax(x0, 1)
            ll_predict=np.argmax(x1, 1)

            height, width, _ = img_rs.shape
            DA = da_predict.astype(np.uint8)[0]*255
            LL = ll_predict.astype(np.uint8)[0]*255
            img_rs[DA>100]=[255,0,0]
            img_rs[LL>100]=[0,255,0]
            #print(f"Processed image in: {end-st}")

            # Crop image to 60% of the original height in the middle half of the image
            img_rs = img_rs[: (60*height) // 100, width//4:width*3//4]
            crop_h = 60*height // 100
            crop_w = width //2
            image_center_x = crop_w / 2
            # Create binary mask based on pixels in the range 240 to 255
            img_bin = cv2.inRange(img_rs, (240, 0, 0), (255, 0, 0))
            kernel = np.ones((3, 3))
            #img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
            #img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=8)

            # Detect contours in the binary image
            contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            scaled_offset_x = 0
            throttle = 0
            contour_center_x = contour_center_y = 0 
            # Detect the centroid of the road - which should be the largest contour
            if len(contours) >= 1:
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)

                # Get the bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                bounding_center_x = x + w / 2
                bounding_center_y = y + h / 2

                # Find moments of the road and calculate centroid
                # M["m00"]: area of the object
                # M["m10"] and M["m01"]: intrinsic moment values based on the shape of the object
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:  # To avoid division by zero
                    contour_center_x = M["m10"] / M["m00"]
                    contour_center_y = M["m01"] / M["m00"]
                else:
                    contour_center_x = contour_center_y = 0 

                #cv2.circle(
                #    img_rs, (int(bounding_center_x), int(bounding_center_y)), 5, (0, 255, 0), -1
                #)

                # Draw the contour center using moments (Blue)
                cv2.circle(
                    img, (int(contour_center_x), int(contour_center_y)), 5, (0, 255, 0), -1
                )  # Blue for contour moments center

                # Compute the center of the bounding box
                bounding_center_x = x + w / 2
                bounding_center_y = y + h / 2
                
                # Calculate offset based on the X component of the centroid
                offset_x = contour_center_x - image_center_x
                scaled_offset_x = OFFSET_MULTIPLIER * (offset_x / width) # 2
                
                # Set throttle to the straightaway constant if the centroid offset is small
                if abs(scaled_offset_x) <= OFFSET_CUTOFF: # 0.06
                    throttle = STRAIGHTAWAY_THROTTLE # 0.53 
                # Map the throttle to a function based on the value of the X offset of the centroid from the center of the image
                elif abs(scaled_offset_x) >= OFFSET_CUTOFF: # 0.06
                    throttle=-0.0599*math.log(0.0211*abs(scaled_offset_x)) 
                
                # Update the PID setpoint to the current scaled offset TODO make this more descriptive, I don't really understand it
                self.pid.setpoint = scaled_offset_x
                
                print(f"Bounding box center X: {bounding_center_x}")
                print(f"Contour center X: {contour_center_x}")
                print(f"Image center X: {image_center_x}")
                print(f"Offset X (scaled): {scaled_offset_x}")
            else:
                print(f"No contours found")
            
            # Update the steering value based on the previous steering value
            steering = self.pid(self.prev_steer)
            
            return img_rs, (contour_center_x,contour_center_y), steering, throttle

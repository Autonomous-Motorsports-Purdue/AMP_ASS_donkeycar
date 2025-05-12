import pyzed.sl as sl
import cv2
import numpy as np
import pickle

class Zed_Frame_Publisher:
    def __init__(self):
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        #init.camera_disable_self_calib = True
        init.depth_minimum_distance = 0.5

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()

        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        # Access intrinsic parameters
        fx = calibration_params.left_cam.fx  # Focal length in x
        fy = calibration_params.left_cam.fy  # Focal length in y
        cx = calibration_params.left_cam.cx  # Principal point x
        cy = calibration_params.left_cam.cy  # Principal point y
        k1 = calibration_params.left_cam.disto[0]  # Radial distortion coefficient 1
        k2 = calibration_params.left_cam.disto[1]  # Radial distortion coefficient 2
        k3 = calibration_params.left_cam.disto[2]  # Radial distortion coefficient 3
        p1 = calibration_params.left_cam.disto[3]  # Tangential distortion coefficient 1
        p2 = calibration_params.left_cam.disto[5]  # Tangential distortion coefficient 2


        # Turn into a dictionary
        calibration_params = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "p1": p1,
            "p2": p2
        }

        hard_coded = {
            "fx":701.12,
            "fy":701.12,
            "cx":610.83,
            "cy":380.3405,
            "k1":-0.175609,
            "k2":0.0273627,
            "p1":0.000324635,
            "p2":0.00135292,
            "k3":0.0
        }

        print(calibration_params)

        # Dump to binary file
        with open("zed_calibration_params.bin", "wb") as f:
            pickle.dump(hard_coded, f)

        print("ZED camera connected")

    def run(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            left = sl.Mat()
            self.zed.retrieve_image(left, sl.VIEW.LEFT)
            right = sl.Mat()
            self.zed.retrieve_image(right, sl.VIEW.RIGHT)
            depth = sl.Mat()
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            #self.zed.retrieve_image(depth, sl.VIEW.DEPTH)
            # cv2.imshow("ZED", image.get_data())

            return np.array(left.get_data()), np.array(right.get_data()), np.array(depth.get_data())

import numpy as np
import cv2
class Zed_Viewer():
    def __init__(self):
        pass
    def run(self, image, zed_calibration_params):
        if image is not None:
            camera_matrix = np.array([[zed_calibration_params["fx"], 0, zed_calibration_params["cx"]],
                                       [0, zed_calibration_params["fy"], zed_calibration_params["cy"]],
                                       [0, 0, 1]])
            
            dist_coeffs = np.array([zed_calibration_params["k1"], zed_calibration_params["k2"], zed_calibration_params["p1"], zed_calibration_params["p2"], zed_calibration_params["k3"]])

            undistort = cv2.undistort(image, camera_matrix, dist_coeffs, newCameraMatrix=camera_matrix)
            #cv2.imshow("image", np.array(image))
            cv2.imshow("undistort", np.array(undistort))
            cv2.imshow("warped", self.birds_eye(undistort))
            cv2.waitKey(1)
        else:
            print("image is None")

    def birds_eye(self, undistort):
        bottom_offset = 200
        width_offset = 500
        height_offset = 350
        h, w = undistort.shape[:2] # 720, 1280, 550 should be cut off
        src = np.array([[w, h-bottom_offset],    # br
                        [0, h-bottom_offset],    # bl
                        [width_offset, height_offset],   # tl
                        [w-width_offset, height_offset]])  # tr
        dst = np.array([[w, h], [0, h], [0, 0], [w, 0]])
        to_bird = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        from_bird = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))

        

        warped = cv2.warpPerspective(undistort, to_bird, (w, h), flags=cv2.INTER_LINEAR)
        for point in src:
            cv2.circle(undistort, point, 5, (0, 255, 255), -1)
        return warped

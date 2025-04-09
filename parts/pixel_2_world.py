import numpy as np
import pickle

class Pixel_2_world:
    def __init__(self, height=.2, pitch_deg = 0, calib_file=r"parts\zed_calibration_params.bin"):
        self.valid = True
        self.height = height
        self.pitch_deg = pitch_deg

        try:
            with open(calib_file, "rb") as f:
                zed_calibration_params = pickle.load(f)

            self.fx = zed_calibration_params['fx']
            self.fy = zed_calibration_params['fy']
            self.cx = zed_calibration_params['cx']
            self.cy = zed_calibration_params['cy']

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[Pixel_2_world] Failed to load calibration data: {e}")
            self.valid = False
            self.fx = self.fy = self.cx = self.cy = None  # Optional: set to None for safety
            
    def rotate_ray(self,ray,pitch_deg):
        pitch_rad = np.deg2rad(pitch_deg)
        R = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ])
        return R @ ray
    
    def run(self,mid):
        
        if not self.valid:
            return np.array([])
        
        
        positions = []
        
        for v,row in enumerate(mid):
            for u,value in enumerate(row):
                if value:
                    
                    x_n = (u - self.cx) / self.fx
                    y_n = (v - self.cy) / self.fy
                    
                    ray = np.array([x_n, y_n, 1.0])
                    #ray = self.rotate_ray(ray,self.pitch_deg)
                    
                    t = -self.height / ray[2]
                    x = ray[0] * t
                    y = ray[1] * t
                    positions.append([x,y])
                    
                    
        print(positions)
        return np.array(positions)

        
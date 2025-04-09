import numpy as np

class Control_points():
    
    def __init__(self,fov_degrees = 100, height = .3, normal = 1):
        
        self.fov = np.deg2rad(fov_degrees)  # Convert degrees to radians
        self.height = height
        self.normal = normal
    
    def convert_to_points(self,mid):
        H,W = mid.shape[:2]
        
        positions = []
        
        for h,row in enumerate(mid):
            for w,value in enumerate(row):
                if value:
                    deltaH = (H/2 - h)
                    deltaW = (W/2 - w)
                    
                    theta = self.fov*(deltaH/(H/2))
                    phi = self.fov*(deltaW/(W/2))
                    
                    if np.isclose(np.tan(theta),0):
                        continue
                    
                    t = -self.height/np.tan(theta)
                    x = self.normal*t
                    y = np.tan(phi)*t
                    
                    positions.append([x,y])

        return np.array(positions)
    
    def run(self,image):
        
        points = self.convert_to_points(image)
        print(points)
        return points
                    

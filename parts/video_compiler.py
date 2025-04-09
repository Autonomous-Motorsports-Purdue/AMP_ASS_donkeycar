import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2
import os

class Video_compiler:
    def __init__(self,path_1 = r"videos\mid", path_2 = r"videos\points", path_3 = r"videos\stacked"):
        plt.ion()
        self.path_1, self.path_2, self.path_3 = path_1, path_2, path_3
        self.fig, self.ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)  
        
        os.makedirs(self.path_1, exist_ok=True)
        os.makedirs(self.path_2, exist_ok=True)
        os.makedirs(self.path_3, exist_ok=True)
        self.frame_id = 0
        
    def mid_parse(self,mid):
        cv2.imwrite(os.path.join(self.path_1,f"frame_{self.frame_id:04d}.jpg"),mid)
        
    
    def points_parse(self,points):
       
        self.ax.cla()
        x = points[:,1]
        y = points[:,0]
        self.ax.plot(x, y, marker='o')
        self.ax.set_title("Real-Time Point Plot")
        self.ax.set_xlim(left=-.1, right=.1)
        self.ax.set_ylim(bottom=-.1,top=.1)  # Ensures the x-axis includes 0
        self.ax.grid(True)
        self.fig.savefig(os.path.join(self.path_2,f"frame_{self.frame_id:04d}.jpg"))
        self.fig.canvas.flush_events()


    def parse_all(self, mid, points):
        self.ax.cla()
        x = points[:, 1]
        y = points[:, 0]
        self.ax.plot(x, y, marker='o')
        self.ax.set_title("Real-Time Point Plot")
        self.ax.set_xlim(left=-0.1, right=0.1)
        self.ax.set_ylim(bottom=-0.1, top=0.1)
        self.ax.grid(True)

        # Draw the canvas and get RGB image from the plot
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()  # Note: returns (width, height)
        img_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)

        print(f"Original img_array size: {img_array.size}, w: {w}, h: {h}")  # Debugging

        # Reshape img_array to (h, w, 3)
        img_array = img_array.reshape((h, w, 3))

        # Resize the image to match (480, 640)
        target_size = (640, 480)  # OpenCV expects (width, height)
        img_array_resized = cv2.resize(img_array, target_size)

        self.fig.canvas.flush_events()

        # Convert mid (grayscale) to RGB
        mid_rgb = cv2.cvtColor(mid, cv2.COLOR_GRAY2RGB)

        # Resize mid to match the same size as img_array_resized
        mid_rgb_resized = cv2.resize(mid_rgb, target_size)

        # Stack images horizontally
        spliced = np.hstack((mid_rgb_resized, img_array_resized))

        # Save the result
        save_path = os.path.join(self.path_3, f"frame_{self.frame_id:04d}.jpg")
        cv2.imwrite(save_path, spliced)



        
        
    def run(self, mid, points):
        self.parse_all(mid,points)
        self.frame_id+=1
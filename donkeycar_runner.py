import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from parts.camera import Camera
from parts.onnx import Onnx
from parts.process import Process
from parts.viewer import Viewer
from parts.undistort import Undistort
from parts.control_points import Control_points
from parts.pixel_2_world import Pixel_2_world
from parts.points_plot import Points_plot
from parts.video_compiler import Video_compiler
from Parts2.process_o import Process_o

if __name__ == "__main__":
    '''
    # web controller
    V = dk.Vehicle()
    V.mem['img_num'] = 200
    V.add(Camera(), outputs=['img'])
    V.add(Onnx(), inputs=['img'], outputs=['lane', 'drive'])
    #V.add(Process_o(), inputs=['lane', 'drive'], outputs=['sobel', 'curve'])
    V.add(Process(), inputs=['lane', 'drive'], outputs=['sobel', 'curve'])
    V.add(Viewer(), inputs=['drive'])
    V.start(rate_hz=4)
    '''
    # web controller
    V = dk.Vehicle()
    V.mem['img_num'] = 200
    V.add(Camera(), outputs=['img'])
    #V.add(Undistort(),inputs=["img"],outputs=['img'])
    V.add(Onnx(), inputs=['img'], outputs=['lane', 'drive'])
    #V.add(Process_o(), inputs=['lane', 'drive'], outputs=['sobel', 'curve'])
    V.add(Process(), inputs=['lane', 'drive'], outputs=['sobel', 'curve','mid'])
    V.add(Control_points(),inputs = ['mid'],outputs=['points'])
    V.add(Points_plot(),inputs=['points']) 
    V.add(Viewer(), inputs=['drive'])
    V.start(rate_hz=4)
    

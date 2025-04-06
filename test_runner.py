import donkeycar as dk

import donkeycar as dk
from parts.camera import Camera
from parts.viewer import Viewer
from parts.curve_fit import Curve_fit
from parts.onnx import Onnx
from parts.preprocessor import Preprocessor

if __name__ == '__main__':
    V = dk.Vehicle()
    V.mem['img_num'] = 200
    V.add(Camera(), outputs=['raw_img'])
    V.add(Preprocessor(), inputs = ['raw_img'], outputs = ['img'])
    V.add(Onnx(), inputs=['img'], outputs=['lane', 'drive'])
    V.add(Curve_fit(), inputs=['lane', 'drive'], outputs=['sobel', 'curve'])
    V.add(Viewer(), inputs=['drive'])
    V.start(rate_hz=4)

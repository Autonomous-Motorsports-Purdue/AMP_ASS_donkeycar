import numpy as np
import cv2
import onnxruntime as rt

class Onnx:
    def __init__(self):
        self.sess = rt.InferenceSession(
            "./logger/model.onnx",
            providers=[('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"})])
        print('model loaded')

    def run(self, img):
        imgcopy = img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)  # add a batch dimension
        img = img.astype(np.float32)
        img = img / 255.0

        img_out = self.sess.run(None, {'input.1': img})

        x0 = img_out[0]
        x1 = img_out[1]

        # da = driveable area
        # ll = lane lines
        da_predict = np.argmax(x0, 1)
        ll_predict = np.argmax(x1, 1)

        drivable = da_predict.astype(np.uint8)[0]*255
        lanes = ll_predict.astype(np.uint8)[0]*255

        imgcopy[drivable>100]=[255,0,0]
        imgcopy[lanes>100]=[0,255,0]
        cv2.imshow('img', imgcopy)
        cv2.waitKey(1)


        return lanes, drivable
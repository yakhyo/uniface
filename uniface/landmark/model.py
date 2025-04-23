import cv2
import onnx
import onnxruntime
import numpy as np


# from ..data import get_object

from uniface.face_utils import bbox_center_alignment, trans_points

__all__ = [
    'Landmark',
]


class Landmark:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session

        model = onnx.load(self.model_file)

        input_mean = 0.0
        input_std = 1.0

        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', model_file, self.input_mean, self.input_std)

        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name

        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)

        self.input_name = input_name
        self.output_names = output_names

        assert len(self.output_names) == 1

        output_shape = outputs[0].shape
        self.require_pose = False

        self.lmk_dim = 2
        self.lmk_num = output_shape[1]//self.lmk_dim
        self.taskname = 'landmark_%dd_%d' % (self.lmk_dim, self.lmk_num)

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, bbox):
        
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h)*1.5)
        # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        
        aimg, M = bbox_center_alignment(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0/self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)

        return pred


if __name__ == "__main__":
    model = Landmark("2d106det.onnx")

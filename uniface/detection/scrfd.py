import os
import cv2
import numpy as np
import onnxruntime

from typing import Tuple, Optional, List, Literal

# from uniface.logger import Logger
from .utils import non_max_supression, distance2bbox, distance2kps, resize_image

__all__ = ['SCRFD']


class SCRFD:
    """
    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4
    ) -> None:
        """SCRFD initialization

        Args:
            model_path (str): Path model .onnx file.
            input_size (int): Input image size. Defaults to (640, 640)
            max_num (int): Maximum number of detections
            conf_thres (float, optional): Confidence threshold. Defaults to 0.5.
            iou_thres (float, optional): Non-max supression (NMS) threshold. Defaults to 0.4.
        """

        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # SCRFD model params --------------
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

        self.center_cache = {}
        # ---------------------------------

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
            self.input_names = self.session.get_inputs()[0].name
            self.output_names = [x.name for x in self.session.get_outputs()]
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for inference.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Preprocessed blob and input size
        """
        input_size = tuple(image.shape[0:2][::-1])

        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)

        return image, input_size

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Raw model outputs.
        """
        return self.session.run(self.output_names, {self.input_names: input_tensor})

    def postprocess(self, outputs: List[np.ndarray], image_dim: Tuple[int, int]):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_height, input_width = image_dim

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[fmc + idx] * stride
            kps_preds = outputs[2*fmc + idx] * stride

            # Generate anchors
            fm_height = input_height // stride
            fm_width = input_width // stride
            cache_key = (fm_height, fm_width, stride)

            if cache_key in self.center_cache:
                anchor_centers = self.center_cache[cache_key]
            else:
                y, x = np.mgrid[:fm_height, :fm_width]
                anchor_centers = np.stack((x, y), axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)

                if self._num_anchors > 1:
                    anchor_centers = np.tile(anchor_centers[:, None, :], (1, self._num_anchors, 1)).reshape(-1, 2)

                if len(self.center_cache) < 100:
                    self.center_cache[cache_key] = anchor_centers

            pos_indices = np.where(scores >= self.conf_thres)[0]
            if len(pos_indices) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)[pos_indices]
            scores_selected = scores[pos_indices]
            scores_list.append(scores_selected)
            bboxes_list.append(bboxes)

            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            kpss_list.append(kpss[pos_indices])

        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        image: np.ndarray,
        max_num: Optional[int] = 0,
        metric: Literal["default", "max"] = "max",
        center_weight: Optional[float] = 2
    ) -> Tuple[np.ndarray, np.ndarray]:

        original_height, original_width = image.shape[:2]

        image, resize_factor = resize_image(image, target_shape=self.input_size)

        image_tensor, _ = self.preprocess(image)

        outputs = self.inference(image_tensor)

        scores_list, bboxes_list, kpss_list = self.postprocess(outputs, image.shape[:2])

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / resize_factor

        kpss = np.vstack(kpss_list) / resize_factor

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = non_max_supression(pre_det, threshold=self.iou_thres)
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            center = (original_height // 2, original_width // 2)

            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - center[1],
                    (det[:, 1] + det[:, 3]) / 2 - center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * center_weight  # some extra weight on the centering

            sorted_indices = np.argsort(values)[::-1][:max_num]
            det = det[sorted_indices]
            kpss = kpss[sorted_indices]

        return det, kpss


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox[:4].astype(np.int32)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    score = bbox[4]
    cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_keypoints(frame, points, color=(0, 0, 255), radius=2):
    for (x, y) in points.astype(np.int32):
        cv2.circle(frame, (x, y), radius, color, -1)


if __name__ == "__main__":
    detector = SCRFD(model_path="det_10g.onnx")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        exit()

    print("📷 Webcam started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        boxes_list, points_list = detector.detect(frame)

        for boxes, points in zip(boxes_list, points_list):
            draw_bbox(frame, boxes)

            if points is not None:
                draw_keypoints(frame, points)

        cv2.imshow("FaceDetection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

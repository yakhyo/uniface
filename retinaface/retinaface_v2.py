"""
Author: Yakhyokhuja Valikhujaev
Date: 2024-08-07
Description: RetinaFace ONNX Inference for Face Detection
"""

import os
import cv2
import numpy as np
import onnxruntime
import argparse
import torch
from typing import Tuple, List
from .utils import (
    nms,
    decode,
    draw_detections,
    generate_anchors,
    decode_landmarks
)
from .model_store import verify_model_weights


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX Inference Arguments for RetinaFace")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/retinaface_mv2.onnx',
        help='Path to the trained model weights'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save the detection results as images'
    )
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )

    # Image input
    parser.add_argument(
        '--image-path',
        type=str,
        default='./assets/test.jpg',
        help='Path to the input image'
    )

    return parser.parse_args()


class RetinaFaceONNX:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.02,
        pre_nms_topk: int = 5000,
        nms_threshold: float = 0.4,
        post_nms_topk: int = 750,
        vis_threshold: float = 0.6
    ) -> None:
        """Initialize RetinaFace ONNX Inference Model

        Args:
            model_path (str): Path to .onnx model file.
            conf_threshold (float): Confidence threshold for detections.
            pre_nms_topk (int): Max detections before NMS.
            nms_threshold (float): NMS threshold.
            post_nms_topk (int): Max detections after NMS.
            vis_threshold (float): Visualization threshold for displaying detections.
        """
        self.conf_threshold = conf_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.post_nms_topk = post_nms_topk
        self.vis_threshold = vis_threshold

        # Initialize the ONNX session
        self._initialize_model(model_path)

    def __call__(self, image_path: str, save_image: bool = False) -> None:
        """Run inference on the image and optionally save the output."""
        original_image = cv2.imread(image_path)
        input_tensor = self.preprocess(original_image)
        
        outputs = self.inference(input_tensor)
        detections = self.postprocess(outputs, original_image.shape[1], original_image.shape[0])

        draw_detections(original_image, detections, self.vis_threshold)
        
        if save_image:
            self._save_output_image(original_image, image_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize ONNX model session from the given path."""
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, 
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, C, H, W)
        """
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        return np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            List[np.ndarray]: Raw model outputs.
        """
        return self.session.run(None, {self.input_name: input_tensor})

    def postprocess(self, outputs: List[np.ndarray], img_width: int, img_height: int) -> np.ndarray:
        """Process model outputs to final detection results.

        Args:
            outputs (List[np.ndarray]): Raw model outputs.
            img_width (int): Width of the original image.
            img_height (int): Height of the original image.

        Returns:
            np.ndarray: Filtered and processed detections with landmarks.
        """
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Generate anchor boxes and decode boxes, landmarks
        priors = generate_anchors((img_height, img_width))
        boxes = decode(torch.tensor(loc), priors).cpu().numpy()
        landmarks = decode_landmarks(torch.tensor(landmarks), priors).cpu().numpy()

        # Scale boxes and landmarks to original image size
        boxes, landmarks = self._scale_detections(boxes, landmarks, img_width, img_height)
        
        # Extract confidence scores for the face class
        scores = conf[:, 1]
        mask = scores > self.conf_threshold

        # Filter, sort, and apply NMS
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]
        return self._apply_nms(boxes, landmarks, scores)

    def _scale_detections(self, boxes: np.ndarray, landmarks: np.ndarray, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Scale bounding boxes and landmarks to the original image size."""
        bbox_scale = np.array([img_width, img_height] * 2)
        boxes = boxes * bbox_scale

        landmark_scale = np.array([img_width, img_height] * 5)
        landmarks = landmarks * landmark_scale

        return boxes, landmarks

    def _apply_nms(self, boxes: np.ndarray, landmarks: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression to filter overlapping boxes.

        Args:
            boxes (np.ndarray): Bounding boxes.
            landmarks (np.ndarray): Landmark points.
            scores (np.ndarray): Confidence scores.

        Returns:
            np.ndarray: Detections after applying NMS.
        """
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
        
        # Stack boxes and scores for NMS and keep top detections
        detections = np.hstack((boxes, scores[:, None]))
        keep = nms(detections, self.nms_threshold)
        return np.hstack((boxes[keep], scores[keep][:, None], landmarks[keep]))

    def _save_output_image(self, original_image: np.ndarray, image_path: str) -> None:
        """Save the detection output image."""
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{im_name}_onnx_out.jpg"
        cv2.imwrite(save_name, original_image)
        print(f"Image saved at '{save_name}'")


if __name__ == '__main__':
    args = parse_arguments()

    # Initialize and run the ONNX inference
    retinaface_inference = RetinaFaceONNX(
        model_path=verify_model_weights("retinaface_mnet_v2"),
        conf_threshold=args.conf_threshold,
        pre_nms_topk=args.pre_nms_topk,
        nms_threshold=args.nms_threshold,
        post_nms_topk=args.post_nms_topk,
        vis_threshold=args.vis_threshold
    )

    import time
    avg = 0
    for _ in range(50):
        st = time.time()
        retinaface_inference(args.image_path, save_image=args.save_image)
        d = time.time() - st
        avg += d
    print("avg", avg/50)

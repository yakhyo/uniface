import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import torch

from .utils import (
    nms,
    decode,
    draw_detections,
    generate_anchors,
    decode_landmarks
)
from .model_store import verify_model_weights
from typing import Tuple, List
from .log import logger


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


def resize_image(frame, target_shape=(640, 640)):
    width, height = target_shape

    # Aspect-ratio preserving resize
    im_ratio = float(frame.shape[0]) / frame.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resize_factor = float(new_height) / frame.shape[0]
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create blank image and place resized image on it
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:new_height, :new_width, :] = resized_frame

    return image, resize_factor


class RetinaFace:
    def __init__(
        self,
        model_path,
        conf_threshold=0.02,
        pre_nms_topk=5000,
        nms_threshold=0.4,
        post_nms_topk=750,
        vis_threshold=0.6
    ) -> None:

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.post_nms_topk = post_nms_topk
        self.vis_threshold = vis_threshold

        # Generate anchor boxes
        self.priors = generate_anchors(image_size=(640, 640))

        # Load ONNX model
        self._initialize_model(model_path)

    def _initialize_model(self, model_path: str) -> None:
        """
        Initializes an ONNX model session from the given path.

        Args:
            model_path (str): The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load, logs an error and raises an exception.
        """
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Successfully initialized the model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from '{model_path}': {e}")
            raise RuntimeError(f"Failed to initialize model session for '{model_path}'") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image for model inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, C, H, W)
        """
        image = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image

    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Perform model inference on the preprocessed image tensor.

        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.

        Returns:
            List[np.ndarray]: Raw model outputs.
        """
        return self.session.run(None, {self.input_name: input_tensor})

    def infer(self, image):
        # Load and preprocess image
        image, resize_factor = resize_image(image)
        height, width, _ = image.shape

        image_tensor = self.preprocess(image)

        # Run ONNX model inference
        outputs = self.inference(image_tensor)
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Decode boxes and landmarks
        boxes = decode(torch.tensor(loc), self.priors).cpu().numpy()
        landmarks = decode_landmarks(torch.tensor(landmarks), self.priors).cpu().numpy()

        boxes, landmarks = self._scale_detections(boxes, landmarks, resize_factor, height, width)

        scores = conf[:, 1]  # Confidence scores for class 1 (face)

        # Filter by confidence threshold
        inds = scores > self.conf_threshold
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # Sort by scores
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k detections
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]

        # Concatenate detections and landmarks
        return np.concatenate((detections, landmarks), axis=1)
    
    
    def _scale_detections(self, boxes: np.ndarray, landmarks: np.ndarray, resize_factor: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Scale bounding boxes and landmarks to the original image size."""
        bbox_scale = np.array([width, height] * 2)
        boxes = boxes * bbox_scale / resize_factor

        landmark_scale = np.array([width, height] * 5)
        landmarks = landmarks * landmark_scale / resize_factor

        return boxes, landmarks

    def save_output_image(self, original_image, image_path):
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{im_name}_onnx_out.jpg"
        cv2.imwrite(save_name, original_image)
        print(f"Image saved at '{save_name}'")

    def run_inference(self, image_path, save_image=False):
        original_image = original_image = cv2.imread(image_path)
        detections = self.infer(original_image)
        draw_detections(original_image, detections, self.vis_threshold)

        if save_image:
            self.save_output_image(original_image, image_path)


if __name__ == '__main__':
    args = parse_arguments()
    import time
    # Initialize and run the ONNX inference
    retinaface_inference = RetinaFace(
        model_path=verify_model_weights("retinaface_mnet_v2"),
        conf_threshold=args.conf_threshold,
        pre_nms_topk=args.pre_nms_topk,
        nms_threshold=args.nms_threshold,
        post_nms_topk=args.post_nms_topk,
        vis_threshold=args.vis_threshold
    )

    retinaface_inference.run_inference(args.image_path, save_image=args.save_image)
    # avg = 0
    # for _ in range(50):
    #     st = time.time()
    #     retinaface_inference.run_inference(args.image_path, save_image=args.save_image)
    #     d = time.time() - st
    #     avg += d
    # print("avg", avg/50)

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
from .config import get_config

from.model_store import download


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ONNX model
        self.ort_session = ort.InferenceSession(model_path)
        
        
    @staticmethod
    def preprocess_image(image, rgb_mean=(104, 117, 123)):
        image = np.float32(image)
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image

    def infer(self, image_path):
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        img_height, img_width, _ = original_image.shape
        image = self.preprocess_image(original_image)

        # Run ONNX model inference
        outputs = self.ort_session.run(None, {'input': image})
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Generate anchor boxes
        priors = generate_anchors(image_size=(img_height, img_width))

        # Decode boxes and landmarks
        boxes = decode(torch.tensor(loc), priors).to(self.device)
        landmarks = decode_landmarks(torch.tensor(landmarks), priors).to(self.device)

        # Adjust scales for boxes and landmarks
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=self.device)
        landmarks = (landmarks * landmark_scale).cpu().numpy()

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
        return np.concatenate((detections, landmarks), axis=1), original_image

    def save_output_image(self, original_image, image_path):
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{im_name}_onnx_out.jpg"
        cv2.imwrite(save_name, original_image)
        print(f"Image saved at '{save_name}'")

    def run_inference(self, image_path, save_image=False):
        detections, original_image = self.infer(image_path)
        draw_detections(original_image, detections, self.vis_threshold)

        if save_image:
            self.save_output_image(original_image, image_path)


if __name__ == '__main__':
    args = parse_arguments()

    # Initialize and run the ONNX inference
    retinaface_inference = RetinaFace(
        model_path=download("retinaface_mnet_v2"),
        conf_threshold=args.conf_threshold,
        pre_nms_topk=args.pre_nms_topk,
        nms_threshold=args.nms_threshold,
        post_nms_topk=args.post_nms_topk,
        vis_threshold=args.vis_threshold
    )

    retinaface_inference.run_inference(args.image_path, save_image=args.save_image)

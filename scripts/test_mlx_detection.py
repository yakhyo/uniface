#!/usr/bin/env python3
"""
End-to-end test of MLX RetinaFace face detection.

This script tests the complete detection pipeline including:
- Image preprocessing
- MLX inference with fused weights
- Postprocessing (decode, NMS)
- Comparison with ONNX detection results
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mlx.core as mx

from uniface.common import (
    decode_boxes,
    decode_landmarks,
    generate_anchors,
    non_max_suppression,
    resize_image,
)
from uniface.constants import RetinaFaceWeights
from uniface.detection.retinaface_mlx import RetinaFaceNetworkFused
from uniface.mlx_utils import load_mlx_fused_weights, synchronize
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session


def detect_faces_mlx(model, image, conf_thresh=0.5, nms_thresh=0.4):
    """Run face detection using MLX model."""
    original_height, original_width = image.shape[:2]
    input_size = (640, 640)

    # Resize
    image_resized, resize_factor = resize_image(image, target_shape=input_size)
    height, width, _ = image_resized.shape

    # Preprocess (subtract mean, NHWC format)
    processed = np.float32(image_resized) - np.array([104, 117, 123], dtype=np.float32)
    input_tensor = mx.array(np.expand_dims(processed, 0))

    # Inference
    cls_preds, bbox_preds, landmark_preds = model(input_tensor)
    synchronize(cls_preds, bbox_preds, landmark_preds)

    # Apply softmax
    cls_probs = mx.softmax(cls_preds, axis=-1)

    # Convert to numpy
    loc = np.array(bbox_preds).squeeze(0)
    conf = np.array(cls_probs).squeeze(0)
    landmarks = np.array(landmark_preds).squeeze(0)

    # Generate anchors
    priors = generate_anchors(image_size=input_size)

    # Decode boxes and landmarks
    boxes = decode_boxes(loc, priors)
    landmarks = decode_landmarks(landmarks, priors)

    # Scale back to original image size
    bbox_scale = np.array([width, height] * 2)
    boxes = boxes * bbox_scale / resize_factor

    landmark_scale = np.array([width, height] * 5)
    landmarks = landmarks * landmark_scale / resize_factor

    # Extract face class scores
    scores = conf[:, 1]
    mask = scores > conf_thresh

    boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

    # Sort by score and apply NMS
    order = scores.argsort()[::-1][:5000]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = non_max_suppression(detections, nms_thresh)
    detections, landmarks = detections[keep], landmarks[keep]

    # Build output
    faces = []
    for i in range(detections.shape[0]):
        face_dict = {
            'bbox': detections[i, :4],
            'confidence': float(detections[i, 4]),
            'landmarks': landmarks[i].reshape(5, 2),
        }
        faces.append(face_dict)

    return faces


def detect_faces_onnx(session, image, conf_thresh=0.5, nms_thresh=0.4):
    """Run face detection using ONNX model."""
    original_height, original_width = image.shape[:2]
    input_size = (640, 640)

    # Resize
    image_resized, resize_factor = resize_image(image, target_shape=input_size)
    height, width, _ = image_resized.shape

    # Preprocess (subtract mean, NCHW format)
    processed = np.float32(image_resized) - np.array([104, 117, 123], dtype=np.float32)
    processed = processed.transpose(2, 0, 1)  # HWC -> CHW
    input_tensor = np.expand_dims(processed, 0)

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    loc = outputs[0].squeeze(0)
    conf = outputs[1].squeeze(0)  # Already softmaxed
    landmarks = outputs[2].squeeze(0)

    # Generate anchors
    priors = generate_anchors(image_size=input_size)

    # Decode boxes and landmarks
    boxes = decode_boxes(loc, priors)
    landmarks = decode_landmarks(landmarks, priors)

    # Scale back to original image size
    bbox_scale = np.array([width, height] * 2)
    boxes = boxes * bbox_scale / resize_factor

    landmark_scale = np.array([width, height] * 5)
    landmarks = landmarks * landmark_scale / resize_factor

    # Extract face class scores
    scores = conf[:, 1]
    mask = scores > conf_thresh

    boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]

    # Sort by score and apply NMS
    order = scores.argsort()[::-1][:5000]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = non_max_suppression(detections, nms_thresh)
    detections, landmarks = detections[keep], landmarks[keep]

    # Build output
    faces = []
    for i in range(detections.shape[0]):
        face_dict = {
            'bbox': detections[i, :4],
            'confidence': float(detections[i, 4]),
            'landmarks': landmarks[i].reshape(5, 2),
        }
        faces.append(face_dict)

    return faces


def main():
    print('=' * 70)
    print('End-to-End Face Detection Test: MLX vs ONNX')
    print('=' * 70)

    # Load test image
    test_image_path = Path(__file__).parent.parent / 'assets' / 'test.jpg'
    if not test_image_path.exists():
        test_image_path = Path('/tmp/test_face.jpg')
    if not test_image_path.exists():
        print('No test image found. Downloading a sample...')
        import urllib.request

        url = 'https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg'
        urllib.request.urlretrieve(url, '/tmp/test_face.jpg')
        test_image_path = Path('/tmp/test_face.jpg')

    image = cv2.imread(str(test_image_path))
    print(f'Test image: {test_image_path}')
    print(f'Image size: {image.shape}')

    # Load ONNX model
    print('\nLoading ONNX model...')
    onnx_path = verify_model_weights(RetinaFaceWeights.MNET_V2)
    onnx_session = create_onnx_session(onnx_path)

    # Load MLX model
    print('Loading MLX model...')
    mlx_model = RetinaFaceNetworkFused(backbone_type='mobilenetv2', width_mult=1.0)
    weights_path = Path(__file__).parent.parent / 'weights_mlx_fused' / 'retinaface_mnet_v2.safetensors'

    if weights_path.exists():
        load_mlx_fused_weights(mlx_model, str(weights_path))
    else:
        print(f'ERROR: Fused weights not found at {weights_path}')
        return

    mlx_model.train(False)

    # Run detection
    print('\nRunning ONNX detection...')
    onnx_faces = detect_faces_onnx(onnx_session, image)

    print('Running MLX detection...')
    mlx_faces = detect_faces_mlx(mlx_model, image)

    # Compare results
    print('\n' + '=' * 70)
    print('Detection Results')
    print('=' * 70)

    print(f'\nONNX detected: {len(onnx_faces)} face(s)')
    print(f'MLX detected:  {len(mlx_faces)} face(s)')

    if len(onnx_faces) > 0 and len(mlx_faces) > 0:
        print('\nONNX faces:')
        for i, face in enumerate(onnx_faces[:5]):
            print(f'  Face {i + 1}: bbox={face["bbox"].astype(int)}, conf={face["confidence"]:.4f}')

        print('\nMLX faces:')
        for i, face in enumerate(mlx_faces[:5]):
            print(f'  Face {i + 1}: bbox={face["bbox"].astype(int)}, conf={face["confidence"]:.4f}')

        # Compare detections
        print('\n' + '=' * 70)
        print('Comparison (first detection)')
        print('=' * 70)

        onnx_bbox = onnx_faces[0]['bbox']
        mlx_bbox = mlx_faces[0]['bbox']
        bbox_diff = np.abs(onnx_bbox - mlx_bbox).max()

        onnx_lmk = onnx_faces[0]['landmarks']
        mlx_lmk = mlx_faces[0]['landmarks']
        lmk_diff = np.abs(onnx_lmk - mlx_lmk).max()

        conf_diff = abs(onnx_faces[0]['confidence'] - mlx_faces[0]['confidence'])

        print(f'\nBBox max diff: {bbox_diff:.4f} pixels')
        print(f'Landmark max diff: {lmk_diff:.4f} pixels')
        print(f'Confidence diff: {conf_diff:.6f}')

        if bbox_diff < 1.0 and lmk_diff < 1.0:
            print('\n✓ SUCCESS: MLX detections match ONNX!')
        else:
            print('\n✗ WARNING: Detections differ more than expected')

    # Save visualization
    output_dir = Path(__file__).parent.parent / 'assets'
    output_dir.mkdir(exist_ok=True)

    # Draw MLX detections
    viz_image = image.copy()
    for face in mlx_faces:
        bbox = face['bbox'].astype(int)
        cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            viz_image,
            f'{face["confidence"]:.2f}',
            (bbox[0], bbox[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        for lmk in face['landmarks']:
            cv2.circle(viz_image, (int(lmk[0]), int(lmk[1])), 2, (0, 0, 255), -1)

    output_path = output_dir / 'mlx_detection_result.jpg'
    cv2.imwrite(str(output_path), viz_image)
    print(f'\nVisualization saved to: {output_path}')


if __name__ == '__main__':
    main()

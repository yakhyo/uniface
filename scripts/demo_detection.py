#!/usr/bin/env python3
"""Demo script to detect faces and save annotated image with timing."""

import time
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

import os
os.environ['UNIFACE_BACKEND'] = 'mlx'

import mlx.core as mx
from uniface.detection.retinaface_mlx import RetinaFaceNetworkFused
from uniface.mlx_utils import load_mlx_fused_weights, synchronize
from uniface.common import generate_anchors, decode_boxes, decode_landmarks, non_max_suppression, resize_image

def main():
    # Paths
    input_path = project_dir / 'images' / 'image.png'
    output_path = project_dir / 'images' / 'image_detected.png'
    weights_path = project_dir / 'weights_mlx_fused' / 'retinaface_mnet_v2.safetensors'
    
    print(f"Input image: {input_path}")
    
    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"ERROR: Could not load image from {input_path}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Initialize detector
    print("\nInitializing RetinaFace (MLX)...")
    init_start = time.perf_counter()
    
    model = RetinaFaceNetworkFused(backbone_type='mobilenetv2', width_mult=1.0)
    load_mlx_fused_weights(model, str(weights_path))
    model.train(False)
    
    init_time = time.perf_counter() - init_start
    print(f"Initialization time: {init_time:.3f}s")
    
    # Detection parameters
    input_size = (640, 640)
    conf_thresh = 0.5
    nms_thresh = 0.4
    mean = np.array([104, 117, 123], dtype=np.float32)
    
    # Precompute anchors
    anchors = generate_anchors(image_size=input_size)
    
    def detect_faces(img):
        # Resize image
        resized, scale = resize_image(img, input_size)
        
        # Preprocess
        input_data = np.float32(resized) - mean
        input_data = np.expand_dims(input_data, 0)
        input_tensor = mx.array(input_data)
        
        # Inference
        cls_preds, bbox_preds, lmk_preds = model(input_tensor)
        synchronize(cls_preds, bbox_preds, lmk_preds)
        
        # Convert to numpy
        cls_scores = np.array(cls_preds)[0]
        bbox_deltas = np.array(bbox_preds)[0]
        lmk_deltas = np.array(lmk_preds)[0]
        
        # Apply softmax
        cls_scores = np.exp(cls_scores) / np.exp(cls_scores).sum(axis=-1, keepdims=True)
        scores = cls_scores[:, 1]
        
        # Decode
        boxes = decode_boxes(bbox_deltas, anchors, variances=[0.1, 0.2])
        landmarks = decode_landmarks(lmk_deltas, anchors, variances=[0.1, 0.2])
        
        # Filter by confidence
        mask = scores > conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        landmarks = landmarks[mask]
        
        # NMS - needs [x1, y1, x2, y2, score] format
        if len(boxes) > 0:
            dets = np.hstack([boxes, scores[:, np.newaxis]])
            keep = non_max_suppression(dets, nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            landmarks = landmarks[keep]
        
        # Scale back
        boxes = boxes / scale
        landmarks = landmarks / scale
        
        return boxes, scores, landmarks
    
    # Warm-up
    print("\nWarm-up inference...")
    _ = detect_faces(image)
    
    # Run detection with timing (average of 10 runs)
    print("\nRunning face detection (averaging 10 runs)...")
    times = []
    for _ in range(10):
        detect_start = time.perf_counter()
        boxes, scores, landmarks = detect_faces(image)
        detect_time = time.perf_counter() - detect_start
        times.append(detect_time)
    
    avg_time = np.mean(times)
    num_faces = len(boxes)
    
    print(f"\n{'='*50}")
    print(f"RESULTS (MLX Backend on Apple Silicon)")
    print(f"{'='*50}")
    print(f"Image size: {w}x{h}")
    print(f"Faces detected: {num_faces}")
    print(f"Average detection time: {avg_time*1000:.1f}ms")
    print(f"FPS: {1/avg_time:.1f}")
    print(f"{'='*50}")
    
    # Draw detections
    output_image = image.copy()
    for i in range(num_faces):
        bbox = boxes[i].astype(int)
        conf = scores[i]
        lmks = landmarks[i]
        
        # Bounding box
        cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Confidence label
        label = f"{conf:.2f}"
        cv2.putText(output_image, label, (bbox[0], bbox[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Landmarks (5 points x 2 coords = 10 values, reshape to 5x2)
        if lmks.size >= 10:
            lmks = lmks.reshape(-1, 2)[:5]
            for px, py in lmks:
                cv2.circle(output_image, (int(px), int(py)), 3, (0, 0, 255), -1)
    
    # Save
    cv2.imwrite(str(output_path), output_image)
    print(f"\nSaved: {output_path}")

if __name__ == '__main__':
    main()

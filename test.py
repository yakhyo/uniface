import os
import cv2
import numpy as np

from uniface import RetinaFace, draw_detections


def run_inference(image_path, save_image=False, vis_threshold=0.6):
    """
    Perform inference on an image, draw detections, and optionally save the output image.

    Args:
        image_path (str): Path to the input image.
        save_image (bool): Whether to save the output image with detections.
        vis_threshold (float): Confidence threshold for displaying detections.
    """
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Perform face detection
    boxes, landmarks = retinaface_inference.detect(original_image)

    # Draw detections on the image
    draw_detections(original_image, (boxes, landmarks), vis_threshold)

    # Save the output image if requested
    if save_image:
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{im_name}_out.jpg"
        cv2.imwrite(save_name, original_image)
        print(f"Image saved at '{save_name}'")


if __name__ == '__main__':
    import time

    # Initialize and run the ONNX inference
    retinaface_inference = RetinaFace(
        model="retinaface_mnet_v2",
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
    )

    img_path = "assets/test.jpg"
    avg = 0
    for _ in range(50):
        st = time.time()
        run_inference(img_path, save_image=True, vis_threshold=0.6)
        d = time.time() - st
        print(d)
        avg += d
    print("avg", avg / 50)

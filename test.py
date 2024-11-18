from retinaface import RetinaFace
import cv2
import os
import numpy as np

def draw_detections(original_image, detections, vis_threshold):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    detections, landmarks = detections

    # Filter by confidence
    filtered = detections[:, 4] >= vis_threshold

    print(f"#faces: {sum(filtered)}")

    # Slice arrays efficiently
    detections = detections[filtered]
    landmarks = landmarks[filtered]

    boxes = detections[:, :4].astype(np.int32)
    scores = detections[:, 4]

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)


def save_output_image(original_image, image_path):
    im_name = os.path.splitext(os.path.basename(image_path))[0]
    save_name = f"{im_name}_onnx_out.jpg"
    cv2.imwrite(save_name, original_image)
    print(f"Image saved at '{save_name}'")


def run_inference(image_path, save_image=False, vis_threshold=0.6):
    original_image = original_image = cv2.imread(image_path)
    detections, landmarks = retinaface_inference.detect(original_image)
    draw_detections(original_image, (detections, landmarks), vis_threshold)

    if save_image:
        save_output_image(original_image, image_path)


if __name__ == '__main__':
    import time
    # Initialize and run the ONNX inference
    retinaface_inference = RetinaFace(
        model="retinaface_mnet_v2",
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
        # input_size=(1024, 1024)
    )
    
    img_path = "assets/test.jpg"
    avg = 0
    for _ in range(50):
        st = time.time()
        run_inference(img_path, save_image=True, vis_threshold=0.6)
        d = time.time() - st
        print(d)
        avg += d
    print("avg", avg/50)

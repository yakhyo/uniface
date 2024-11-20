import pytest
import numpy as np
from uniface import RetinaFace


@pytest.fixture
def retinaface_model():
    """
    Fixture to initialize the RetinaFace model for testing.
    """
    return RetinaFace(
        model="retinaface_mnet_v2",
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
    )


def test_model_initialization(retinaface_model):
    """
    Test that the RetinaFace model initializes correctly.
    """
    assert retinaface_model is not None, "Model initialization failed."


def test_inference_on_640x640_image(retinaface_model):
    """
    Test inference on a 640x640 BGR image.
    """
    # Generate a mock 640x640 BGR image
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run inference
    detections, landmarks = retinaface_model.detect(mock_image)

    # Check output types
    assert isinstance(detections, np.ndarray), "Detections should be a numpy array."
    assert isinstance(landmarks, np.ndarray), "Landmarks should be a numpy array."

    # Check that detections have the expected shape
    if detections.size > 0:  # If faces are detected
        assert detections.shape[1] == 5, "Each detection should have 5 values (x1, y1, x2, y2, score)."

    # Check landmarks shape
    if landmarks.size > 0:
        assert landmarks.shape[1:] == (5, 2), "Landmarks should have shape (N, 5, 2)."


def test_confidence_threshold(retinaface_model):
    """
    Test that detections respect the confidence threshold.
    """
    # Generate a mock 640x640 BGR image
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run inference
    detections, _ = retinaface_model.detect(mock_image)

    # Ensure all detections have confidence scores above the threshold
    if detections.size > 0:  # If faces are detected
        confidence_scores = detections[:, 4]
        assert (confidence_scores >= 0.5).all(), "Some detections have confidence below the threshold."


def test_no_faces_detected(retinaface_model):
    """
    Test inference on an image without detectable faces.
    """
    # Generate an empty (black) 640x640 image
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Run inference
    detections, landmarks = retinaface_model.detect(empty_image)

    # Ensure no detections or landmarks are found
    assert detections.size == 0, "Detections should be empty for a blank image."
    assert landmarks.size == 0, "Landmarks should be empty for a blank image."

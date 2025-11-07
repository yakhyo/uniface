import numpy as np
import pytest

from uniface.constants import RetinaFaceWeights
from uniface.detection import RetinaFace


@pytest.fixture
def retinaface_model():
    """
    Fixture to initialize the RetinaFace model for testing.
    """
    return RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
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

    # Run inference - returns list of dictionaries
    faces = retinaface_model.detect(mock_image)

    # Check output type
    assert isinstance(faces, list), "Detections should be a list."

    # Check that each face has the expected structure
    for face in faces:
        assert isinstance(face, dict), "Each detection should be a dictionary."
        assert "bbox" in face, "Each detection should have a 'bbox' key."
        assert "confidence" in face, "Each detection should have a 'confidence' key."
        assert "landmarks" in face, "Each detection should have a 'landmarks' key."

        # Check bbox format
        bbox = face["bbox"]
        assert len(bbox) == 4, "BBox should have 4 values (x1, y1, x2, y2)."

        # Check landmarks format
        landmarks = face["landmarks"]
        assert len(landmarks) == 5, "Should have 5 landmark points."
        assert all(len(pt) == 2 for pt in landmarks), "Each landmark should be (x, y)."


def test_confidence_threshold(retinaface_model):
    """
    Test that detections respect the confidence threshold.
    """
    # Generate a mock 640x640 BGR image
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run inference
    faces = retinaface_model.detect(mock_image)

    # Ensure all detections have confidence scores above the threshold
    for face in faces:
        confidence = face["confidence"]
        assert confidence >= 0.5, f"Detection has confidence {confidence} below threshold 0.5"


def test_no_faces_detected(retinaface_model):
    """
    Test inference on an image without detectable faces.
    """
    # Generate an empty (black) 640x640 image
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Run inference
    faces = retinaface_model.detect(empty_image)

    # Ensure no detections are found
    assert len(faces) == 0, "Should detect no faces in a blank image."

import numpy as np
import pytest

from uniface.constants import SCRFDWeights
from uniface.detection import SCRFD


@pytest.fixture
def scrfd_model():
    return SCRFD(
        model_name=SCRFDWeights.SCRFD_500M_KPS,
        conf_thresh=0.5,
        nms_thresh=0.4,
    )


def test_model_initialization(scrfd_model):
    assert scrfd_model is not None, 'Model initialization failed.'


def test_inference_on_640x640_image(scrfd_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(mock_image)

    assert isinstance(faces, list), 'Detections should be a list.'

    for face in faces:
        assert isinstance(face, dict), 'Each detection should be a dictionary.'
        assert 'bbox' in face, "Each detection should have a 'bbox' key."
        assert 'confidence' in face, "Each detection should have a 'confidence' key."
        assert 'landmarks' in face, "Each detection should have a 'landmarks' key."

        bbox = face['bbox']
        assert len(bbox) == 4, 'BBox should have 4 values (x1, y1, x2, y2).'

        landmarks = face['landmarks']
        assert len(landmarks) == 5, 'Should have 5 landmark points.'
        assert all(len(pt) == 2 for pt in landmarks), 'Each landmark should be (x, y).'


def test_confidence_threshold(scrfd_model):
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(mock_image)

    for face in faces:
        confidence = face['confidence']
        assert confidence >= 0.5, f'Detection has confidence {confidence} below threshold 0.5'


def test_no_faces_detected(scrfd_model):
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = scrfd_model.detect(empty_image)
    assert len(faces) == 0, 'Should detect no faces in a blank image.'


def test_different_input_sizes(scrfd_model):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        faces = scrfd_model.detect(mock_image)
        assert isinstance(faces, list), f'Should return list for size {size}'


def test_scrfd_10g_model():
    model = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, conf_thresh=0.5)
    assert model is not None, 'SCRFD 10G model initialization failed.'

    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    faces = model.detect(mock_image)
    assert isinstance(faces, list), 'SCRFD 10G should return list of detections.'

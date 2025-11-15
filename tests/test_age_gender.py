import numpy as np
import pytest

from uniface.attribute import AgeGender


@pytest.fixture
def age_gender_model():
    return AgeGender()


@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_bbox():
    return [100, 100, 300, 300]


def test_model_initialization(age_gender_model):
    assert age_gender_model is not None, "AgeGender model initialization failed."


def test_prediction_output_format(age_gender_model, mock_image, mock_bbox):
    gender, age = age_gender_model.predict(mock_image, mock_bbox)
    assert isinstance(gender, str), f"Gender should be string, got {type(gender)}"
    assert isinstance(age, int), f"Age should be int, got {type(age)}"


def test_gender_values(age_gender_model, mock_image, mock_bbox):
    gender, age = age_gender_model.predict(mock_image, mock_bbox)
    assert gender in ['Male', 'Female'], f"Gender should be 'Male' or 'Female', got '{gender}'"


def test_age_range(age_gender_model, mock_image, mock_bbox):
    gender, age = age_gender_model.predict(mock_image, mock_bbox)
    assert 0 <= age <= 120, f"Age should be between 0 and 120, got {age}"


def test_different_bbox_sizes(age_gender_model, mock_image):
    test_bboxes = [
        [50, 50, 150, 150],
        [100, 100, 300, 300],
        [50, 50, 400, 400],
    ]

    for bbox in test_bboxes:
        gender, age = age_gender_model.predict(mock_image, bbox)
        assert gender in ['Male', 'Female'], f"Failed for bbox {bbox}"
        assert 0 <= age <= 120, f"Age out of range for bbox {bbox}"


def test_different_image_sizes(age_gender_model, mock_bbox):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        gender, age = age_gender_model.predict(mock_image, mock_bbox)
        assert gender in ['Male', 'Female'], f"Failed for image size {size}"
        assert 0 <= age <= 120, f"Age out of range for image size {size}"


def test_consistency(age_gender_model, mock_image, mock_bbox):
    gender1, age1 = age_gender_model.predict(mock_image, mock_bbox)
    gender2, age2 = age_gender_model.predict(mock_image, mock_bbox)

    assert gender1 == gender2, "Same input should produce same gender prediction"
    assert age1 == age2, "Same input should produce same age prediction"


def test_bbox_list_format(age_gender_model, mock_image):
    bbox_list = [100, 100, 300, 300]
    gender, age = age_gender_model.predict(mock_image, bbox_list)
    assert gender in ['Male', 'Female'], "Should work with bbox as list"
    assert 0 <= age <= 120, "Age should be in valid range"


def test_bbox_array_format(age_gender_model, mock_image):
    bbox_array = np.array([100, 100, 300, 300])
    gender, age = age_gender_model.predict(mock_image, bbox_array)
    assert gender in ['Male', 'Female'], "Should work with bbox as numpy array"
    assert 0 <= age <= 120, "Age should be in valid range"


def test_multiple_predictions(age_gender_model, mock_image):
    bboxes = [
        [50, 50, 150, 150],
        [200, 200, 350, 350],
        [400, 400, 550, 550],
    ]

    results = []
    for bbox in bboxes:
        gender, age = age_gender_model.predict(mock_image, bbox)
        results.append((gender, age))

    assert len(results) == 3, "Should have 3 predictions"
    for gender, age in results:
        assert gender in ['Male', 'Female']
        assert 0 <= age <= 120


def test_age_is_positive(age_gender_model, mock_image, mock_bbox):
    for _ in range(5):
        gender, age = age_gender_model.predict(mock_image, mock_bbox)
        assert age >= 0, f"Age should be non-negative, got {age}"


def test_output_format_for_visualization(age_gender_model, mock_image, mock_bbox):
    gender, age = age_gender_model.predict(mock_image, mock_bbox)
    text = f"{gender}, {age}y"
    assert isinstance(text, str), "Should be able to format as string"
    assert "Male" in text or "Female" in text, "Text should contain gender"
    assert "y" in text, "Text should contain 'y' for years"

# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for AgeGender attribute predictor."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.attribute import AgeGender, AttributeResult


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
    assert age_gender_model is not None, 'AgeGender model initialization failed.'


def test_prediction_output_format(age_gender_model, mock_image, mock_bbox):
    result = age_gender_model.predict(mock_image, mock_bbox)
    assert isinstance(result, AttributeResult), f'Result should be AttributeResult, got {type(result)}'
    assert isinstance(result.gender, int), f'Gender should be int, got {type(result.gender)}'
    assert isinstance(result.age, int), f'Age should be int, got {type(result.age)}'
    assert isinstance(result.sex, str), f'Sex should be str, got {type(result.sex)}'


def test_gender_values(age_gender_model, mock_image, mock_bbox):
    result = age_gender_model.predict(mock_image, mock_bbox)
    assert result.gender in [0, 1], f'Gender should be 0 (Female) or 1 (Male), got {result.gender}'
    assert result.sex in ['Female', 'Male'], f'Sex should be Female or Male, got {result.sex}'


def test_age_range(age_gender_model, mock_image, mock_bbox):
    result = age_gender_model.predict(mock_image, mock_bbox)
    assert 0 <= result.age <= 120, f'Age should be between 0 and 120, got {result.age}'


def test_different_bbox_sizes(age_gender_model, mock_image):
    test_bboxes = [
        [50, 50, 150, 150],
        [100, 100, 300, 300],
        [50, 50, 400, 400],
    ]

    for bbox in test_bboxes:
        result = age_gender_model.predict(mock_image, bbox)
        assert result.gender in [0, 1], f'Failed for bbox {bbox}'
        assert 0 <= result.age <= 120, f'Age out of range for bbox {bbox}'


def test_different_image_sizes(age_gender_model, mock_bbox):
    test_sizes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]

    for size in test_sizes:
        mock_image = np.random.randint(0, 255, size, dtype=np.uint8)
        result = age_gender_model.predict(mock_image, mock_bbox)
        assert result.gender in [0, 1], f'Failed for image size {size}'
        assert 0 <= result.age <= 120, f'Age out of range for image size {size}'


def test_consistency(age_gender_model, mock_image, mock_bbox):
    result1 = age_gender_model.predict(mock_image, mock_bbox)
    result2 = age_gender_model.predict(mock_image, mock_bbox)

    assert result1.gender == result2.gender, 'Same input should produce same gender prediction'
    assert result1.age == result2.age, 'Same input should produce same age prediction'


def test_bbox_list_format(age_gender_model, mock_image):
    bbox_list = [100, 100, 300, 300]
    result = age_gender_model.predict(mock_image, bbox_list)
    assert result.gender in [0, 1], 'Should work with bbox as list'
    assert 0 <= result.age <= 120, 'Age should be in valid range'


def test_bbox_array_format(age_gender_model, mock_image):
    bbox_array = np.array([100, 100, 300, 300])
    result = age_gender_model.predict(mock_image, bbox_array)
    assert result.gender in [0, 1], 'Should work with bbox as numpy array'
    assert 0 <= result.age <= 120, 'Age should be in valid range'


def test_multiple_predictions(age_gender_model, mock_image):
    bboxes = [
        [50, 50, 150, 150],
        [200, 200, 350, 350],
        [400, 400, 550, 550],
    ]

    results = []
    for bbox in bboxes:
        result = age_gender_model.predict(mock_image, bbox)
        results.append(result)

    assert len(results) == 3, 'Should have 3 predictions'
    for result in results:
        assert result.gender in [0, 1]
        assert 0 <= result.age <= 120


def test_age_is_positive(age_gender_model, mock_image, mock_bbox):
    for _ in range(5):
        result = age_gender_model.predict(mock_image, mock_bbox)
        assert result.age >= 0, f'Age should be non-negative, got {result.age}'


def test_output_format_for_visualization(age_gender_model, mock_image, mock_bbox):
    result = age_gender_model.predict(mock_image, mock_bbox)
    text = f'{result.sex}, {result.age}y'
    assert isinstance(text, str), 'Should be able to format as string'
    assert 'Male' in text or 'Female' in text, 'Text should contain gender'
    assert 'y' in text, "Text should contain 'y' for years"


def test_attribute_result_fields(age_gender_model, mock_image, mock_bbox):
    """Test that AttributeResult has correct fields for AgeGender model."""
    result = age_gender_model.predict(mock_image, mock_bbox)

    # AgeGender should set gender and age
    assert result.gender is not None
    assert result.age is not None

    # AgeGender should NOT set race and age_group (FairFace only)
    assert result.race is None
    assert result.age_group is None

# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface import EDifFIQA, QualityResult
from uniface.constants import EDifFIQAWeights


def test_ediffiqa_initialization():
    """Test EDifFIQA initialization with default weights (T)."""
    estimator = EDifFIQA()
    assert estimator is not None
    assert estimator.input_size == (112, 112)


def test_ediffiqa_with_explicit_variant():
    """Test EDifFIQA initialization with explicit T variant."""
    estimator = EDifFIQA(model_name=EDifFIQAWeights.T)
    assert estimator is not None
    assert estimator.input_size == (112, 112)


def test_ediffiqa_with_providers():
    """Test that EDifFIQA accepts providers kwarg."""
    estimator = EDifFIQA(providers=['CPUExecutionProvider'])
    assert isinstance(estimator, EDifFIQA)


def test_ediffiqa_preprocess_shape():
    """Test preprocessing produces correct tensor shape and dtype."""
    estimator = EDifFIQA()
    aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    blob = estimator.preprocess(aligned)

    assert blob.dtype == np.float32
    assert blob.shape == (1, 3, 112, 112)


def test_ediffiqa_score_aligned_returns_quality_result():
    """Test scoring a pre-aligned crop returns a QualityResult."""
    estimator = EDifFIQA()
    aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    result = estimator.score_aligned(aligned)

    assert isinstance(result, QualityResult)
    assert isinstance(result.score, float)


def test_ediffiqa_predict_with_landmarks():
    """Test end-to-end prediction from full image + landmarks."""
    estimator = EDifFIQA()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # 5-point landmarks roughly centered in the image
    landmarks = np.array(
        [
            [260.0, 220.0],
            [380.0, 220.0],
            [320.0, 280.0],
            [275.0, 340.0],
            [365.0, 340.0],
        ],
        dtype=np.float32,
    )
    result = estimator.predict(image, landmarks)

    assert isinstance(result, QualityResult)
    assert isinstance(result.score, float)


def test_ediffiqa_callable():
    """Test that EDifFIQA is callable via __call__."""
    estimator = EDifFIQA()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array(
        [
            [260.0, 220.0],
            [380.0, 220.0],
            [320.0, 280.0],
            [275.0, 340.0],
            [365.0, 340.0],
        ],
        dtype=np.float32,
    )
    result = estimator(image, landmarks)

    assert isinstance(result, QualityResult)


def test_quality_result_repr():
    """Test QualityResult repr formatting."""
    result = QualityResult(score=0.7620)
    repr_str = repr(result)
    assert 'QualityResult' in repr_str
    assert '0.7620' in repr_str


def test_quality_result_frozen():
    """Test that QualityResult is immutable."""
    result = QualityResult(score=0.5)
    with pytest.raises(AttributeError):
        result.score = 0.9  # type: ignore[misc]

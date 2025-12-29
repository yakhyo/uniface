# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for BiSeNet face parsing model."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import ParsingWeights
from uniface.parsing import BiSeNet, create_face_parser


def test_bisenet_initialization():
    """Test BiSeNet initialization."""
    parser = BiSeNet()
    assert parser is not None
    assert parser.input_size == (512, 512)


def test_bisenet_with_different_models():
    """Test BiSeNet with different model weights."""
    parser_resnet18 = BiSeNet(model_name=ParsingWeights.RESNET18)
    parser_resnet34 = BiSeNet(model_name=ParsingWeights.RESNET34)

    assert parser_resnet18 is not None
    assert parser_resnet34 is not None


def test_bisenet_preprocess():
    """Test preprocessing."""
    parser = BiSeNet()

    # Create a dummy face image
    face_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Preprocess
    preprocessed = parser.preprocess(face_image)

    assert preprocessed.shape == (1, 3, 512, 512)
    assert preprocessed.dtype == np.float32


def test_bisenet_postprocess():
    """Test postprocessing."""
    parser = BiSeNet()

    # Create dummy model output (batch_size=1, num_classes=19, H=512, W=512)
    dummy_output = np.random.randn(1, 19, 512, 512).astype(np.float32)

    # Postprocess
    mask = parser.postprocess(dummy_output, original_size=(256, 256))

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert mask.min() >= 0
    assert mask.max() < 19  # 19 classes (0-18)


def test_bisenet_parse():
    """Test end-to-end parsing."""
    parser = BiSeNet()

    # Create a dummy face image
    face_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Parse
    mask = parser.parse(face_image)

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert mask.min() >= 0
    assert mask.max() < 19


def test_bisenet_callable():
    """Test that BiSeNet is callable."""
    parser = BiSeNet()
    face_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Should work as callable
    mask = parser(face_image)

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8


def test_create_face_parser_with_enum():
    """Test factory function with enum."""
    parser = create_face_parser(ParsingWeights.RESNET18)
    assert parser is not None
    assert isinstance(parser, BiSeNet)


def test_create_face_parser_with_string():
    """Test factory function with string."""
    parser = create_face_parser('parsing_resnet18')
    assert parser is not None
    assert isinstance(parser, BiSeNet)


def test_create_face_parser_invalid_model():
    """Test factory function with invalid model name."""
    with pytest.raises(ValueError, match='Unknown face parsing model'):
        create_face_parser('invalid_model')


def test_bisenet_different_input_sizes():
    """Test parsing with different input image sizes."""
    parser = BiSeNet()

    # Test with different sizes
    sizes = [(128, 128), (256, 256), (512, 512), (640, 480)]

    for h, w in sizes:
        face_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = parser.parse(face_image)

        assert mask.shape == (h, w), f'Failed for size {h}x{w}'
        assert mask.dtype == np.uint8

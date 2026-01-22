# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for face parsing models (BiSeNet and XSeg)."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.constants import ParsingWeights, XSegWeights
from uniface.parsing import BiSeNet, XSeg, create_face_parser


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


# XSeg Tests


def test_xseg_initialization():
    """Test XSeg initialization."""
    parser = XSeg()
    assert parser is not None
    assert parser.input_size == (256, 256)
    assert parser.align_size == 256
    assert parser.blur_sigma == 0


def test_xseg_with_custom_params():
    """Test XSeg with custom parameters."""
    parser = XSeg(align_size=512, blur_sigma=5)
    assert parser.align_size == 512
    assert parser.blur_sigma == 5


def test_xseg_preprocess():
    """Test XSeg preprocessing."""
    parser = XSeg()

    # Create a dummy aligned face crop
    face_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Preprocess
    preprocessed = parser.preprocess(face_crop)

    assert preprocessed.shape == (1, 256, 256, 3)  # NHWC format
    assert preprocessed.dtype == np.float32
    assert preprocessed.min() >= 0
    assert preprocessed.max() <= 1


def test_xseg_postprocess():
    """Test XSeg postprocessing."""
    parser = XSeg()

    # Create dummy model output (NHWC format)
    dummy_output = np.random.rand(1, 256, 256, 1).astype(np.float32)

    # Postprocess
    mask = parser.postprocess(dummy_output, crop_size=(256, 256))

    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert mask.min() >= 0
    assert mask.max() <= 1


def test_xseg_parse_aligned():
    """Test XSeg parse_aligned method."""
    parser = XSeg()

    # Create a dummy aligned face crop
    face_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Parse
    mask = parser.parse_aligned(face_crop)

    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert mask.min() >= 0
    assert mask.max() <= 1


def test_xseg_parse_with_landmarks():
    """Test XSeg parse method with landmarks."""
    parser = XSeg()

    # Create a dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create dummy 5-point landmarks
    landmarks = np.array(
        [
            [250, 200],  # left eye
            [390, 200],  # right eye
            [320, 280],  # nose
            [260, 350],  # left mouth
            [380, 350],  # right mouth
        ],
        dtype=np.float32,
    )

    # Parse
    mask = parser.parse(image, landmarks)

    assert mask.shape == (480, 640)
    assert mask.dtype == np.float32
    assert mask.min() >= 0
    assert mask.max() <= 1


def test_xseg_parse_invalid_landmarks():
    """Test XSeg parse with invalid landmarks shape."""
    parser = XSeg()
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Wrong shape
    invalid_landmarks = np.array([[0, 0], [1, 1], [2, 2]])

    with pytest.raises(ValueError, match='Landmarks must have shape'):
        parser.parse(image, invalid_landmarks)


def test_xseg_parse_with_inverse():
    """Test XSeg parse_with_inverse method."""
    parser = XSeg()

    # Create a dummy image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create dummy 5-point landmarks
    landmarks = np.array(
        [
            [250, 200],
            [390, 200],
            [320, 280],
            [260, 350],
            [380, 350],
        ],
        dtype=np.float32,
    )

    # Parse with inverse
    mask, face_crop, inverse_matrix = parser.parse_with_inverse(image, landmarks)

    assert mask.shape == (256, 256)
    assert face_crop.shape == (256, 256, 3)
    assert inverse_matrix.shape == (2, 3)


def test_create_face_parser_xseg_enum():
    """Test factory function with XSeg enum."""
    parser = create_face_parser(XSegWeights.DEFAULT)
    assert parser is not None
    assert isinstance(parser, XSeg)


def test_create_face_parser_xseg_string():
    """Test factory function with XSeg string."""
    parser = create_face_parser('xseg')
    assert parser is not None
    assert isinstance(parser, XSeg)

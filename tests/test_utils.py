# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for utility functions (compute_similarity, face_alignment, etc.)."""

from __future__ import annotations

import numpy as np
import pytest

from uniface import compute_similarity, face_alignment


@pytest.fixture
def mock_image():
    """
    Create a mock 640x640 BGR image.
    """
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_landmarks():
    """
    Create mock 5-point facial landmarks.
    Standard positions for a face roughly centered at (112/2, 112/2).
    """
    return np.array(
        [
            [38.2946, 51.6963],  # Left eye
            [73.5318, 51.5014],  # Right eye
            [56.0252, 71.7366],  # Nose
            [41.5493, 92.3655],  # Left mouth corner
            [70.7299, 92.2041],  # Right mouth corner
        ],
        dtype=np.float32,
    )


# compute_similarity tests
def test_compute_similarity_same_embedding():
    """
    Test that similarity of an embedding with itself is 1.0.
    """
    embedding = np.random.randn(1, 512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize

    similarity = compute_similarity(embedding, embedding)
    assert np.isclose(similarity, 1.0, atol=1e-5), f'Self-similarity should be 1.0, got {similarity}'


def test_compute_similarity_range():
    """
    Test that similarity is always in the range [-1, 1].
    """
    # Test with multiple random embeddings
    for _ in range(10):
        emb1 = np.random.randn(1, 512).astype(np.float32)
        emb2 = np.random.randn(1, 512).astype(np.float32)

        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        similarity = compute_similarity(emb1, emb2)
        assert -1.0 <= similarity <= 1.0, f'Similarity should be in [-1, 1], got {similarity}'


def test_compute_similarity_orthogonal():
    """
    Test that orthogonal embeddings have similarity close to 0.
    """
    # Create orthogonal embeddings
    emb1 = np.zeros((1, 512), dtype=np.float32)
    emb1[0, 0] = 1.0  # [1, 0, 0, ..., 0]

    emb2 = np.zeros((1, 512), dtype=np.float32)
    emb2[0, 1] = 1.0  # [0, 1, 0, ..., 0]

    similarity = compute_similarity(emb1, emb2)
    assert np.isclose(similarity, 0.0, atol=1e-5), f'Orthogonal embeddings should have similarity 0.0, got {similarity}'


def test_compute_similarity_opposite():
    """
    Test that opposite embeddings have similarity close to -1.
    """
    emb1 = np.ones((1, 512), dtype=np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)

    emb2 = -emb1  # Opposite direction

    similarity = compute_similarity(emb1, emb2)
    assert np.isclose(similarity, -1.0, atol=1e-5), f'Opposite embeddings should have similarity -1.0, got {similarity}'


def test_compute_similarity_symmetry():
    """
    Test that similarity(A, B) == similarity(B, A).
    """
    emb1 = np.random.randn(1, 512).astype(np.float32)
    emb2 = np.random.randn(1, 512).astype(np.float32)

    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    sim_12 = compute_similarity(emb1, emb2)
    sim_21 = compute_similarity(emb2, emb1)

    assert np.isclose(sim_12, sim_21), 'Similarity should be symmetric'


def test_compute_similarity_dtype():
    """
    Test that compute_similarity returns a float.
    """
    emb1 = np.random.randn(1, 512).astype(np.float32)
    emb2 = np.random.randn(1, 512).astype(np.float32)

    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    similarity = compute_similarity(emb1, emb2)
    assert isinstance(similarity, float | np.floating), f'Similarity should be float, got {type(similarity)}'


# face_alignment tests
def test_face_alignment_output_shape(mock_image, mock_landmarks):
    """
    Test that face_alignment produces output with the correct shape.
    """
    aligned, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))

    assert aligned.shape == (112, 112, 3), f'Expected shape (112, 112, 3), got {aligned.shape}'


def test_face_alignment_dtype(mock_image, mock_landmarks):
    """
    Test that aligned face has the correct data type.
    """
    aligned, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))

    assert aligned.dtype == np.uint8, f'Expected uint8, got {aligned.dtype}'


def test_face_alignment_different_sizes(mock_image, mock_landmarks):
    """
    Test face alignment with different output sizes.
    """
    # Only test sizes that are multiples of 112 or 128 as required by the function
    test_sizes = [(112, 112), (128, 128), (224, 224)]

    for size in test_sizes:
        aligned, _ = face_alignment(mock_image, mock_landmarks, image_size=size)
        assert aligned.shape == (*size, 3), f'Failed for size {size}'


def test_face_alignment_consistency(mock_image, mock_landmarks):
    """
    Test that the same input produces the same aligned face.
    """
    aligned1, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))
    aligned2, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))

    assert np.allclose(aligned1, aligned2), 'Same input should produce same aligned face'


def test_face_alignment_landmarks_as_list(mock_image):
    """
    Test that landmarks can be passed as a list of lists (converted to array).
    """
    landmarks_list = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ]

    # Convert list to numpy array before passing to face_alignment
    landmarks_array = np.array(landmarks_list, dtype=np.float32)
    aligned, _ = face_alignment(mock_image, landmarks_array, image_size=(112, 112))
    assert aligned.shape == (112, 112, 3), 'Should work with landmarks as array'


def test_face_alignment_value_range(mock_image, mock_landmarks):
    """
    Test that aligned face pixel values are in valid range [0, 255].
    """
    aligned, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))

    assert np.all(aligned >= 0), 'Pixel values should be >= 0'
    assert np.all(aligned <= 255), 'Pixel values should be <= 255'


def test_face_alignment_not_all_zeros(mock_image, mock_landmarks):
    """
    Test that aligned face is not all zeros (actual transformation occurred).
    """
    aligned, _ = face_alignment(mock_image, mock_landmarks, image_size=(112, 112))

    # At least some pixels should be non-zero
    assert np.any(aligned > 0), 'Aligned face should have some non-zero pixels'


def test_face_alignment_from_different_positions(mock_image):
    """
    Test alignment with landmarks at different positions in the image.
    """
    # Landmarks at different positions
    positions = [
        np.array(
            [[100, 100], [150, 100], [125, 130], [110, 150], [140, 150]],
            dtype=np.float32,
        ),
        np.array(
            [[300, 200], [350, 200], [325, 230], [310, 250], [340, 250]],
            dtype=np.float32,
        ),
        np.array(
            [[500, 400], [550, 400], [525, 430], [510, 450], [540, 450]],
            dtype=np.float32,
        ),
    ]

    for landmarks in positions:
        aligned, _ = face_alignment(mock_image, landmarks, image_size=(112, 112))
        assert aligned.shape == (112, 112, 3), f'Failed for landmarks at {landmarks[0]}'


def test_face_alignment_landmark_count(mock_image):
    """
    Test that face_alignment works specifically with 5-point landmarks.
    """
    # Standard 5-point landmarks
    landmarks_5pt = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    aligned, _ = face_alignment(mock_image, landmarks_5pt, image_size=(112, 112))
    assert aligned.shape == (112, 112, 3), 'Should work with 5-point landmarks'


def test_compute_similarity_with_recognition_embeddings():
    """
    Test compute_similarity with realistic embedding dimensions.
    """
    # Simulate ArcFace/MobileFace/SphereFace embeddings (512-dim)
    emb1 = np.random.randn(1, 512).astype(np.float32)
    emb2 = np.random.randn(1, 512).astype(np.float32)

    # Normalize (as done in get_normalized_embedding)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    similarity = compute_similarity(emb1, emb2)

    # Should be a valid similarity score
    assert -1.0 <= similarity <= 1.0
    assert isinstance(similarity, float | np.floating)

# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Tests for face recognition models (ArcFace, MobileFace, SphereFace)."""

from __future__ import annotations

import numpy as np
import pytest

from uniface.recognition import ArcFace, MobileFace, SphereFace


@pytest.fixture
def arcface_model():
    """
    Fixture to initialize the ArcFace model for testing.
    """
    return ArcFace()


@pytest.fixture
def mobileface_model():
    """
    Fixture to initialize the MobileFace model for testing.
    """
    return MobileFace()


@pytest.fixture
def sphereface_model():
    """
    Fixture to initialize the SphereFace model for testing.
    """
    return SphereFace()


@pytest.fixture
def mock_aligned_face():
    """
    Create a mock 112x112 aligned face image.
    """
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def mock_landmarks():
    """
    Create mock 5-point facial landmarks.
    """
    return np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )


# ArcFace Tests
def test_arcface_initialization(arcface_model):
    """
    Test that the ArcFace model initializes correctly.
    """
    assert arcface_model is not None, 'ArcFace model initialization failed.'


def test_arcface_embedding_shape(arcface_model, mock_aligned_face):
    """
    Test that ArcFace produces embeddings with the correct shape.
    """
    embedding = arcface_model.get_embedding(mock_aligned_face)

    # ArcFace typically produces 512-dimensional embeddings
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_arcface_normalized_embedding(arcface_model, mock_landmarks):
    """
    Test that normalized embeddings have unit length.
    """
    # Create a larger mock image for alignment
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = arcface_model.get_normalized_embedding(mock_image, mock_landmarks)

    # Check that embedding is normalized (L2 norm ≈ 1.0)
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


def test_arcface_embedding_dtype(arcface_model, mock_aligned_face):
    """
    Test that embeddings have the correct data type.
    """
    embedding = arcface_model.get_embedding(mock_aligned_face)
    assert embedding.dtype == np.float32, f'Expected float32, got {embedding.dtype}'


def test_arcface_consistency(arcface_model, mock_aligned_face):
    """
    Test that the same input produces the same embedding.
    """
    embedding1 = arcface_model.get_embedding(mock_aligned_face)
    embedding2 = arcface_model.get_embedding(mock_aligned_face)

    assert np.allclose(embedding1, embedding2), 'Same input should produce same embedding'


# MobileFace Tests
def test_mobileface_initialization(mobileface_model):
    """
    Test that the MobileFace model initializes correctly.
    """
    assert mobileface_model is not None, 'MobileFace model initialization failed.'


def test_mobileface_embedding_shape(mobileface_model, mock_aligned_face):
    """
    Test that MobileFace produces embeddings with the correct shape.
    """
    embedding = mobileface_model.get_embedding(mock_aligned_face)

    # MobileFace typically produces 512-dimensional embeddings
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_mobileface_normalized_embedding(mobileface_model, mock_landmarks):
    """
    Test that MobileFace normalized embeddings have unit length.
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = mobileface_model.get_normalized_embedding(mock_image, mock_landmarks)

    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


# SphereFace Tests
def test_sphereface_initialization(sphereface_model):
    """
    Test that the SphereFace model initializes correctly.
    """
    assert sphereface_model is not None, 'SphereFace model initialization failed.'


def test_sphereface_embedding_shape(sphereface_model, mock_aligned_face):
    """
    Test that SphereFace produces embeddings with the correct shape.
    """
    embedding = sphereface_model.get_embedding(mock_aligned_face)

    # SphereFace typically produces 512-dimensional embeddings
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_sphereface_normalized_embedding(sphereface_model, mock_landmarks):
    """
    Test that SphereFace normalized embeddings have unit length.
    """
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = sphereface_model.get_normalized_embedding(mock_image, mock_landmarks)

    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


# Cross-model comparison tests
def test_different_models_different_embeddings(arcface_model, mobileface_model, mock_aligned_face):
    """
    Test that different models produce different embeddings for the same input.
    """
    arcface_emb = arcface_model.get_embedding(mock_aligned_face)
    mobileface_emb = mobileface_model.get_embedding(mock_aligned_face)

    # Embeddings should be different (with high probability for random input)
    # We check that they're not identical
    assert not np.allclose(arcface_emb, mobileface_emb), 'Different models should produce different embeddings'


def test_embedding_similarity_computation(arcface_model, mock_aligned_face):
    """
    Test computing similarity between embeddings.
    """
    # Get two embeddings
    emb1 = arcface_model.get_embedding(mock_aligned_face)

    # Create a slightly different image
    mock_aligned_face2 = mock_aligned_face.copy()
    mock_aligned_face2[:10, :10] = 0  # Modify a small region
    emb2 = arcface_model.get_embedding(mock_aligned_face2)

    # Compute cosine similarity
    from uniface import compute_similarity

    similarity = compute_similarity(emb1, emb2)

    # Similarity should be between -1 and 1
    assert -1.0 <= similarity <= 1.0, f'Similarity should be in [-1, 1], got {similarity}'


def test_same_face_high_similarity(arcface_model, mock_aligned_face):
    """
    Test that the same face produces high similarity.
    """
    emb1 = arcface_model.get_embedding(mock_aligned_face)
    emb2 = arcface_model.get_embedding(mock_aligned_face)

    from uniface import compute_similarity

    similarity = compute_similarity(emb1, emb2)

    # Same image should have similarity close to 1.0
    assert similarity > 0.99, f'Same face should have similarity > 0.99, got {similarity}'

# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uniface.recognition import ArcFace, AuraFace, EdgeFace, MobileFace, SphereFace

SOURCE = Path(__file__).resolve().parent.parent / 'assets' / 'source'


@pytest.fixture
def arcface_model():
    """Fixture to initialize the ArcFace model for testing."""
    return ArcFace()


@pytest.fixture
def mobileface_model():
    """Fixture to initialize the MobileFace model for testing."""
    return MobileFace()


@pytest.fixture
def sphereface_model():
    """Fixture to initialize the SphereFace model for testing."""
    return SphereFace()


@pytest.fixture
def edgeface_model():
    """Fixture to initialize the EdgeFace model for testing."""
    return EdgeFace()


@pytest.fixture
def auraface_model():
    """Fixture to initialize the AuraFace model for testing."""
    return AuraFace()


@pytest.fixture
def mock_aligned_face():
    """Create a mock 112x112 aligned face image."""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def mock_landmarks():
    """Create mock 5-point facial landmarks."""
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
    """Test that the ArcFace model initializes correctly."""
    assert arcface_model is not None, 'ArcFace model initialization failed.'


def test_arcface_embedding_shape(arcface_model, mock_aligned_face):
    """Test that ArcFace produces embeddings with the correct shape."""
    embedding = arcface_model.get_embedding(mock_aligned_face)

    # ArcFace get_embedding returns raw ONNX output with batch dimension
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_arcface_normalized_embedding(arcface_model, mock_landmarks):
    """Test that normalized embeddings have unit length."""
    # Create a larger mock image for alignment
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = arcface_model.get_normalized_embedding(mock_image, mock_landmarks)

    # Check shape and normalization
    assert embedding.shape == (512,), f'Expected shape (512,), got {embedding.shape}'
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


def test_arcface_embedding_dtype(arcface_model, mock_aligned_face):
    """Test that embeddings have the correct data type."""
    embedding = arcface_model.get_embedding(mock_aligned_face)
    assert embedding.dtype == np.float32, f'Expected float32, got {embedding.dtype}'


def test_arcface_consistency(arcface_model, mock_aligned_face):
    """Test that the same input produces the same embedding."""
    embedding1 = arcface_model.get_embedding(mock_aligned_face)
    embedding2 = arcface_model.get_embedding(mock_aligned_face)

    assert np.allclose(embedding1, embedding2), 'Same input should produce same embedding'


# MobileFace Tests
def test_mobileface_initialization(mobileface_model):
    """Test that the MobileFace model initializes correctly."""
    assert mobileface_model is not None, 'MobileFace model initialization failed.'


def test_mobileface_embedding_shape(mobileface_model, mock_aligned_face):
    """Test that MobileFace produces embeddings with the correct shape."""
    embedding = mobileface_model.get_embedding(mock_aligned_face)

    # MobileFace get_embedding returns raw ONNX output with batch dimension
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_mobileface_normalized_embedding(mobileface_model, mock_landmarks):
    """Test that MobileFace normalized embeddings have unit length."""
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = mobileface_model.get_normalized_embedding(mock_image, mock_landmarks)

    assert embedding.shape == (512,), f'Expected shape (512,), got {embedding.shape}'
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


# SphereFace Tests
def test_sphereface_initialization(sphereface_model):
    """Test that the SphereFace model initializes correctly."""
    assert sphereface_model is not None, 'SphereFace model initialization failed.'


def test_sphereface_embedding_shape(sphereface_model, mock_aligned_face):
    """Test that SphereFace produces embeddings with the correct shape."""
    embedding = sphereface_model.get_embedding(mock_aligned_face)

    # SphereFace get_embedding returns raw ONNX output with batch dimension
    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_sphereface_normalized_embedding(sphereface_model, mock_landmarks):
    """Test that SphereFace normalized embeddings have unit length."""
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = sphereface_model.get_normalized_embedding(mock_image, mock_landmarks)

    assert embedding.shape == (512,), f'Expected shape (512,), got {embedding.shape}'
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


# EdgeFace Tests
def test_edgeface_initialization(edgeface_model):
    """Test that the EdgeFace model initializes correctly."""
    assert edgeface_model is not None, 'EdgeFace model initialization failed.'


def test_edgeface_embedding_shape(edgeface_model, mock_aligned_face):
    """Test that EdgeFace produces embeddings with the correct shape."""
    embedding = edgeface_model.get_embedding(mock_aligned_face)

    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_edgeface_normalized_embedding(edgeface_model, mock_landmarks):
    """Test that EdgeFace normalized embeddings have unit length."""
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = edgeface_model.get_normalized_embedding(mock_image, mock_landmarks)

    assert embedding.shape == (512,), f'Expected shape (512,), got {embedding.shape}'
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


def test_edgeface_embedding_dtype(edgeface_model, mock_aligned_face):
    """Test that EdgeFace embeddings have the correct data type."""
    embedding = edgeface_model.get_embedding(mock_aligned_face)
    assert embedding.dtype == np.float32, f'Expected float32, got {embedding.dtype}'


def test_edgeface_consistency(edgeface_model, mock_aligned_face):
    """Test that the same input produces the same EdgeFace embedding."""
    embedding1 = edgeface_model.get_embedding(mock_aligned_face)
    embedding2 = edgeface_model.get_embedding(mock_aligned_face)

    assert np.allclose(embedding1, embedding2), 'Same input should produce same embedding'


# AuraFace Tests
def test_auraface_initialization(auraface_model):
    """Test that the AuraFace model initializes correctly."""
    assert auraface_model is not None, 'AuraFace model initialization failed.'


def test_auraface_embedding_shape(auraface_model, mock_aligned_face):
    """Test that AuraFace produces embeddings with the correct shape."""
    embedding = auraface_model.get_embedding(mock_aligned_face)

    assert embedding.shape[1] == 512, f'Expected 512-dim embedding, got {embedding.shape[1]}'
    assert embedding.shape[0] == 1, 'Embedding should have batch dimension of 1'


def test_auraface_normalized_embedding(auraface_model, mock_landmarks):
    """Test that AuraFace normalized embeddings have unit length."""
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    embedding = auraface_model.get_normalized_embedding(mock_image, mock_landmarks)

    assert embedding.shape == (512,), f'Expected shape (512,), got {embedding.shape}'
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f'Normalized embedding should have norm 1.0, got {norm}'


def test_auraface_embedding_dtype(auraface_model, mock_aligned_face):
    """Test that AuraFace embeddings have the correct data type."""
    embedding = auraface_model.get_embedding(mock_aligned_face)
    assert embedding.dtype == np.float32, f'Expected float32, got {embedding.dtype}'


def test_auraface_consistency(auraface_model, mock_aligned_face):
    """Test that the same input produces the same AuraFace embedding."""
    embedding1 = auraface_model.get_embedding(mock_aligned_face)
    embedding2 = auraface_model.get_embedding(mock_aligned_face)

    assert np.allclose(embedding1, embedding2), 'Same input should produce same embedding'


def test_auraface_separates_identities(auraface_model):
    """Test that AuraFace scores one person's photos above a different person's."""
    import cv2

    from uniface import RetinaFace, compute_similarity

    detector = RetinaFace()
    embeddings = []
    for name in ('verify_now_2010', 'verify_now_2014', 'parse_face'):
        image = cv2.imread(str(SOURCE / f'{name}.jpg'))
        face = max(detector.detect(image), key=lambda f: f.confidence)
        embeddings.append(auraface_model.get_normalized_embedding(image, face.landmarks))

    same = compute_similarity(embeddings[0], embeddings[1], normalized=True)
    different = compute_similarity(embeddings[0], embeddings[2], normalized=True)

    assert same > 0.5, f'Same person should score above 0.5, got {same}'
    assert different < 0.2, f'Different people should score below 0.2, got {different}'


# Cross-model comparison tests
def test_different_models_different_embeddings(arcface_model, mobileface_model, mock_aligned_face):
    """Test that different models produce different embeddings for the same input."""
    arcface_emb = arcface_model.get_embedding(mock_aligned_face)
    mobileface_emb = mobileface_model.get_embedding(mock_aligned_face)

    # Embeddings should be different (with high probability for random input)
    # We check that they're not identical
    assert not np.allclose(arcface_emb, mobileface_emb), 'Different models should produce different embeddings'


def test_embedding_similarity_computation(arcface_model, mock_aligned_face):
    """Test computing similarity between embeddings."""
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
    """Test that the same face produces high similarity."""
    emb1 = arcface_model.get_embedding(mock_aligned_face)
    emb2 = arcface_model.get_embedding(mock_aligned_face)

    from uniface import compute_similarity

    similarity = compute_similarity(emb1, emb2)

    # Same image should have similarity close to 1.0
    assert similarity > 0.99, f'Same face should have similarity > 0.99, got {similarity}'

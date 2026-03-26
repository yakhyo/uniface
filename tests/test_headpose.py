# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import numpy as np
import pytest

from uniface import HeadPose, HeadPoseResult, create_head_pose_estimator
from uniface.headpose import BaseHeadPoseEstimator
from uniface.headpose.models import HeadPose as HeadPoseModel


def test_create_head_pose_estimator_default():
    """Test creating a head pose estimator with default parameters."""
    estimator = create_head_pose_estimator()
    assert isinstance(estimator, HeadPose), 'Should return HeadPose instance'


def test_create_head_pose_estimator_aliases():
    """Test that factory accepts all documented aliases."""
    for alias in ('headpose', 'head_pose', '6drepnet'):
        estimator = create_head_pose_estimator(alias)
        assert isinstance(estimator, HeadPose), f"Alias '{alias}' should return HeadPose"


def test_create_head_pose_estimator_invalid():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match='Unsupported head pose estimation method'):
        create_head_pose_estimator('invalid_method')


def test_head_pose_inference():
    """Test that HeadPose can run inference on a mock image."""
    estimator = HeadPose()
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = estimator.estimate(mock_image)

    assert isinstance(result, HeadPoseResult), 'Should return HeadPoseResult'
    assert isinstance(result.pitch, float), 'pitch should be float'
    assert isinstance(result.yaw, float), 'yaw should be float'
    assert isinstance(result.roll, float), 'roll should be float'


def test_head_pose_callable():
    """Test that HeadPose is callable via __call__."""
    estimator = HeadPose()
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = estimator(mock_image)

    assert isinstance(result, HeadPoseResult), '__call__ should return HeadPoseResult'


def test_head_pose_result_repr():
    """Test HeadPoseResult repr formatting."""
    result = HeadPoseResult(pitch=10.5, yaw=-20.3, roll=5.1)
    repr_str = repr(result)
    assert 'HeadPoseResult' in repr_str
    assert '10.5' in repr_str
    assert '-20.3' in repr_str
    assert '5.1' in repr_str


def test_head_pose_result_frozen():
    """Test that HeadPoseResult is immutable."""
    result = HeadPoseResult(pitch=1.0, yaw=2.0, roll=3.0)
    with pytest.raises(AttributeError):
        result.pitch = 99.0  # type: ignore[misc]


def test_rotation_matrix_to_euler_identity():
    """Test that identity rotation matrix gives zero angles."""
    identity = np.eye(3).reshape(1, 3, 3)
    euler = HeadPoseModel.rotation_matrix_to_euler(identity)

    assert euler.shape == (1, 3), 'Should return (1, 3) shaped array'
    np.testing.assert_allclose(euler[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_rotation_matrix_to_euler_90deg_yaw():
    """Test 90-degree yaw rotation."""
    angle = np.radians(90)
    R = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    ).reshape(1, 3, 3)
    euler = HeadPoseModel.rotation_matrix_to_euler(R)

    np.testing.assert_allclose(euler[0, 1], 90.0, atol=1e-3)


def test_rotation_matrix_to_euler_batch():
    """Test batch processing of rotation matrices."""
    batch = np.stack([np.eye(3), np.eye(3), np.eye(3)], axis=0)
    euler = HeadPoseModel.rotation_matrix_to_euler(batch)

    assert euler.shape == (3, 3), 'Batch of 3 should return (3, 3)'
    np.testing.assert_allclose(euler, 0.0, atol=1e-5)


def test_factory_returns_correct_type():
    """Test that factory function returns BaseHeadPoseEstimator subclass."""
    estimator = create_head_pose_estimator()
    assert isinstance(estimator, BaseHeadPoseEstimator), 'Should be BaseHeadPoseEstimator subclass'


def test_head_pose_with_providers():
    """Test that HeadPose accepts providers kwarg."""
    estimator = HeadPose(providers=['CPUExecutionProvider'])
    assert isinstance(estimator, HeadPose), 'Should create with explicit providers'

# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from uniface.common import validate_image
from uniface.constants import FaceMeshWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import FaceMeshResult

from .base import BaseLandmarker

if TYPE_CHECKING:
    from uniface.types import Face

__all__ = ['FaceMesh', 'roi_from_box', 'warp_roi']

# MediaPipe's detection_to_roi expands the detector box by 1.5x; as a per-side
# fraction of the box that is 0.25.
_DEFAULT_MARGIN = 0.25

#: An ROI is (center_x, center_y, side, angle_degrees) in full-image coordinates.
Roi = tuple[float, float, float, float]


def roi_from_box(
    bbox: np.ndarray,
    keypoints: np.ndarray | None = None,
    margin: float = _DEFAULT_MARGIN,
) -> Roi:
    """Build a Face Mesh ROI from a detector box, following MediaPipe's rule.

    MediaPipe's `detection_to_roi` produces a *square* region, the long side of the box
    scaled, rotated so the eye line is horizontal. Only rows 0 and 1 of `keypoints` are
    read: those are the eyes in both the 5-point alignment layout and BlazeFace's
    6-point layout, so landmarks from any UniFace detector work here.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2].
        keypoints: Optional landmarks with shape (K, 2), K >= 2, whose first two
            rows are the eyes. They set the ROI rotation (roll). Without them the
            ROI is axis-aligned, which degrades the mesh on tilted heads.
        margin: Fraction of the box side to expand by on each side. The default
            0.25 reproduces MediaPipe's 1.5x scale.

    Returns:
        A square, optionally rotated ROI as (center_x, center_y, side, angle).

    Raises:
        ValueError: If fewer than 2 keypoints are given.
    """
    x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox).ravel()[:4])
    side = (1.0 + 2.0 * margin) * max(x2 - x1, y2 - y1)

    angle = 0.0
    if keypoints is not None:
        points = np.asarray(keypoints, dtype=np.float64)
        if points.shape[0] < 2:
            raise ValueError(f'roi_from_box needs at least 2 keypoints (the eyes), got {points.shape[0]}')
        dx, dy = (float(v) for v in points[1] - points[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, side, angle


def warp_roi(image: np.ndarray, roi: Roi, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rotated square ROI directly at the model's input resolution.

    A single bilinear resampling straight to the model size, mirroring MediaPipe's
    ImageToTensorCalculator; warp-then-resize would interpolate twice and quantize the
    crop side to whole pixels. Regions outside the image are zero-padded.

    Args:
        image: Full BGR image with shape (H, W, 3).
        roi: The (center_x, center_y, side, angle) region to cut.
        size: Output edge length in pixels; the model's input resolution.

    Returns:
        A tuple of the (size, size, 3) crop and the inverse 2x3 affine mapping
        crop-pixel coordinates back to full-image coordinates.
    """
    center_x, center_y, side, angle = roi
    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, size / side)
    matrix[0, 2] += size / 2.0 - center_x
    matrix[1, 2] += size / 2.0 - center_y

    crop = cv2.warpAffine(image, matrix, (size, size))
    return crop, cv2.invertAffineTransform(matrix)


class FaceMesh(BaseLandmarker):
    """MediaPipe dense 3D face landmarks: 468 points, or 478 with irises.

    Paper: https://arxiv.org/abs/1907.06724
    Code: https://github.com/google-ai-edge/mediapipe

    Detector-agnostic. The crop follows MediaPipe's ROI recipe, a square region at 1.5x
    the detector box rotated so the eye line is horizontal, and the model runs once per
    face, matching MediaPipe's static-image flow (`static_image_mode=True`). Video-mode
    ROI-from-previous-frame tracking is not reproduced; `roi_from_box` and `warp_roi`
    are public so it can be built on top.

    Args:
        model_name (FaceMeshWeights): Which model to load. `V1_468` (the default) is the
            classic 468-point Face Mesh at 192x192; `V2_478` is MediaPipe's Face
            Landmarker at 256x256, which appends ten iris points to the same mesh for
            roughly 3x the compute.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Raises:
        ValueError: If the model weights are invalid or not found.
        RuntimeError: If the ONNX model fails to load or initialize.

    Example:
        >>> from uniface import SCRFD, FaceMesh
        >>> from uniface.constants import FaceMeshWeights
        >>>
        >>> detector, mesher = SCRFD(), FaceMesh()
        >>> faces = detector.detect(image)
        >>> results = mesher.predict(image, faces)
        >>> results[0].landmarks.shape
        (468, 3)
        >>>
        >>> # Or one face at a time, 2D only, like any other landmarker
        >>> mesher.get_landmarks(image, faces[0].bbox).shape
        (468, 2)
        >>>
        >>> # The 478-point model appends irises; everything else is unchanged
        >>> from uniface.landmark import IRIS_LEFT
        >>> iris_mesher = FaceMesh(model_name=FaceMeshWeights.V2_478)
        >>> iris_mesher.predict(image, faces)[0].landmarks[IRIS_LEFT].shape
        (5, 3)
    """

    def __init__(
        self,
        *,
        model_name: FaceMeshWeights = FaceMeshWeights.V1_468,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing FaceMesh with model={model_name}')
        self.providers = providers
        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX session and read the model's geometry.

        Landmark count and input resolution come from the graph rather than being
        hardcoded, so a variant with a different point count (e.g. MediaPipe's
        478-point iris model) needs no code change.

        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_size = int(input_meta.shape[2])

            outputs = self.session.get_outputs()
            self.output_names = [o.name for o in outputs]
            self._num_landmarks = int(outputs[0].shape[1])

            Logger.info(f'Model initialized with {self._num_landmarks} landmarks ({self.input_size}px input)')

        except Exception as e:
            Logger.error(f"Failed to load FaceMesh model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize FaceMesh model: {e}') from e

    @property
    def num_landmarks(self) -> int:
        """Number of landmarks this model predicts, read from the ONNX graph."""
        return self._num_landmarks

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop into a CHW float32 slice in [0, 1], RGB order.

        Args:
            crop: Face crop with shape (input_size, input_size, 3) in BGR.

        Returns:
            A (3, input_size, input_size) float32 array.
        """
        if crop.shape[:2] != (self.input_size, self.input_size):
            crop = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def postprocess(
        self,
        landmarks: np.ndarray,
        logits: np.ndarray,
        rois: list[Roi],
        inverses: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map raw crop-space predictions back to full-image coordinates.

        Args:
            landmarks: Raw (N, num_landmarks, 3) output in crop pixels.
            logits: Raw (N, 1) presence output. These are logits, not
                probabilities, and are sigmoided here.
            rois: The ROI used for each face, in order.
            inverses: The inverse affine for each face, from `warp_roi`.

        Returns:
            A tuple of (N, num_landmarks, 3) landmarks in full-image pixels and
            (N,) presence scores in [0, 1].
        """
        points = landmarks.astype(np.float64)
        for idx, (roi, inverse) in enumerate(zip(rois, inverses, strict=True)):
            points[idx, :, :2] = points[idx, :, :2] @ inverse[:, :2].T + inverse[:, 2]
            # Put z on the same pixel scale as x/y.
            points[idx, :, 2] *= roi[2] / self.input_size

        scores = 1.0 / (1.0 + np.exp(-logits.ravel().astype(np.float64)))
        return points.astype(np.float32), scores.astype(np.float32)

    def _run_rois(self, image: np.ndarray, rois: list[Roi]) -> tuple[np.ndarray, np.ndarray]:
        """Warp, batch, and run every ROI through a single session call."""
        blobs, inverses = [], []
        for roi in rois:
            crop, inverse = warp_roi(image, roi, self.input_size)
            blobs.append(self.preprocess(crop))
            inverses.append(inverse)

        landmarks, logits = self.session.run(self.output_names, {self.input_name: np.stack(blobs)})
        return self.postprocess(landmarks, logits, rois, inverses)

    def predict(
        self,
        image: np.ndarray,
        faces: list[Face] | None = None,
        *,
        bboxes: list[np.ndarray] | np.ndarray | None = None,
        keypoints: list[np.ndarray] | np.ndarray | None = None,
        margin: float = _DEFAULT_MARGIN,
    ) -> list[FaceMeshResult]:
        """Predict dense landmarks for every face in an image.

        Accepts either `Face` objects from any detector, or raw boxes. With `Face`
        objects the ROI is roll-normalized automatically from each face's landmarks.
        All faces run in one session call.

        Args:
            image: Full BGR image with shape (H, W, 3).
            faces: Detected faces, e.g. from `SCRFD().detect(image)`.
            bboxes: One [x1, y1, x2, y2] per face. Mutually exclusive with `faces`.
            keypoints: Optional landmarks per face, shape (K, 2) each and aligned
                with `bboxes`, whose first two rows are the eyes.
            margin: Fraction of the box side to expand each crop by on each side.

        Returns:
            One `FaceMeshResult` per input face, in input order.

        Raises:
            ValueError: If the image is empty, not 3-channel BGR, or not uint8; if
                both or neither of `faces` and `bboxes` are given; or if `keypoints`
                and `bboxes` have different lengths.

        Example:
            >>> results = mesher.predict(image, detector.detect(image))
            >>> results = mesher.predict(image, bboxes=[[10, 20, 110, 140]])
        """
        # `preprocess` rescales by 255, so a float [0, 1] image would be divided twice
        # and yield garbage landmarks that `get_landmarks` returns without a score.
        validate_image(image)

        if (faces is None) == (bboxes is None):
            raise ValueError('Provide either faces or bboxes, not both')

        if faces is not None:
            bboxes = [f.bbox for f in faces]
            keypoints = [f.landmarks for f in faces]
        elif keypoints is not None and len(keypoints) != len(bboxes):
            raise ValueError(f'Got {len(bboxes)} boxes but {len(keypoints)} keypoint sets')

        if len(bboxes) == 0:
            return []

        rois = [
            roi_from_box(bbox, None if keypoints is None else keypoints[idx], margin) for idx, bbox in enumerate(bboxes)
        ]
        landmarks, scores = self._run_rois(image, rois)
        return [FaceMeshResult(landmarks=landmarks[i], score=float(scores[i])) for i in range(len(rois))]

    def get_landmarks(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict dense landmarks for a single face, without depth.

        Satisfies the `BaseLandmarker` interface, so this is a drop-in for `Landmark106`
        and `PIPNet`. Use `predict` when you need depth, the presence score, or more
        than one face at a time.

        Args:
            image: Full BGR image in BGR format.
            bbox: Face bounding box [x1, y1, x2, y2].
            keypoints: Optional landmarks whose first two rows are the eyes,
                used to roll-normalize the crop.

        Returns:
            Landmark points with shape (num_landmarks, 2), float32, in the
            original image's pixel coordinates.
        """
        results = self.predict(image, bboxes=[bbox], keypoints=None if keypoints is None else [keypoints])
        return results[0].points_2d

    def __call__(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Callable shortcut for `get_landmarks`.

        Overrides `BaseLandmarker.__call__`, which forwards only (image, bbox) and
        would silently drop the roll normalization.
        """
        return self.get_landmarks(image, bbox, keypoints)

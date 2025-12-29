# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

__all__ = ['BlurFace', 'EllipticalBlur']


def _gaussian_blur(region: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """Apply Gaussian blur to a region."""
    h, w = region.shape[:2]
    kernel_size = max(3, int((min(h, w) / 7) * strength)) | 1
    return cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)


def _median_blur(region: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """Apply median blur to a region."""
    h, w = region.shape[:2]
    kernel_size = max(3, int((min(h, w) / 7) * strength)) | 1
    return cv2.medianBlur(region, kernel_size)


def _pixelate_blur(region: np.ndarray, blocks: int = 10) -> np.ndarray:
    """Apply pixelation to a region."""
    h, w = region.shape[:2]
    temp_h, temp_w = max(1, h // blocks), max(1, w // blocks)
    temp = cv2.resize(region, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def _blackout_blur(region: np.ndarray, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Replace region with solid color."""
    return np.full_like(region, color)


class EllipticalBlur:
    """Elliptical blur with soft, feathered edges.

    This blur applies Gaussian blur within an elliptical mask that follows
    the natural oval shape of faces, requiring full image context for proper blending.

    Args:
        blur_strength (float): Blur intensity multiplier. Defaults to 3.0.
        margin (int): Extra pixels to extend ellipse beyond bbox. Defaults to 20.
    """

    def __init__(self, blur_strength: float = 3.0, margin: int = 20):
        self.blur_strength = blur_strength
        self.margin = margin

    def __call__(
        self,
        image: np.ndarray,
        bboxes: list[tuple | list],
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()

        h, w = image.shape[:2]

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            axes_x = (x2 - x1) // 2 + self.margin
            axes_y = (y2 - y1) // 2 + self.margin

            # Create soft elliptical mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0) / 255.0
            mask = mask[:, :, np.newaxis]

            kernel_size = max(3, int((min(axes_y, axes_x) * 2 / 7) * self.blur_strength)) | 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            image = (blurred * mask + image * (1 - mask)).astype(np.uint8)

        return image


class BlurFace:
    """Face blurring with multiple anonymization methods.

    Args:
        method (str): Blur method - 'gaussian', 'pixelate', 'blackout', 'elliptical', or 'median'.
            Defaults to 'pixelate'.
        blur_strength (float): Intensity for gaussian/elliptical/median. Defaults to 3.0.
        pixel_blocks (int): Block count for pixelate. Defaults to 10.
        color (Tuple[int, int, int]): Fill color (BGR) for blackout. Defaults to (0, 0, 0).
        margin (int): Edge margin for elliptical. Defaults to 20.

    Example:
        >>> blurrer = BlurFace(method='pixelate')
        >>> anonymized = blurrer.anonymize(image, faces)
    """

    VALID_METHODS: ClassVar[set[str]] = {'gaussian', 'pixelate', 'blackout', 'elliptical', 'median'}

    def __init__(
        self,
        method: str = 'pixelate',
        blur_strength: float = 3.0,
        pixel_blocks: int = 15,
        color: tuple[int, int, int] = (0, 0, 0),
        margin: int = 20,
    ):
        self.method = method.lower()
        self._blur_strength = blur_strength
        self._pixel_blocks = pixel_blocks
        self._color = color
        self._margin = margin

        if self.method not in self.VALID_METHODS:
            raise ValueError(f"Invalid blur method: '{method}'. Choose from: {sorted(self.VALID_METHODS)}")

        if self.method == 'elliptical':
            self._elliptical = EllipticalBlur(blur_strength, margin)

    def _blur_region(self, region: np.ndarray) -> np.ndarray:
        """Apply blur to a single region based on the configured method."""
        if self.method == 'gaussian':
            return _gaussian_blur(region, self._blur_strength)
        elif self.method == 'median':
            return _median_blur(region, self._blur_strength)
        elif self.method == 'pixelate':
            return _pixelate_blur(region, self._pixel_blocks)
        elif self.method == 'blackout':
            return _blackout_blur(region, self._color)
        return region  # Fallback (should not reach here)

    def anonymize(
        self,
        image: np.ndarray,
        faces: list,
        inplace: bool = False,
    ) -> np.ndarray:
        """Anonymize faces in an image.

        Args:
            image (np.ndarray): Input image (BGR format).
            faces (List[Dict]): Face detections with 'bbox' key containing [x1, y1, x2, y2].
            inplace (bool): Modify image in-place if True. Defaults to False.

        Returns:
            np.ndarray: Image with anonymized faces.
        """
        if not faces:
            return image if inplace else image.copy()

        bboxes = [face.bbox for face in faces]
        return self.blur_regions(image, bboxes, inplace)

    def blur_regions(
        self,
        image: np.ndarray,
        bboxes: list[tuple | list],
        inplace: bool = False,
    ) -> np.ndarray:
        """Blur specific rectangular regions in an image.

        Args:
            image (np.ndarray): Input image (BGR format).
            bboxes (List): Bounding boxes as [x1, y1, x2, y2].
            inplace (bool): Modify image in-place if True. Defaults to False.

        Returns:
            np.ndarray: Image with blurred regions.
        """
        if not bboxes:
            return image if inplace else image.copy()

        if self.method == 'elliptical':
            return self._elliptical(image, bboxes, inplace)

        if not inplace:
            image = image.copy()

        h, w = image.shape[:2]

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                image[y1:y2, x1:x2] = self._blur_region(image[y1:y2, x1:x2])

        return image

    def __repr__(self) -> str:
        return f"BlurFace(method='{self.method}')"

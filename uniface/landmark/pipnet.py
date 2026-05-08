# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


from __future__ import annotations

import cv2
import numpy as np

from uniface.constants import PIPNetWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session

from ._meanface import get_meanface_info
from .base import BaseLandmarker

__all__ = ['PIPNet']

# PIPNet's upstream training preprocessing pads the bbox asymmetrically:
# +10% on the left, right, and bottom and -10% on the top before cropping.
_BBOX_PAD_RATIO = 0.1


class PIPNet(BaseLandmarker):
    """PIPNet facial landmark detector (98 or 68 points).

    PIPNet (Pixel-in-Pixel Net) detects landmarks via a heatmap classification
    head plus per-pixel offset and neighbor regression heads. The neighbor
    predictions are gathered through a reverse-index table built from a
    pre-trained meanface and then averaged with each landmark's own
    prediction for sub-pixel accuracy.

    Both the WFLW (98 points) and 300W+CelebA (68 points) variants share the
    same ResNet-18 backbone and 256x256 input. The number of landmarks is
    inferred from the ONNX output shape and the corresponding meanface table
    is selected automatically.

    Args:
        model_name (PIPNetWeights): Which PIPNet ONNX model to load.
            Defaults to ``PIPNetWeights.WFLW_98``.
        providers (list[str] | None): ONNX Runtime execution providers. If None,
            auto-detects the best available provider.

    Example:
        >>> from uniface.landmark import PIPNet
        >>> from uniface.constants import PIPNetWeights
        >>>
        >>> landmarker = PIPNet()  # WFLW_98 by default
        >>> landmarks = landmarker.get_landmarks(image, bbox)
        >>> print(landmarks.shape)
        (98, 2)
        >>>
        >>> landmarker_68 = PIPNet(model_name=PIPNetWeights.DW300_CELEBA_68)
        >>> landmarks_68 = landmarker_68.get_landmarks(image, bbox)
        >>> print(landmarks_68.shape)
        (68, 2)
    """

    def __init__(
        self,
        model_name: PIPNetWeights = PIPNetWeights.WFLW_98,
        providers: list[str] | None = None,
    ) -> None:
        Logger.info(f'Initializing PIPNet with model={model_name}')
        self.providers = providers

        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Number of meanface neighbors used at training time.
        self.num_neighbors = 10

        self.model_path = verify_model_weights(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX model and precompute the meanface tables.

        Raises:
            RuntimeError: If the model fails to load or initialize.
            ValueError: If the model output channel count does not match a
                supported meanface table (expected 68 or 98 landmarks).
        """
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)

            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            _, _, self.input_h, self.input_w = input_meta.shape

            outputs = self.session.get_outputs()
            self.output_names = [o.name for o in outputs]

            cls_shape = outputs[0].shape  # (1, num_lms, feat_h, feat_w)
            self.num_lms = int(cls_shape[1])
            self.feat_h = int(cls_shape[2]) if isinstance(cls_shape[2], int) else self.input_h // 32
            self.feat_w = int(cls_shape[3]) if isinstance(cls_shape[3], int) else self.input_w // 32
            self.net_stride = self.input_h // self.feat_h

            self._reverse_index1, self._reverse_index2, self._max_len = get_meanface_info(
                self.num_lms, self.num_neighbors
            )

            Logger.info(f'Model initialized with {self.num_lms} landmarks ({self.input_h}x{self.input_w} input)')

        except ValueError:
            raise
        except Exception as e:
            Logger.error(f"Failed to load PIPNet model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f'Failed to initialize PIPNet model: {e}') from e

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the face region and produce the network input blob.

        The crop follows the upstream PIPNet convention: pad ``+10%`` on the
        left, right, and bottom of the bbox and ``-10%`` on the top, then
        clamp to the image bounds. The crop is resized to the model's input
        resolution, BGR->RGB converted, and ImageNet-normalized.

        Args:
            image (np.ndarray): Full source image in BGR format, ``(H, W, 3)``.
            bbox (np.ndarray): Face bounding box ``[x1, y1, x2, y2]``.

        Returns:
            Tuple of:
                - The preprocessed ``(1, 3, H, W)`` float32 blob.
                - The crop metadata ``(x1, y1, crop_w, crop_h)`` used to
                  rescale predictions back to original image coordinates.
        """
        crop, crop_meta = self._crop_face(image, bbox)
        resized = cv2.resize(crop, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.input_mean) / self.input_std
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32), crop_meta

    def _crop_face(self, image: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        det_w = x2 - x1 + 1
        det_h = y2 - y1 + 1

        # Asymmetric: +10% left/right/bottom, -10% top.
        x1 -= int(det_w * _BBOX_PAD_RATIO)
        y1 += int(det_h * _BBOX_PAD_RATIO)
        x2 += int(det_w * _BBOX_PAD_RATIO)
        y2 += int(det_h * _BBOX_PAD_RATIO)

        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), img_w - 1)
        y2 = min(int(y2), img_h - 1)
        crop_w = x2 - x1 + 1
        crop_h = y2 - y1 + 1

        crop = image[y1 : y2 + 1, x1 : x2 + 1, :]
        return crop, (x1, y1, crop_w, crop_h)

    def postprocess(
        self,
        outputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        crop_meta: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Decode raw network outputs into original-image landmark coordinates.

        Combines each landmark's own (cls, offset) prediction with the
        predictions made about it by its ``num_nb`` nearest meanface neighbors,
        then maps the normalized result back to the original image using the
        crop metadata.

        Args:
            outputs (tuple): The five raw ONNX outputs in order
                ``(cls_map, offset_x, offset_y, nb_x, nb_y)``.
            crop_meta (tuple): The ``(x1, y1, crop_w, crop_h)`` returned by
                :meth:`preprocess`.

        Returns:
            np.ndarray: ``(num_lms, 2)`` float32 landmarks in original image space.
        """
        cls_map, offset_x, offset_y, nb_x, nb_y = outputs
        n = self.num_lms
        nb = self.num_neighbors
        h = self.feat_h
        w = self.feat_w

        cls_flat = cls_map.reshape(n, h * w)
        max_ids = np.argmax(cls_flat, axis=1)
        cols = (max_ids % w).astype(np.float32)
        rows = (max_ids // w).astype(np.float32)

        off_x_flat = offset_x.reshape(n, h * w)
        off_y_flat = offset_y.reshape(n, h * w)
        own_x = np.take_along_axis(off_x_flat, max_ids[:, None], axis=1).squeeze(1)
        own_y = np.take_along_axis(off_y_flat, max_ids[:, None], axis=1).squeeze(1)

        # Neighbor channels are ordered (num_nb * num_lms); reshape so axis 1 is neighbor index.
        nb_x_flat = nb_x.reshape(n, nb, h * w)
        nb_y_flat = nb_y.reshape(n, nb, h * w)
        nb_ids = np.broadcast_to(max_ids[:, None, None], (n, nb, 1))
        nb_own_x = np.take_along_axis(nb_x_flat, nb_ids, axis=2).squeeze(2)
        nb_own_y = np.take_along_axis(nb_y_flat, nb_ids, axis=2).squeeze(2)

        scale_x = self.input_w / self.net_stride
        scale_y = self.input_h / self.net_stride
        pred_x = (cols + own_x) / scale_x
        pred_y = (rows + own_y) / scale_y

        nb_pred_x = (cols[:, None] + nb_own_x) / scale_x
        nb_pred_y = (rows[:, None] + nb_own_y) / scale_y

        # Reverse gather: collect predictions about landmark i made by its neighbors,
        # then average with landmark i's own prediction.
        rev_x = nb_pred_x.reshape(-1)[self._reverse_index1 * nb + self._reverse_index2].reshape(n, self._max_len)
        rev_y = nb_pred_y.reshape(-1)[self._reverse_index1 * nb + self._reverse_index2].reshape(n, self._max_len)

        merged_x = np.mean(np.concatenate([pred_x[:, None], rev_x], axis=1), axis=1)
        merged_y = np.mean(np.concatenate([pred_y[:, None], rev_y], axis=1), axis=1)

        x1, y1, crop_w, crop_h = crop_meta
        merged_x = merged_x * crop_w + x1
        merged_y = merged_y * crop_h + y1

        return np.stack([merged_x, merged_y], axis=1).astype(np.float32)

    def get_landmarks(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Predict facial landmarks for the given face bounding box.

        Args:
            image (np.ndarray): Full source image in BGR format.
            bbox (np.ndarray): Face bounding box ``[x1, y1, x2, y2]``.

        Returns:
            np.ndarray: Landmark points as a ``(num_lms, 2)`` float32 array
                in the original image's pixel coordinates.
        """
        blob, crop_meta = self.preprocess(image, bbox)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, crop_meta)

# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo


import numpy as np

from uniface.attribute.base import BaseAttribute
from uniface.common import letterbox_resize
from uniface.constants import FaceAttribNetWeights
from uniface.log import Logger
from uniface.model_store import verify_model_weights
from uniface.onnx_utils import create_onnx_session
from uniface.types import Face, FaceStateResult

__all__ = ['FaceAttribNet']


class FaceAttribNet(BaseAttribute):
    """FaceAttribNet face state prediction model using ONNX Runtime.

    This class inherits from the `BaseAttribute` base class and implements the
    functionality for predicting five independent binary face attributes from
    a face crop: left/right eye openness, eyeglasses, face mask, and
    sunglasses. It requires a bounding box to locate the face.

    Each attribute comes from its own binary classifier head, so the five
    probabilities do not sum to 1 and several can be high at once (a face can
    wear both sunglasses and a mask). Threshold each one separately.

    The model is Qualcomm's "Facial-Attribute-Detection" (FaceAttribNet),
    mean/std normalization is baked into the ONNX graph itself, so
    preprocessing only scales pixel values to [0, 1].
    Original model: https://github.com/qualcomm/ai-hub-models/tree/main/src/qai_hub_models/models/face_attrib_net
    ONNX export and inference: https://github.com/yakhyo/face-attribute

    Args:
        model_name (FaceAttribNetWeights): The enum specifying the model weights to load.
            Defaults to `FaceAttribNetWeights.DEFAULT`.
        input_size (tuple[int, int] | None): Input size (height, width).
            If None, defaults to (128, 128). Defaults to None.
        margin (float): Fraction of box size to expand the face crop by on each
            side before inference. Defaults to 0.0.
        providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
            the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

    Raises:
        ValueError: If the model weights are invalid or not found, or `input_size` is not square.
        RuntimeError: If the ONNX model fails to load or initialize.
    """

    def __init__(
        self,
        *,
        model_name: FaceAttribNetWeights = FaceAttribNetWeights.DEFAULT,
        input_size: tuple[int, int] | None = None,
        margin: float = 0.0,
        providers: list[str] | None = None,
    ) -> None:
        """Initializes the FaceAttribNet prediction model.

        Args:
            model_name (FaceAttribNetWeights): The enum specifying the model weights to load.
            input_size (tuple[int, int] | None): Input size (height, width). Must be square:
                preprocessing letterboxes into a square canvas. If None, defaults to (128, 128).
            margin (float): Fraction of box size to expand the face crop by on each side.
            providers (list[str] | None): ONNX Runtime execution providers. If None, auto-detects
                the best available provider. Example: ['CPUExecutionProvider'] to force CPU.

        Raises:
            ValueError: If `input_size` is not square.
        """
        Logger.info(f'Initializing FaceAttribNet with model={model_name.name}')
        self.model_path = verify_model_weights(model_name)
        # Normalized to a tuple so a [128, 128] list still compares equal to the ONNX metadata.
        self.input_size = tuple(input_size) if input_size is not None else (128, 128)
        # letterbox_resize pads into a square canvas, so a non-square request cannot be honored.
        if self.input_size[0] != self.input_size[1]:
            raise ValueError(f'input_size must be square, got {self.input_size}')
        self.margin = margin
        self.providers = providers
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initializes the ONNX model and creates an inference session."""
        try:
            self.session = create_onnx_session(self.model_path, providers=self.providers)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name

            # Warn when a custom input_size disagrees with the model metadata
            model_input_size = tuple(input_meta.shape[2:4])  # (height, width)
            if all(isinstance(v, int) for v in model_input_size) and self.input_size != model_input_size:
                Logger.warning(
                    f'Using custom input_size {self.input_size}, '
                    f'but model expects {model_input_size}. This may affect accuracy.'
                )

            self.output_names = [output.name for output in self.session.get_outputs()]
            Logger.info(f'Successfully initialized FaceAttribNet model with input size {self.input_size}')
        except Exception as e:
            Logger.error(
                f"Failed to load FaceAttribNet model from '{self.model_path}'",
                exc_info=True,
            )
            raise RuntimeError(f'Failed to initialize FaceAttribNet model: {e}') from e

    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray | None = None) -> np.ndarray:
        """Preprocesses the face image for inference.

        Crops the face (optionally expanded by `margin`), letterboxes it to
        the model input size with centered zero padding, and scales pixel
        values to [0, 1]. No mean/std normalization is applied here: it is
        baked into the model graph.

        Args:
            image (np.ndarray): The input image in BGR format.
            bbox (list | np.ndarray | None): Face bounding box [x1, y1, x2, y2].
                If None, uses the entire image.

        Returns:
            np.ndarray: The preprocessed image blob ready for inference.

        Raises:
            ValueError: If the bounding box does not overlap the image.
        """
        if bbox is not None:
            height, width = image.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox)[:4])

            if self.margin:
                dw, dh = (x2 - x1) * self.margin, (y2 - y1) * self.margin
                x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh

            x1, y1 = max(0, round(x1)), max(0, round(y1))
            x2, y2 = min(width, round(x2)), min(height, round(y2))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Bounding box {bbox[:4]} does not overlap the {width}x{height} image')

            image = image[y1:y2, x1:x2]

        blob, _, _ = letterbox_resize(image, self.input_size[0], fill_value=0)
        return blob

    def postprocess(self, prediction: np.ndarray) -> FaceStateResult:
        """Processes the raw model output into per-attribute probabilities.

        Args:
            prediction (np.ndarray): Raw model output with shape (1, 5).

        Returns:
            FaceStateResult: Five independent probabilities in [0, 1].
        """
        left_eye_open, right_eye_open, eyeglasses, mask, sunglasses = np.squeeze(prediction).astype(float).tolist()
        return FaceStateResult(
            left_eye_open=left_eye_open,
            right_eye_open=right_eye_open,
            eyeglasses=eyeglasses,
            mask=mask,
            sunglasses=sunglasses,
        )

    def predict(self, image: np.ndarray, face: Face) -> FaceStateResult:
        """Predict face states and enrich the Face in-place.

        Args:
            image: The full input image in BGR format.
            face: Detected face; `face.bbox` is used for cropping.

        Returns:
            `FaceStateResult` with the five attribute probabilities.
        """
        input_blob = self.preprocess(image, face.bbox)
        outputs = self.session.run(self.output_names, {self.input_name: input_blob})
        result = self.postprocess(outputs[0])

        face.left_eye_open = result.left_eye_open
        face.right_eye_open = result.right_eye_open
        face.eyeglasses = result.eyeglasses
        face.mask = result.mask
        face.sunglasses = result.sunglasses
        return result

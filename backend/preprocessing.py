"""Eye crop preprocessing and normalization for DrishtiDrive AI.

Converts OpenCV BGR eye crops to RGB, resizes to target dimensions (96x96),
converts pixel values to [0.0, 1.0] tensors, applies ImageNet normalization,
and validates tensor shape and value ranges before inference.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from backend.config import PreprocessingConfig


class PreprocessingError(Exception):
    """Raised when an eye crop cannot be validated or preprocessed."""


class EyePreprocessor:
    """Handles crop normalization and tensor conversion matching training pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initialize preprocessor with target size and normalization stats."""
        self.config = config or PreprocessingConfig()
        self.target_size = (self.config.image_size, self.config.image_size)

        self.normalize = transforms.Normalize(
            mean=list(self.config.mean),
            std=list(self.config.std),
        )
        self.torchvision_pipeline = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Transform single OpenCV BGR crop into normalized (1, C, H, W) float32 tensor."""
        self.validate_crop(crop)

        # Convert OpenCV BGR to RGB
        if len(crop.shape) == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        tensor = self.torchvision_pipeline(pil_img)

        # Add batch dimension: (1, 3, H, W)
        tensor = tensor.unsqueeze(0)
        self.validate_tensor(tensor, expected_batch=1)
        return tensor

    def preprocess_crops_batch(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Batch-preprocess multiple eye crops into a single (B, C, H, W) tensor."""
        if not crops:
            raise PreprocessingError("Crops list cannot be empty for batch preprocessing.")

        tensors = [self.preprocess_crop(crop) for crop in crops]
        batch = torch.cat(tensors, dim=0)
        self.validate_tensor(batch, expected_batch=len(crops))
        return batch

    def validate_crop(self, crop: np.ndarray) -> None:
        """Validate input crop ndarray integrity, shape, and minimum size."""
        if crop is None or not isinstance(crop, np.ndarray):
            raise PreprocessingError("Invalid crop: Input must be a non-null numpy ndarray.")

        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            raise PreprocessingError(
                f"Crop dimensions too small: shape={crop.shape}. Minimum size is 8x8."
            )

    def validate_tensor(self, tensor: torch.Tensor, expected_batch: Optional[int] = None) -> None:
        """Verify tensor dtype, dimensions, and numerical validity."""
        if not isinstance(tensor, torch.Tensor):
            raise PreprocessingError("Preprocessed output must be a torch.Tensor.")

        if tensor.dtype != torch.float32:
            raise PreprocessingError(f"Unexpected tensor dtype: {tensor.dtype}, expected float32.")

        if tensor.ndim != 4:
            raise PreprocessingError(f"Expected 4D tensor (B, C, H, W), got shape: {tensor.shape}.")

        if expected_batch is not None and tensor.shape[0] != expected_batch:
            raise PreprocessingError(
                f"Batch size mismatch: expected {expected_batch}, got {tensor.shape[0]}."
            )

        if tensor.shape[1] != self.config.channels:
            raise PreprocessingError(
                f"Channel mismatch: expected {self.config.channels}, got {tensor.shape[1]}."
            )

        if tensor.shape[2:] != self.target_size:
            raise PreprocessingError(
                f"Spatial size mismatch: expected {self.target_size}, got {tensor.shape[2:]}."
            )

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise PreprocessingError("Preprocessed tensor contains NaN or Inf values.")

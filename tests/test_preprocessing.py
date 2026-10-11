"""Unit tests for eye crop preprocessing consistency with training pipeline."""
from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms

from backend.config import PreprocessingConfig
from backend.preprocessing import EyePreprocessor, PreprocessingError
from backend.training.dataset import get_transforms


def test_preprocessing_matches_training_pipeline() -> None:
    """Verify inference preprocessing produces identical tensor to training eval_transform."""
    preprocessor = EyePreprocessor()
    _, eval_transform = get_transforms(image_size=96)

    # Generate synthetic RGB test pattern
    np.random.seed(42)
    rgb_arr = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    bgr_arr = rgb_arr[:, :, ::-1]  # OpenCV BGR format

    # Method 1: Via EyePreprocessor (BGR input)
    tensor_infer = preprocessor.preprocess_crop(bgr_arr)

    # Method 2: Via training eval_transform (PIL Image input)
    pil_img = Image.fromarray(rgb_arr)
    tensor_train = eval_transform(pil_img).unsqueeze(0)

    # Verify tensor shape, dtype, and exact numerical match
    assert tensor_infer.shape == (1, 3, 96, 96)
    assert tensor_train.shape == (1, 3, 96, 96)
    assert tensor_infer.dtype == torch.float32
    assert torch.allclose(tensor_infer, tensor_train, atol=1e-5), "Inference preprocessing does not match training!"


def test_batch_preprocessing() -> None:
    """Verify batch preprocessing stacks multiple crops correctly."""
    preprocessor = EyePreprocessor()
    crop1 = np.full((50, 50, 3), 120, dtype=np.uint8)
    crop2 = np.full((60, 60, 3), 200, dtype=np.uint8)

    batch_tensor = preprocessor.preprocess_crops_batch([crop1, crop2])
    assert batch_tensor.shape == (2, 3, 96, 96)
    assert batch_tensor.dtype == torch.float32


def test_invalid_crop_validation() -> None:
    """Verify preprocessor rejects null, degenerate, or tiny crops."""
    preprocessor = EyePreprocessor()

    with pytest.raises(PreprocessingError):
        preprocessor.preprocess_crop(None)  # type: ignore

    with pytest.raises(PreprocessingError):
        preprocessor.preprocess_crop(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(PreprocessingError):
        preprocessor.preprocess_crop(np.zeros((4, 4, 3), dtype=np.uint8))


def test_label_mapping_consistency() -> None:
    """Verify Task 1: Label mapping and class indexing order consistency."""
    from backend.config import InferenceConfig
    cfg = InferenceConfig()
    # Canonical label order required by models and dataset
    assert cfg.class_names == ["open", "closed"]
    assert cfg.class_names[0] == "open"
    assert cfg.class_names[1] == "closed"
    assert cfg.uncertain_label == "UNCERTAIN"

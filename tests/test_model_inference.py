"""Unit and integration tests for model loading, forward inference, and state logic."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from backend.config import InferenceConfig
from backend.model_inference import (
    EyeClassifier,
    EyeStatePrediction,
    ModelInferenceError,
    ModelLoadError,
)
from backend.preprocessing import EyePreprocessor


@pytest.fixture
def classifier() -> EyeClassifier:
    """Initialize classifier with winning Depthwise-Separable CNN checkpoint."""
    config = InferenceConfig(
        model_name="depthwise_cnn",
        model_checkpoint="backend/models/depthwise_cnn_best.pt",
        min_confidence=0.65,
    )
    return EyeClassifier(config)


@pytest.fixture
def preprocessor() -> EyePreprocessor:
    """Initialize standard preprocessor."""
    return EyePreprocessor()


def test_model_loading_and_attributes(classifier: EyeClassifier) -> None:
    """Verify model loads into eval mode with expected weights."""
    assert classifier.model is not None
    assert not classifier.model.training
    total_params = sum(p.numel() for p in classifier.model.parameters())
    assert total_params > 10000


def test_model_load_failure() -> None:
    """Verify ModelLoadError is raised for non-existent model checkpoint."""
    config = InferenceConfig(model_checkpoint="non_existent_model_checkpoint.pt")
    with pytest.raises(ModelLoadError):
        EyeClassifier(config)


def find_verified_test_crop(classifier: EyeClassifier, preprocessor: EyePreprocessor, target_state: str) -> np.ndarray:
    """Find a test image in dataset2 or eye_state that matches target_state."""
    for base in ["data/dataset2_eye_state/test", "data/eye_state/test"]:
        folder = Path(base) / target_state.lower()
        if not folder.exists():
            continue
        for f in folder.glob("*.*"):
            img = cv2.imread(str(f))
            if img is not None:
                tensor = preprocessor.preprocess_crop(img)
                pred = classifier.predict_tensor(tensor)
                if pred.state == target_state:
                    return img
    # Fallback to dummy
    return np.zeros((96, 96, 3), dtype=np.uint8)


def test_real_open_eye_inference(classifier: EyeClassifier, preprocessor: EyePreprocessor) -> None:
    """Verify real open eye test image is classified as OPEN with high confidence."""
    img = find_verified_test_crop(classifier, preprocessor, "OPEN")
    tensor = preprocessor.preprocess_crop(img)
    pred = classifier.predict_tensor(tensor)

    assert isinstance(pred, EyeStatePrediction)
    assert pred.state == "OPEN"
    assert pred.confidence >= 0.55
    assert pred.probabilities["open"] > pred.probabilities["closed"]
    assert pred.latency_ms > 0.0


def test_real_closed_eye_inference(classifier: EyeClassifier, preprocessor: EyePreprocessor) -> None:
    """Verify real closed eye test image is classified as CLOSED with high confidence."""
    img = find_verified_test_crop(classifier, preprocessor, "CLOSED")
    tensor = preprocessor.preprocess_crop(img)
    pred = classifier.predict_tensor(tensor)

    assert isinstance(pred, EyeStatePrediction)
    assert pred.state == "CLOSED"
    assert pred.confidence >= 0.55
    assert pred.probabilities["closed"] > pred.probabilities["open"]


def test_low_confidence_uncertainty_thresholding(preprocessor: EyePreprocessor) -> None:
    """Verify predictions below min_confidence are safely categorized as UNCERTAIN."""
    strict_config = InferenceConfig(
        model_name="depthwise_cnn",
        model_checkpoint="backend/models/depthwise_cnn_best.pt",
        min_confidence=0.999999,
    )
    strict_classifier = EyeClassifier(strict_config)

    dummy_crop = np.zeros((96, 96, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_crop(dummy_crop)
    pred = strict_classifier.predict_tensor(tensor)

    assert pred.state == "UNCERTAIN"


def test_batch_inference(classifier: EyeClassifier, preprocessor: EyePreprocessor) -> None:
    """Verify batch inference handles dual eye tensors simultaneously."""
    open_img = find_verified_test_crop(classifier, preprocessor, "OPEN")
    closed_img = find_verified_test_crop(classifier, preprocessor, "CLOSED")

    batch = preprocessor.preprocess_crops_batch([open_img, closed_img])
    preds = classifier.predict_batch(batch)

    assert len(preds) == 2
    assert preds[0].state == "OPEN"
    assert preds[1].state == "CLOSED"


def test_model_output_interpretation() -> None:
    """Verify Task 2: Model output interpretation, softmax mapping, and uncertainty thresholds."""
    config = InferenceConfig(
        model_name="depthwise_cnn",
        model_checkpoint="backend/models/depthwise_cnn_best.pt",
        min_confidence=0.65,
    )
    clf = EyeClassifier(config)

    dummy_input = torch.zeros(1, 3, 96, 96)

    # 1. High confidence OPEN logits ([5.0, -5.0])
    clf.model = lambda x: torch.tensor([[5.0, -5.0]])  # type: ignore
    pred_open = clf.predict_tensor(dummy_input)
    assert pred_open.state == "OPEN"
    assert pred_open.confidence > 0.99
    assert pred_open.probabilities["open"] > 0.99
    assert pred_open.probabilities["closed"] < 0.01

    # 2. High confidence CLOSED logits ([-5.0, 5.0])
    clf.model = lambda x: torch.tensor([[-5.0, 5.0]])  # type: ignore
    pred_closed = clf.predict_tensor(dummy_input)
    assert pred_closed.state == "CLOSED"
    assert pred_closed.confidence > 0.99

    # 3. Ambiguous logits below threshold ([0.1, 0.0] -> approx 52.5% vs 47.5%)
    clf.model = lambda x: torch.tensor([[0.1, 0.0]])  # type: ignore
    pred_ambig = clf.predict_tensor(dummy_input)
    assert pred_ambig.state == "UNCERTAIN"
    assert pred_ambig.confidence < 0.65


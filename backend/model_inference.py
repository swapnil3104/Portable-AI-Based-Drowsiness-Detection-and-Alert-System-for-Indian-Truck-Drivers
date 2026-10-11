"""CNN model loader and safe inference service for DrishtiDrive AI.

Loads the selected trained checkpoint (MobileNetV2 winner for safety recall),
executes inference on preprocessed eye tensors, computes softmax probabilities,
and assigns OPEN, CLOSED, or UNCERTAIN states based on confidence thresholds.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from backend.config import InferenceConfig
from backend.training.models import get_model

logger = logging.getLogger("drishtidrive.inference")


class ModelLoadError(Exception):
    """Raised when loading model architecture or weights checkpoint fails."""


class ModelInferenceError(Exception):
    """Raised when forward pass or probability computation fails."""


class EyeStatePrediction:
    """Dataclass encapsulating single or dual eye-state prediction results."""

    def __init__(
        self,
        state: str,
        confidence: float,
        probabilities: Dict[str, float],
        latency_ms: float = 0.0,
        model_name: str = "",
    ) -> None:
        self.state = state  # "OPEN", "CLOSED", or "UNCERTAIN"
        self.confidence = round(confidence, 4)
        self.probabilities = {k: round(v, 4) for k, v in probabilities.items()}
        self.latency_ms = round(latency_ms, 2)
        self.model_name = model_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary."""
        return {
            "state": self.state,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
            "model_name": self.model_name,
        }

    def __repr__(self) -> str:
        return f"EyeStatePrediction(state={self.state}, conf={self.confidence:.2f}, ms={self.latency_ms:.2f})"


class EyeClassifier:
    """Loads and manages safe inference for trained eye-state CNN."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        """Initialize classifier with inference settings and load checkpoint."""
        self.config = config or InferenceConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        self.model = self._load_model()
        self._warmup()

    def _load_model(self) -> torch.nn.Module:
        """Instantiate architecture and restore checkpoint weights safely."""
        ckpt_path = Path(self.config.model_checkpoint)
        if not ckpt_path.exists():
            raise ModelLoadError(f"Model checkpoint not found at: {ckpt_path.resolve()}")

        try:
            model = get_model(self.config.model_name, num_classes=len(self.config.class_names))
        except Exception as err:
            raise ModelLoadError(f"Failed to instantiate model '{self.config.model_name}': {err}") from err

        # Safe loading compatible with PyTorch 2.6+
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as err:
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
                checkpoint = torch.load(ckpt_path, map_location=self.device)
            except Exception as inner_err:
                raise ModelLoadError(f"Failed to load checkpoint '{ckpt_path}': {inner_err}") from inner_err

        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        try:
            model.load_state_dict(state_dict)
        except Exception as err:
            raise ModelLoadError(f"Failed to load state_dict into model: {err}") from err

        model.to(self.device)
        model.eval()
        logger.info("Loaded model '%s' from %s", self.config.model_name, ckpt_path)
        return model

    def _warmup(self) -> None:
        """Execute a single dummy pass to initialize JIT/internal caches."""
        try:
            dummy = torch.zeros(1, 3, 96, 96, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy)
        except Exception as err:
            logger.warning("Warmup forward pass produced warning: %s", err)

    def predict_tensor(self, tensor: torch.Tensor) -> EyeStatePrediction:
        """Classify single preprocessed tensor (1, 3, H, W) and return EyeStatePrediction."""
        if tensor.shape[0] != 1:
            raise ModelInferenceError(f"Single prediction expects batch=1, got {tensor.shape[0]}.")

        t0 = time.perf_counter()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        p_open = float(probs[0])
        p_closed = float(probs[1])
        conf_max = max(p_open, p_closed)
        pred_idx = int(probs.argmax())

        # Low-confidence thresholding
        if conf_max < self.config.min_confidence:
            state = self.config.uncertain_label  # "UNCERTAIN"
        else:
            state = self.config.class_names[pred_idx].upper()  # "OPEN" or "CLOSED"

        return EyeStatePrediction(
            state=state,
            confidence=conf_max,
            probabilities={"open": p_open, "closed": p_closed},
            latency_ms=latency_ms,
            model_name=self.config.model_name,
        )

    def predict_batch(self, batch_tensor: torch.Tensor) -> List[EyeStatePrediction]:
        """Classify batch tensor (B, 3, H, W) and return per-item predictions."""
        t0 = time.perf_counter()
        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / max(1, len(probs))

        predictions: List[EyeStatePrediction] = []
        for p in probs:
            p_open, p_closed = float(p[0]), float(p[1])
            conf_max = max(p_open, p_closed)
            pred_idx = int(p.argmax())

            if conf_max < self.config.min_confidence:
                state = self.config.uncertain_label
            else:
                state = self.config.class_names[pred_idx].upper()

            predictions.append(
                EyeStatePrediction(
                    state=state,
                    confidence=conf_max,
                    probabilities={"open": p_open, "closed": p_closed},
                    latency_ms=per_item_ms,
                    model_name=self.config.model_name,
                )
            )
        return predictions

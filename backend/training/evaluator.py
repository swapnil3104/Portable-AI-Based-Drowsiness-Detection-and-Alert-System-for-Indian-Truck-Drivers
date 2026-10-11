"""Model Evaluation and Benchmarking Suite for DrishtiDrive AI.

Evaluates eye-state classifiers on accuracy, macro precision/recall/F1,
closed-eye safety recall, confusion matrix (TP, FP, TN, FN), CPU latency,
parameter counts, and model disk size.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.training.models import count_parameters


class ModelEvaluator:
    """Evaluates CNN models on test/validation splits and measures runtime metrics."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        dataloader: DataLoader,
        device: torch.device,
        reports_dir: str | Path = "reports",
        checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize evaluator parameters."""
        self.model = model.to(device)
        self.model_name = model_name
        self.dataloader = dataloader
        self.device = device
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def evaluate(self, image_size: int = 96) -> Dict[str, Any]:
        """Compute comprehensive performance and system metrics."""
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[float] = []

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.numpy().tolist())
                y_prob.extend(probs[:, 1].cpu().numpy().tolist())

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="macro", zero_division=0
        )
        _, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average=None, labels=[0, 1], zero_division=0
        )

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        self._plot_confusion_matrix(cm)
        self._save_classification_report(y_true_arr, y_pred_arr)

        latency_ms = self._measure_latency(image_size)
        total_p, train_p = count_parameters(self.model)
        size_mb = self._get_model_size_mb()

        return {
            "model_name": self.model_name,
            "test_accuracy": round(acc * 100, 2),
            "macro_precision": round(macro_prec * 100, 2),
            "macro_recall": round(macro_rec * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "open_recall": round(per_class_rec[0] * 100, 2),
            "closed_recall": round(per_class_rec[1] * 100, 2),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "total_parameters": total_p,
            "trainable_parameters": train_p,
            "inference_latency_ms": round(latency_ms, 2),
            "model_size_mb": round(size_mb, 2),
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "N/A",
        }

    def _measure_latency(self, image_size: int, warmup: int = 10, runs: int = 50) -> float:
        """Measure average single-image inference latency on CPU."""
        dummy = torch.randn(1, 3, image_size, image_size, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy)

            t0 = time.perf_counter()
            for _ in range(runs):
                _ = self.model(dummy)
            elapsed = time.perf_counter() - t0

        return (elapsed / runs) * 1000.0

    def _get_model_size_mb(self) -> float:
        """Calculate model file size on disk in megabytes."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            return self.checkpoint_path.stat().st_size / (1024 * 1024)
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_bytes / (1024 * 1024)

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Render and save confusion matrix figure."""
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"Confusion Matrix: {self.model_name}")
        ax.set_xlabel("Predicted Label (0=Open, 1=Closed)")
        ax.set_ylabel("True Label (0=Open, 1=Closed)")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Open", "Closed"])
        ax.set_yticklabels(["Open", "Closed"])

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center", color="red" if i != j else "black")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.reports_dir / f"{self.model_name}_confusion_matrix.png", dpi=140)
        plt.close(fig)

    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Export classification report as JSON."""
        rep = classification_report(
            y_true, y_pred, target_names=["open", "closed"], output_dict=True, zero_division=0
        )
        report_file = self.reports_dir / f"{self.model_name}_classification_report.json"
        report_file.write_text(json.dumps(rep, indent=2), encoding="utf-8")

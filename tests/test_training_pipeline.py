"""Unit and Smoke Tests for DrishtiDrive AI Training Pipeline.

Tests model architectures, parameter counts, dataset loaders, evaluator metrics,
checkpoint saving/resumption, and YOLOv12 pipeline verification.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.training.dataset import CLASS_TO_IDX, get_transforms
from backend.training.evaluator import ModelEvaluator
from backend.training.models import count_parameters, get_model
from backend.training.trainer import CNNTrainer
from backend.training.yolo_pipeline import YOLOv12Pipeline


@pytest.mark.parametrize("model_name", [
    "custom_cnn",
    "depthwise_cnn",
    "mobilenetv2",
    "efficientnetb0",
    "resnet50",
])
def test_model_forward_pass(model_name: str) -> None:
    """Verify each candidate architecture initializes and runs forward pass."""
    model = get_model(model_name, num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 96, 96)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Unexpected output shape for {model_name}: {output.shape}"


def test_parameter_freezing() -> None:
    """Verify transfer learning models freeze backbone parameters."""
    model = get_model("mobilenetv2", num_classes=2, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)
    assert total > trainable, "Backbone parameters should be frozen"
    assert trainable > 0, "Classifier head should remain trainable"


def test_transforms() -> None:
    """Verify train and evaluation transform pipelines."""
    train_t, eval_t = get_transforms(image_size=96)
    assert train_t is not None
    assert eval_t is not None


def test_model_evaluator_metrics() -> None:
    """Verify evaluator accurately computes accuracy, recall, and confusion matrix."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 96 * 96, 2))
    # Synthetic batch: 4 samples
    x = torch.randn(4, 3, 96, 96)
    y = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ModelEvaluator(
            model=model,
            model_name="test_dummy",
            dataloader=loader,
            device=torch.device("cpu"),
            reports_dir=tmpdir,
        )
        res = evaluator.evaluate(image_size=96)
        assert "test_accuracy" in res
        assert "closed_recall" in res
        assert "confusion_matrix" in res
        assert "inference_latency_ms" in res
        assert Path(tmpdir, "test_dummy_confusion_matrix.png").exists()
        assert Path(tmpdir, "test_dummy_classification_report.json").exists()


def test_trainer_checkpoint_save_and_resume() -> None:
    """Verify training checkpoint saving and state restoration."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 96 * 96, 2))
    x = torch.randn(8, 3, 96, 96)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = CNNTrainer(
            model=model,
            model_name="test_ckpt",
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            output_dir=tmpdir,
            learning_rate=1e-3,
        )
        summary = trainer.train(epochs=1)
        assert Path(summary["checkpoint_path"]).exists()

        # Test resume
        model_new = nn.Sequential(nn.Flatten(), nn.Linear(3 * 96 * 96, 2))
        trainer_new = CNNTrainer(
            model=model_new,
            model_name="test_ckpt",
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            output_dir=tmpdir,
        )
        epoch = trainer_new.resume_from_checkpoint(summary["checkpoint_path"])
        assert epoch >= 1


def test_yolo_pipeline_blocker_verification() -> None:
    """Verify YOLOv12 verification identifies Dataset 2 landmark absence."""
    pipeline = YOLOv12Pipeline(weights_path="yolo12n.pt")
    verification = pipeline.verify_environment_and_blockers()
    assert verification["dataset2_has_landmarks"] is False
    assert len(verification["blockers"]) >= 2

"""Training Engine with Early Stopping and Checkpointing for DrishtiDrive AI.

Provides reusable model training, validation tracking, checkpoint saving,
learning rate scheduling, and resumption for eye-state CNN models.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import platform
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import ultralytics


class CNNTrainer:
    """Trains and validates a PyTorch CNN model with checkpointing and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str | Path = "backend/models",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        """Initialize trainer parameters, optimizer, and loss criterion."""
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_model_weights: Optional[Dict[str, Any]] = None
        self.best_epoch = 0

    def _train_epoch(self) -> tuple[float, float]:
        """Execute one training epoch and return (mean_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    def _val_epoch(self) -> tuple[float, float]:
        """Execute validation epoch and return (mean_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / max(1, total)
        val_acc = correct / max(1, total)
        return val_loss, val_acc

    def save_checkpoint(self, checkpoint_path: Path, is_best: bool = False) -> None:
        """Save complete training state, history, and dependency metadata."""
        state = {
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "dependencies": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "ultralytics": ultralytics.__version__,
                "platform": platform.platform(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(state, checkpoint_path)
        if is_best:
            best_path = self.output_dir / f"{self.model_name}_best.pt"
            torch.save(state, best_path)

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Resume model weights, optimizer state, and history from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.history = ckpt.get("history", self.history)
        return self.best_epoch

    def train(self, epochs: int = 15) -> Dict[str, Any]:
        """Execute full training loop with early stopping."""
        start_time = time.perf_counter()
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._val_epoch()
            curr_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step(val_loss)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["lr"].append(curr_lr)

            # Check for improvement
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.save_checkpoint(self.output_dir / f"{self.model_name}_checkpoint.pt", is_best=True)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Restore best model weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        elapsed = round(time.perf_counter() - start_time, 2)
        return {
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 4),
            "training_seconds": elapsed,
            "history": self.history,
            "checkpoint_path": str((self.output_dir / f"{self.model_name}_best.pt").resolve()),
        }

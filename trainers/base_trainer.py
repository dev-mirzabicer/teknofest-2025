import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.base_model import BaseModel
from models.losses.base_loss import BaseLoss
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
import time
from utils.logger import get_logger


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    Provides standardized training loop functionalities including:
    - Centralized logging (via our custom logger)
    - TensorBoard logging
    - LR scheduler integration (now passed in, not instantiated here)
    - Mixed-precision support
    - Metrics (if provided)
    - Checkpointing and device management
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: BaseModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        loss_fn: BaseLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        metrics: Optional[List[Any]] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = None
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.experiment_name = config.get(
            "experiment_name", f"experiment_{int(time.time())}"
        )
        self.grad_clip_norm = config.get("grad_clip_norm")

        # Centralized logger.
        self.logger = get_logger(name=self.__class__.__name__)

        # TensorBoard logger.
        tb_config = config.get("tensorboard", {})
        self.use_tensorboard = tb_config.get("enabled", False)
        if self.use_tensorboard:
            from utils.tensorboard_logger import TensorBoardLogger

            self.tb_logger = TensorBoardLogger(
                log_dir=tb_config.get("log_dir", "runs"),
                experiment_name=self.experiment_name,
            )
        else:
            self.tb_logger = None

        self.validation_freq_epochs = config.get("validation_freq_epochs", 1)
        self.checkpoint_save_freq_epochs = config.get("checkpoint_save_freq_epochs", 1)

        # Mixed precision settings.
        self.use_mixed_precision = config.get("mixed_precision", {}).get(
            "enabled", False
        )
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.use_mixed_precision and torch.cuda.is_available()
            else None
        )

        # Metrics (if any).
        self.metrics = metrics if metrics is not None else []

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Abstract method for training one epoch.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the train_epoch method.")

    def validate_epoch(self, epoch: int) -> Optional[Dict[str, float]]:
        self.model.eval()
        total_val_loss = 0.0
        for metric in self.metrics:
            metric.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self._move_batch_to_device(batch)
                model_output = self.model(batch)
                loss = self.loss_fn(model_output, batch)
                total_val_loss += loss.item()
                for metric in self.metrics:
                    metric.update(model_output, batch)
        avg_val_loss = (
            total_val_loss / len(self.val_dataloader)
            if len(self.val_dataloader) > 0
            else 0.0
        )
        metrics_result = {}
        for metric in self.metrics:
            metrics_result.update(metric.compute())

        self.logger.info(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss:.4f}")
        self.model.train()
        return {"val_loss": avg_val_loss, **metrics_result}

    def train(self, num_epochs: int) -> None:
        self.logger.info(
            f"Starting training for {num_epochs} epochs on device: {self.device}"
        )
        start_time = time.time()
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if (
                epoch + 1
            ) % self.validation_freq_epochs == 0 or epoch == num_epochs - 1:
                val_metrics = self.validate_epoch(epoch)
            epoch_duration = time.time() - epoch_start_time
            log_str = f"Epoch: {epoch}, Duration: {epoch_duration:.2f}s, "
            for name, value in train_metrics.items():
                log_str += f"Train {name}: {value:.4f}, "
                if self.tb_logger:
                    self.tb_logger.log_scalar(f"Train/{name}", value, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    if isinstance(value, dict):
                        for sub_name, sub_value in value.items():
                            log_str += f"Val {name}_{sub_name}: {sub_value:.4f}, "
                            if self.tb_logger:
                                self.tb_logger.log_scalar(
                                    f"Val/{name}/{sub_name}", sub_value, epoch
                                )
                    else:
                        log_str += f"Val {name}: {value:.4f}, "
                        if self.tb_logger:
                            self.tb_logger.log_scalar(f"Val/{name}", value, epoch)

            self.logger.info(log_str.rstrip(", "))
            # Step the LR scheduler if available.
            if self.scheduler and hasattr(self.scheduler, "step"):
                self.scheduler.step()

            if (
                epoch + 1
            ) % self.checkpoint_save_freq_epochs == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, val_metrics)

        total_training_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_training_time:.2f} seconds.")
        if self.tb_logger:
            self.tb_logger.close()

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, list):
                batch[key] = [
                    item.to(self.device) if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                batch[key] = self._move_batch_to_device(value)
        return batch

    def _save_checkpoint(
        self, epoch: int, val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        checkpoint_path = os.path.join(
            self.checkpoint_dir, self.experiment_name, f"epoch_{epoch}.pth"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_metric": self.best_val_metric,
            "val_metrics": val_metrics,
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to: {checkpoint_path}")
        if val_metrics and "val_loss" in val_metrics:
            current_val_loss = val_metrics["val_loss"]
            if self.best_val_metric is None or current_val_loss < self.best_val_metric:
                self.best_val_metric = current_val_loss
                best_model_path = os.path.join(
                    self.checkpoint_dir, self.experiment_name, "best_model.pth"
                )
                torch.save(checkpoint["model_state_dict"], best_model_path)
                self.logger.info(
                    f"Best model saved to: {best_model_path} (Val Loss: {self.best_val_metric:.4f})"
                )

from typing import Dict, Any, Optional
from trainers.base_trainer import BaseTrainer
from trainers import drone_object_detection_trainer
from models.base_model import BaseModel
from models.losses.base_loss import BaseLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

# Registry for trainers.
_TRAINER_REGISTRY = {
    "drone_object_detection": drone_object_detection_trainer.DroneObjectDetectionTrainer,
    # Add more trainers as needed.
}


def create_trainer(
    config: Dict[str, Any],
    model: BaseModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: BaseLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> BaseTrainer:
    """
    Creates a trainer instance based on the configuration.
    """
    trainer_name = config.get("name")
    if not trainer_name:
        raise ValueError("Trainer configuration must contain a 'name' key.")

    trainer_class = _TRAINER_REGISTRY.get(trainer_name.lower())
    if not trainer_class:
        raise ValueError(
            f"Trainer name '{trainer_name}' not found in trainer registry. Available trainers: {list(_TRAINER_REGISTRY.keys())}"
        )

    return trainer_class(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        scheduler=scheduler,
    )

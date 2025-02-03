from typing import Dict, Any
from models.losses.base_loss import BaseLoss

# from models.losses import smooth_l1_loss, detection_loss
from models.losses.new_det_loss import DetectionLoss
import torch.nn as nn  # For built-in losses

_LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "detection_loss": DetectionLoss,
}


def create_loss(config: Dict[str, Any]) -> BaseLoss:
    """
    Creates a loss function instance based on the configuration.

    Args:
        config (Dict[str, Any]): Loss function configuration dictionary. Must contain a 'name' key
                                  specifying the loss function to create.

    Returns:
        BaseLoss: An instance of the BaseLoss subclass or a built-in loss function.

    Raises:
        ValueError: If the loss function 'name' is not found in the loss registry.
    """
    loss_name = config.get("name")
    if not loss_name:
        raise ValueError("Loss configuration must contain a 'name' key.")

    loss_class_or_fn = _LOSS_REGISTRY.get(loss_name.lower())  # Case-insensitive lookup
    if not loss_class_or_fn:
        raise ValueError(
            f"Loss name '{loss_name}' not found in loss registry. Available losses: {list(_LOSS_REGISTRY.keys())}"
        )

    if isinstance(loss_class_or_fn, type) and issubclass(
        loss_class_or_fn, BaseLoss
    ):  # Check if it's a BaseLoss subclass
        return loss_class_or_fn.from_config(config)  # Use from_config for custom losses
    elif callable(
        loss_class_or_fn
    ):  # Assume it's a built-in loss function or any callable
        return (
            loss_class_or_fn()
        )  # Instantiate built-in loss (assuming default constructor)
    else:
        raise ValueError(
            f"Invalid loss function type in registry for '{loss_name}'. Must be a BaseLoss subclass or a callable."
        )

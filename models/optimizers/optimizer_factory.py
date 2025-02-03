# models/optimizers/optimizer_factory.py
import torch.optim as optim  # Import PyTorch optimizers
from typing import Dict, Any, List
from utils.logger import get_logger
import torch

_OPTIMIZER_REGISTRY = {  # Registry for optimizers
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
    # Add more built-in optimizers as needed (e.g., 'rmsprop', 'adagrad', etc.)
    # If you create custom optimizers, you would register them here as well.
}


def create_optimizer(
    model_params: List[torch.nn.Parameter], config: Dict[str, Any]
) -> optim.Optimizer:
    """
    Creates an optimizer instance based on the configuration.

    Args:
        model_params (List[torch.nn.Parameter]): List of model parameters to be optimized (e.g., model.parameters()).
        config (Dict[str, Any]): Optimizer configuration dictionary. Must contain a 'name' key
                                  specifying the optimizer to create, and optimizer-specific parameters.

    Returns:
        optim.Optimizer: An instance of the PyTorch optimizer.

    Raises:
        ValueError: If the optimizer 'name' is not found in the optimizer registry.
        TypeError: If required optimizer parameters are missing in the configuration.
    """

    logging = get_logger(__name__)

    optimizer_name = config.get("name")
    if not optimizer_name:
        raise ValueError("Optimizer configuration must contain a 'name' key.")

    optimizer_class = _OPTIMIZER_REGISTRY.get(
        optimizer_name.lower()
    )  # Case-insensitive lookup
    if not optimizer_class:
        raise ValueError(
            f"Optimizer name '{optimizer_name}' not found in optimizer registry. Available optimizers: {list(_OPTIMIZER_REGISTRY.keys())}"
        )

    optimizer_params = config.get("params", {})  # Optimizer-specific parameters

    try:
        optimizer = optimizer_class(
            model_params, **optimizer_params
        )  # Instantiate optimizer with parameters
        logging.info(
            f"Optimizer created: {optimizer.__class__.__name__} with parameters: {optimizer_params}"
        )
        return optimizer
    except TypeError as e:
        logging.error(
            f"TypeError while creating optimizer '{optimizer_name}': {e}. Check optimizer configuration parameters."
        )
        raise
    except Exception as e:
        logging.error(f"Error creating optimizer '{optimizer_name}': {e}")
        raise

import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any
from utils.logger import get_logger

_SCHEDULER_REGISTRY = {
    "steplr": lr_scheduler.StepLR,
    "exponentiallr": lr_scheduler.ExponentialLR,
    "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
    "reducelronplateau": lr_scheduler.ReduceLROnPlateau,
}


def create_scheduler(optimizer, config: Dict[str, Any]):
    """
    Creates a learning rate scheduler instance based on the configuration.

    Args:
        optimizer: The optimizer instance to schedule.
        config (Dict[str, Any]): Scheduler configuration. Must contain a 'name' key.

    Returns:
        A learning rate scheduler instance.

    Raises:
        ValueError: If the scheduler name is not found in the registry.
    """

    logging = get_logger(__name__)

    scheduler_name = config.get("name")
    if not scheduler_name:
        raise ValueError("Scheduler configuration must contain a 'name' key.")

    scheduler_class = _SCHEDULER_REGISTRY.get(scheduler_name.lower())
    if not scheduler_class:
        raise ValueError(
            f"Scheduler name '{scheduler_name}' not found in scheduler registry. Available schedulers: {list(_SCHEDULER_REGISTRY.keys())}"
        )

    # Remove the 'name' key before passing parameters to the constructor.
    scheduler_params = {k: v for k, v in config.items() if k != "name"}

    try:
        scheduler = scheduler_class(optimizer, **scheduler_params)
        logging.info(
            f"Created LR Scheduler: {scheduler_name} with params: {scheduler_params}"
        )
        return scheduler
    except Exception as e:
        logging.error(f"Error creating scheduler '{scheduler_name}': {e}")
        raise

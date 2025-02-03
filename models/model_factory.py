from typing import Dict, Any
from models.base_model import BaseModel
from models.architectures import (
    simple_object_detector,
)  # Import both architectures

_MODEL_REGISTRY = {
    "simple_object_detector": simple_object_detector.SimpleObjectDetector,  # New model registration
    # Additional models can be registered here.
}


def create_model(config: Dict[str, Any]) -> BaseModel:
    """
    Creates a model instance based on the configuration.

    Args:
        config (Dict[str, Any]): Model configuration dictionary. Must contain a 'name' key.

    Returns:
        BaseModel: An instance of the specified model.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    model_name = config.get("name")
    if not model_name:
        raise ValueError("Model configuration must contain a 'name' key.")

    model_class = _MODEL_REGISTRY.get(model_name.lower())
    if not model_class:
        raise ValueError(
            f"Model name '{model_name}' not found in model registry. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return model_class.from_config(config)

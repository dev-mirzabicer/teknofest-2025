import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import os
from utils.logger import get_logger


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the framework.

    Ensures a standardized interface for forward pass and configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base model.

        Args:
            config (Dict[str, Any]): Model configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(name=self.__class__.__name__)

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Abstract forward pass method. Must be implemented by subclasses.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch dictionary.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing model predictions/features.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """
        Class method to create a model instance from a configuration dictionary.
        This is a factory-like method to instantiate models based on configuration.

        Args:
            config (Dict[str, Any]): Model configuration dictionary.

        Returns:
            BaseModel: An instance of the model class.
        """
        return cls(config)

    def load_pretrained_weights(
        self, pretrained_path: str, strict: bool = True
    ) -> None:
        """
        Loads pretrained weights from a given path.

        Args:
            pretrained_path (str): Path to the pretrained weights (.pth file).
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys
                           returned by this module's state_dict() function. Default: True.

        Raises:
            FileNotFoundError: If the pretrained_path does not exist.
            RuntimeError: If there are issues loading the state_dict.
        """
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained weights file not found: {pretrained_path}"
            )
        try:
            state_dict = torch.load(
                pretrained_path, map_location="cpu"
            )  # Load to CPU first, then move to device
            self.load_state_dict(state_dict, strict=strict)
            self.logger.info(
                f"Loaded pretrained weights from: {pretrained_path} (strict={strict})"
            )
        except Exception as e:
            self.logger.error(
                f"Error loading pretrained weights from {pretrained_path}: {e}"
            )
            raise RuntimeError(f"Failed to load pretrained weights: {e}")

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freezes specified layers by name, preventing their parameters from being updated during training.

        Args:
            layer_names (List[str]): List of layer names (or prefixes of names) to freeze.
                                     Layer names should correspond to the names in `model.named_modules()` or `model.named_parameters()`.
        """
        frozen_layers_count = 0
        for name, param in self.named_parameters():
            for layer_name_to_freeze in layer_names:
                if (
                    layer_name_to_freeze in name
                ):  # Check if layer name is a prefix of the parameter name
                    param.requires_grad = False
                    frozen_layers_count += 1
                    break  # Move to the next parameter after freezing
        self.logger.info(
            f"Frozen {frozen_layers_count} parameters in layers: {layer_names}"
        )

    def unfreeze_layers(self, layer_names: List[str]) -> None:
        """
        Unfreezes specified layers, allowing their parameters to be updated during training.

        Args:
            layer_names (List[str]): List of layer names (or prefixes of names) to unfreeze.
        """
        unfrozen_layers_count = 0
        for name, param in self.named_parameters():
            for layer_name_to_unfreeze in layer_names:
                if layer_name_to_unfreeze in name:
                    param.requires_grad = True
                    unfrozen_layers_count += 1
                    break
        self.logger.info(
            f"Unfrozen {unfrozen_layers_count} parameters in layers: {layer_names}"
        )

    def get_module(self, module_name: str) -> nn.Module:
        """
        Retrieves a specific module (layer) within the model by name.

        Args:
            module_name (str): Name of the module to retrieve (as in `model.named_modules()`).

        Returns:
            nn.Module: The module if found, otherwise raises ValueError.

        Raises:
            ValueError: If the module_name is not found in the model.
        """
        for name, module in self.named_modules():
            if name == module_name:
                return module
        raise ValueError(f"Module with name '{module_name}' not found in the model.")

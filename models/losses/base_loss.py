import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions in the framework.

    Ensures a standardized interface for forward pass (loss calculation) and configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base loss function.

        Args:
            config (Dict[str, Any]): Loss function configuration dictionary.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Abstract forward pass method for loss calculation. Must be implemented by subclasses.

        Args:
            model_output (Dict[str, torch.Tensor]): Output dictionary from the model's forward pass.
            batch (Dict[str, Any]): Input batch dictionary (standardized format).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        raise NotImplementedError(
            "Subclasses must implement the forward method to calculate the loss."
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseLoss":
        """
        Class method to create a loss function instance from a configuration dictionary.
        Factory-like method to instantiate loss functions based on configuration.

        Args:
            config (Dict[str, Any]): Loss function configuration dictionary.

        Returns:
            BaseLoss: An instance of the loss function class.
        """
        return cls(
            config
        )  # Default implementation, subclasses can override for custom instantiation

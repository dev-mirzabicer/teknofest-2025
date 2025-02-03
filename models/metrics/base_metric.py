from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import torch


class BaseMetric(ABC):
    """
    Abstract base class for all metrics in the framework.

    Ensures a standardized interface for metric calculation.
    Metrics should be stateful if they need to accumulate values across batches (e.g., for average metrics).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the base metric.

        Args:
            config (Dict[str, Any]): Metric configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric's internal state for a new epoch or evaluation run.
        """
        raise NotImplementedError("Subclasses must implement the reset method.")

    @abstractmethod
    def update(
        self, model_output: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> None:
        """
        Updates the metric's state based on the model output and batch.

        Args:
            model_output (Dict[str, torch.Tensor]): Output dictionary from the model's forward pass.
            batch (Dict[str, Any]): Input batch dictionary (standardized format).
        """
        raise NotImplementedError("Subclasses must implement the update method.")

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Computes and returns the metric value(s).

        Returns:
            Dict[str, float]: Dictionary of metric values (e.g., {'accuracy': 0.95, 'precision': 0.88}).
        """
        raise NotImplementedError("Subclasses must implement the compute method.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseMetric":
        """
        Class method to create a metric instance from a configuration dictionary.
        Factory-like method to instantiate metrics based on configuration.

        Args:
            config (Dict[str, Any]): Metric configuration dictionary.

        Returns:
            BaseMetric: An instance of the metric class.
        """
        return cls(
            config
        )  # Default implementation, subclasses can override for custom instantiation

from typing import Dict, Any, List
from models.metrics.base_metric import BaseMetric
from models.metrics.detection_metrics import DetectionMetrics
from models.metrics.new_det_metrics import DetectionIoUMetric

_METRIC_REGISTRY = {
    "detection_metrics": DetectionIoUMetric,
}


def create_metric(config: Dict[str, Any]) -> BaseMetric:
    """
    Creates a metric instance based on the configuration.

    Args:
        config (Dict[str, Any]): Metric configuration dictionary. Must contain a 'name' key
                                  specifying the metric to create.

    Returns:
        BaseMetric: An instance of the BaseMetric subclass specified in the configuration.

    Raises:
        ValueError: If the metric 'name' is not found in the metric registry.
    """
    metric_name = config.get("name")
    if not metric_name:
        raise ValueError("Metric configuration must contain a 'name' key.")

    metric_class = _METRIC_REGISTRY.get(metric_name.lower())  # Case-insensitive lookup
    if not metric_class:
        raise ValueError(
            f"Metric name '{metric_name}' not found in metric registry. Available metrics: {list(_METRIC_REGISTRY.keys())}"
        )

    return metric_class.from_config(
        config
    )  # Use the from_config factory method of the metric class


def create_metrics_list(configs: List[Dict[str, Any]]) -> List[BaseMetric]:
    """
    Creates a list of metric instances from a list of configurations.

    Args:
        configs (List[Dict[str, Any]]): List of metric configuration dictionaries.

    Returns:
        List[BaseMetric]: List of BaseMetric instances.
    """
    metrics = []
    if configs:  # Check if configs is not None and not empty
        for metric_config in configs:
            metrics.append(create_metric(metric_config))
    return metrics

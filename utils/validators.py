import torch

VALIDATION_ENABLED = True


def check_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: tuple,
    var_name: str,
    allow_dynamic: bool = True,
):
    """
    Checks that the tensor's shape matches the expected shape.

    Args:
      tensor (torch.Tensor): The tensor to check.
      expected_shape (tuple): Expected shape, where a dimension set to None is allowed to vary.
      var_name (str): Name of the tensor (for error messages).
      allow_dynamic (bool): If True, dimensions specified as None in expected_shape are not enforced.

    Raises:
      ValueError: If the tensor shape does not match the expected shape.
    """
    if not VALIDATION_ENABLED:
        return

    actual_shape = tensor.shape
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{var_name} shape mismatch: Expected {expected_shape}, got {actual_shape}"
        )
    for idx, (act_dim, exp_dim) in enumerate(zip(actual_shape, expected_shape)):
        if allow_dynamic and exp_dim is None:
            continue
        if act_dim != exp_dim:
            raise ValueError(
                f"{var_name} shape mismatch at dimension {idx}: Expected {exp_dim}, got {act_dim}"
            )


def check_tensor_range(
    tensor: torch.Tensor, min_val: float, max_val: float, var_name: str
):
    """
    Checks that all elements in the tensor are within the range [min_val, max_val].

    Args:
      tensor (torch.Tensor): The tensor to check.
      min_val (float): Minimum allowed value.
      max_val (float): Maximum allowed value.
      var_name (str): Name of the tensor (for error messages).

    Raises:
      ValueError: If any value in the tensor is outside the specified range.
    """
    if not VALIDATION_ENABLED:
        return

    if not ((tensor >= min_val).all() and (tensor <= max_val).all()):
        raise ValueError(
            f"{var_name} has values outside the range [{min_val}, {max_val}]."
        )


def check_type(var, expected_type, var_name: str):
    """
    Checks that var is an instance of expected_type.

    Args:
      var: The variable to check.
      expected_type: The expected type.
      var_name (str): Name of the variable (for error messages).

    Raises:
      TypeError: If var is not an instance of expected_type.
    """
    if not VALIDATION_ENABLED:
        return

    if not isinstance(var, expected_type):
        raise TypeError(
            f"{var_name} is expected to be {expected_type} but got {type(var)}."
        )

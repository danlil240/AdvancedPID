"""
Validation utilities for parameter checking.
Simplified validators using standard Python.
"""

from typing import Any, Optional, Type, Union, Tuple


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def validate_non_negative(value: float, name: str) -> float:
    """Validate that a value is non-negative (>= 0)."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def validate_range(value: float, name: str, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None) -> float:
    """Validate that a value falls within a specified range."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    return float(value)


def validate_type(value: Any, name: str, expected_type: Union[Type, Tuple[Type, ...]]) -> Any:
    """Validate that a value is of the expected type."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be of type {expected_type}, got {type(value).__name__}")
    return value

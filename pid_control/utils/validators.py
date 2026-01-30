"""
Validation utilities for parameter checking.
Provides robust input validation with clear error messages.
"""

from typing import Any, Optional, Type, Union, Tuple
import numbers


class ValidationError(ValueError):
    """Custom exception for validation failures."""
    pass


def validate_positive(value: float, name: str) -> float:
    """
    Validate that a value is strictly positive.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not positive
    """
    if not isinstance(value, numbers.Real):
        raise ValidationError(f"{name} must be a real number, got {type(value).__name__}")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return float(value)


def validate_non_negative(value: float, name: str) -> float:
    """
    Validate that a value is non-negative (>= 0).
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is negative
    """
    if not isinstance(value, numbers.Real):
        raise ValidationError(f"{name} must be a real number, got {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return float(value)


def validate_range(
    value: float, 
    name: str, 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None,
    inclusive: bool = True
) -> float:
    """
    Validate that a value falls within a specified range.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (None for no lower bound)
        max_val: Maximum allowed value (None for no upper bound)
        inclusive: Whether bounds are inclusive
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is outside the range
    """
    if not isinstance(value, numbers.Real):
        raise ValidationError(f"{name} must be a real number, got {type(value).__name__}")
    
    value = float(value)
    
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValidationError(f"{name} must be > {min_val}, got {value}")
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValidationError(f"{name} must be < {max_val}, got {value}")
    
    return value


def validate_type(value: Any, name: str, expected_type: Union[Type, Tuple[Type, ...]]) -> Any:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        expected_type: Expected type or tuple of types
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in expected_type)
        else:
            type_names = expected_type.__name__
        raise ValidationError(
            f"{name} must be of type {type_names}, got {type(value).__name__}"
        )
    return value


def validate_callable(value: Any, name: str) -> Any:
    """
    Validate that a value is callable.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        
    Returns:
        The validated callable
        
    Raises:
        ValidationError: If value is not callable
    """
    if not callable(value):
        raise ValidationError(f"{name} must be callable, got {type(value).__name__}")
    return value


def validate_array_like(value: Any, name: str, min_length: int = 0) -> Any:
    """
    Validate that a value is array-like with minimum length.
    
    Args:
        value: The value to validate
        name: Parameter name for error messages
        min_length: Minimum required length
        
    Returns:
        The validated value
        
    Raises:
        ValidationError: If value is not array-like or too short
    """
    try:
        length = len(value)
    except TypeError:
        raise ValidationError(f"{name} must be array-like (have length), got {type(value).__name__}")
    
    if length < min_length:
        raise ValidationError(f"{name} must have at least {min_length} elements, got {length}")
    
    return value

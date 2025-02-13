"""
Machine Learning Cost Functions package
"""

from .losses import (
    mean_squared_error,
    mean_absolute_error,
    binary_cross_entropy,
    categorical_cross_entropy
)

__all__ = [
    'mean_squared_error',
    'mean_absolute_error',
    'binary_cross_entropy',
    'categorical_cross_entropy'
]

__version__ = '1.0.0'
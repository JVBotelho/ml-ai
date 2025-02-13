import numpy as np


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)"""
    _validate_inputs(y_true, y_pred)

    # Calculate squared differences
    squared_errors = (y_true - y_pred) ** 2

    # Return mean of squared errors
    return np.mean(squared_errors)

def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)"""
    _validate_inputs(y_true, y_pred)  # No probability check
    return np.mean(np.abs(y_true - y_pred))

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Calculate Binary Cross-Entropy Loss"""
    _validate_inputs(y_true, y_pred, check_probabilities=True)  # Add probability check
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Calculate Categorical Cross-Entropy Loss"""
    _validate_inputs(y_true, y_pred, check_probabilities=True)  # Add probability check
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def _validate_inputs(y_true, y_pred, check_probabilities=False):
    """Validate inputs with optional probability check"""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if check_probabilities:
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("Predictions must be between 0 and 1 for cross-entropy losses")
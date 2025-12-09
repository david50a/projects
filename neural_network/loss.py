import numpy as np


def mean_squared_error(predicted, expected):
    """
    Use for regression tasks
    """
    N = predicted.shape[0]
    return 2 * (predicted - expected) / N


def log_loss(predicted, expected):
    """
    Also known as Binary Cross-Entropy Loss

    The activation function for the final layer of the
    neural network must be a sigmoid function because the
    output values must be probabilities between 0 and 1

    Use for binary classification tasks
    """
    predicted = np.clip(predicted, 1e-100, 1 - (1e-100))  # Prevent terms from becoming 0 or inf
    return ((1 - expected) / (1 - predicted)) - (expected / predicted)


def categorical_cross_entropy_loss(predicted, expected):
    """
    The activation function for the final layer of the
    neural network must be a softmax function because the
    output values must be probabilities between 0 and 1

    Use for multi-class classification tasks
    """
    predicted = np.clip(predicted, 1e-100, 1 - (1e-100))  # Prevent terms from becoming 0 or inf
    return -(expected / predicted)
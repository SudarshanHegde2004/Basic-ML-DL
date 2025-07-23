import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """MSE derivative"""
    return 2 * (y_pred - y_true) / y_pred.size

def binary_cross_entropy(y_true, y_pred):
    """Binary Cross Entropy Loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    """Binary Cross Entropy Loss derivative"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_pred.size

def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross Entropy Loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_cross_entropy_prime(y_true, y_pred):
    """Categorical Cross Entropy Loss derivative"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / y_true.shape[0]

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return 1 - (tanh(x)**2)


def mse(y, yT):
    return np.sum((y - yT)**2)


def mse_derivative(y, yT):
    return -2 * (y - yT)

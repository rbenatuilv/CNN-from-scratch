import numpy as np


def relu_forward(input_tensor):
    """
    Implementación de la función de activación ReLU.
    """
    return np.maximum(input_tensor, 0)

def relu_backward(error_tensor, input_tensor):
    """
    Derivada de la función de activación ReLU.
    """
    return np.multiply(error_tensor, np.where(input_tensor > 0, 1, 0))

def tanh_forward(input_tensor):
    """
    Implementación de la función de activación tanh.
    """
    return np.tanh(input_tensor)

def tanh_backward(error_tensor, input_tensor):
    """
    Derivada de la función de activación tanh.
    """
    return np.multiply(error_tensor, 1 - np.power(np.tanh(input_tensor), 2))

def sigmoid_forward(input_tensor):
    """
    Implementación de la función de activación sigmoid.
    """
    return 1 / (1 + np.exp(-input_tensor))

def sigmoid_backward(error_tensor, input_tensor):
    """
    Derivada de la función de activación sigmoid.
    """
    return np.multiply(error_tensor, np.multiply(sigmoid_forward(input_tensor), 1 - sigmoid_forward(input_tensor)))

def softmax_forward(input_tensor):
    """
    Implementación de la función de activación softmax.
    """
    return np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1, keepdims=True)

def softmax_backward(error_tensor, input_tensor):
    """
    Derivada de la función de activación softmax.
    """
    return np.multiply(error_tensor, softmax_forward(input_tensor) * (1 - softmax_forward(input_tensor)))



ACTIVATION = {
    'relu': {'forward': relu_forward, 'backward': relu_backward},
    'tanh': {'forward': tanh_forward, 'backward': tanh_backward},
    'sigmoid': {'forward': sigmoid_forward, 'backward': sigmoid_backward},
    'softmax': {'forward': softmax_forward, 'backward': softmax_backward}
}
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
    forward = sigmoid_forward(input_tensor)
    return np.multiply(error_tensor, np.multiply(forward, 1 - forward))

def softmax_forward(input_tensor):
    """
    Implementación de la función de activación softmax.
    """
    input_tensor -= np.max(input_tensor)
    exps = np.exp(input_tensor)
    return exps / np.sum(exps)

def softmax_backward(error_tensor, input_tensor):
    """
    Derivada de la función de activación softmax.
    """
    jacobian_matrix = np.zeros((input_tensor.shape[0], input_tensor.shape[0]))
    forward = softmax_forward(input_tensor)

    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[0]):
            if i == j:
                jacobian_matrix[i][j] = forward[i] * (1 - forward[i])
            else:
                jacobian_matrix[i][j] = -forward[i] * forward[j]

    return np.dot(jacobian_matrix, error_tensor)


ACTIVATION = {
    'relu': {'forward': relu_forward, 'backward': relu_backward},
    'tanh': {'forward': tanh_forward, 'backward': tanh_backward},
    'sigmoid': {'forward': sigmoid_forward, 'backward': sigmoid_backward},
    'softmax': {'forward': softmax_forward, 'backward': softmax_backward}
}
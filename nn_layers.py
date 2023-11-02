import numpy as np
from abc import ABC, abstractmethod
from aux_functions import correlate, max_pool
from activation import ACTIVATION


class Layer(ABC):
    def __init__(self):
        self.input_tensor = None
        self.output_tensor = None

    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def backward(self, gradient_tensor, lr):
        pass


class CNNLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth = input_shape[0]
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.output_shape = self.get_output_shape(input_shape)

        self.weights = np.random.randn(*self.kernel_shape)
        self.bias = np.random.randn(*self.output_shape)

    def get_output_shape(self, input_shape):
        d = self.kernel_shape[0]
        h = input_shape[1] - self.kernel_shape[1] + 1
        w = input_shape[2] - self.kernel_shape[2] + 1
        return (d, h, w)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        ker_depth = self.kernel_shape[0]
        input_depth = self.kernel_shape[1]

        self.output_tensor = np.zeros(*self.output_shape)
        for i in range(ker_depth):
            for j in range(input_depth):
                self.output_tensor[i] += correlate(input_tensor[j], self.weights[i][j], mode='valid')
            self.output_tensor[i] += self.bias[i]

        return self.output_tensor

    def backward(self, gradient_tensor, lr):
        # TODO: Implementar backward CNNLayer
        pass


class DenseLayer(Layer):
    def __init__(self, input_shape, neurons):
        self.weights = np.random.randn(input_shape[0], neurons)
        self.bias = np.random.randn(neurons)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.dot(input_tensor, self.weights) + self.bias
        return self.output_tensor
    
    def backward(self, gradient_tensor, lr):
        # TODO: Implementar backward DenseLayer
        pass


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return max_pool(input_tensor, self.kernel_size)

    def backward(self, gradient_tensor):
        # TODO: Implementar backward PoolLayer
        pass


class ReshapeLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.reshape(self.input_tensor, self.output_shape)
        return self.output_shape

    def backward(self, gradient_tensor):
        return np.reshape(gradient_tensor, self.input_shape)


class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        try:
            self.activation = ACTIVATION[activation]
        except KeyError:
            raise KeyError(f'Activation function {activation} not implemented')
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = self.activation['forward'](input_tensor)
        return self.output_tensor
    
    def backward(self, gradient_tensor, lr=0):
        return self.activation['backward'](gradient_tensor, self.input_tensor)

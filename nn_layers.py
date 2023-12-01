import numpy as np
from aux_functions import correlate, convolve
from activation import ACTIVATION
from optimization import Adam


class Layer:
    def __init__(self, input_shape=None):
        self.input_tensor = None
        self.output_tensor = None
        self.input_shape = input_shape
        self.weights = 0

    def forward(self, input_tensor):
        pass

    def backward(self, gradient_tensor, lr):
        pass

    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights


class CNNLayer(Layer):
    def __init__(self, input_shape: tuple[int], kernel_size: int, depth: int):
        super().__init__(input_shape)

        self.opt = Adam()

        input_depth = input_shape[0]
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
        self.output_shape = self.get_output_shape()

        n = input_depth * kernel_size * kernel_size
        self.weights = np.random.randn(*self.kernel_shape) * np.sqrt(2. / n)
        self.bias = np.random.randn(*self.output_shape)

    def get_output_shape(self):
        d = self.kernel_shape[0]
        h = self.input_shape[1] - self.kernel_shape[2] + 1
        w = self.input_shape[2] - self.kernel_shape[3] + 1
        return (d, h, w)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        ker_depth = self.kernel_shape[0]
        input_depth = self.kernel_shape[1]

        self.output_tensor = np.zeros(self.output_shape)
        for i in range(ker_depth):
            for j in range(input_depth):
                self.output_tensor[i] += correlate(input_tensor[j], self.weights[i][j], mode='valid')
            self.output_tensor[i] += self.bias[i]

        return self.output_tensor

    def backward(self, gradient_tensor, lr):

        assert gradient_tensor.shape == self.output_shape

        self.opt.learning_rate = lr

        ker_depth = self.kernel_shape[0]
        input_depth = self.kernel_shape[1]

        ker_grad_tensor = np.zeros(self.kernel_shape)
        input_grad_tensor = np.zeros(self.input_tensor.shape)

        for i in range(ker_depth):
            for j in range(input_depth):
                ker_grad_tensor[i][j] = correlate(self.input_tensor[j], gradient_tensor[i], mode='valid')
                input_grad_tensor[j] += convolve(gradient_tensor[i], self.weights[i][j], mode='full')

        # self.weights -= self.opt.update(self.weights, ker_grad_tensor)
        # self.bias -= lr * gradient_tensor

        self.weights -= lr * ker_grad_tensor
        self.bias -= lr * gradient_tensor

        return input_grad_tensor


class DenseLayer(Layer):
    def __init__(self, input_shape, neurons):
        self.weights = np.random.randn(neurons, input_shape[0]) * np.sqrt(2. / input_shape[0])
        self.bias = np.random.randn(neurons, 1)
        self.opt = Adam()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.dot(self.weights, self.input_tensor) + self.bias

        return self.output_tensor
    
    def backward(self, gradient_tensor, lr):
        assert gradient_tensor.shape == self.output_tensor.shape

        self.opt.learning_rate = lr

        input_grad_tensor = np.dot(self.weights.T, gradient_tensor)
        weight_grad_tensor = np.dot(gradient_tensor, self.input_tensor.T)

        # self.weights -= self.opt.update(self.weights, weight_grad_tensor)
        # self.bias -= lr * gradient_tensor

        self.weights -= lr * weight_grad_tensor
        self.bias -= lr * gradient_tensor

        return input_grad_tensor
    
    def get_output_shape(self):
        return self.bias.shape


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        depth, height, width = input_tensor.shape
        output_tensor = np.zeros((depth, height // self.kernel_size, width // self.kernel_size))

        for i in range(depth):
            for j in range(height // self.kernel_size):
                for k in range(width // self.kernel_size):
                    output_tensor[i, j, k] = np.max(input_tensor[i, j * self.kernel_size:(j + 1) * self.kernel_size, 
                                                                 k * self.kernel_size:(k + 1) * self.kernel_size])

        self.max_values = output_tensor
        return output_tensor

    def backward(self, gradient_tensor, lr):
        assert gradient_tensor.shape == self.get_output_shape()

        upsampled_gradient = gradient_tensor.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)
        max_mask = self.input_tensor == self.max_values.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)
        back_tensor = upsampled_gradient * max_mask

        return back_tensor
    
    def get_output_shape(self):
        depth, height, width = self.input_shape
        return (depth, height // self.kernel_size, width // self.kernel_size)


class ReshapeLayer(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape)

        if output_shape == 'flatten':
            output_shape = (input_shape[0] * input_shape[1] * input_shape[2], 1)
        self.output_shape = output_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.reshape(self.input_tensor, self.output_shape)
        return self.output_tensor

    def backward(self, gradient_tensor, lr):
        assert gradient_tensor.shape == self.output_shape

        return np.reshape(gradient_tensor, self.input_shape)
    
    def get_output_shape(self):
        return self.output_shape


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        try:
            self.activation = ACTIVATION[activation]
        except KeyError:
            raise KeyError(f'Activation function {activation} not implemented')
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = self.activation['forward'](input_tensor)
        return self.output_tensor
    
    def backward(self, gradient_tensor, lr=0):
        assert gradient_tensor.shape == self.input_tensor.shape

        return self.activation['backward'](gradient_tensor, self.input_tensor)
    
    def get_output_shape(self):
        return self.input_shape


class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.mask = np.random.binomial(1, 1 - self.rate, size=input_tensor.shape) / (1 - self.rate)
        return input_tensor * self.mask

    def backward(self, error_tensor, lr=0):
        return error_tensor * self.mask
    
    def get_output_shape(self):
        return self.input_shape
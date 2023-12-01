import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d
import numpy as np
from sklearn.model_selection import train_test_split


def correlate(matrix1: np.array, matrix2:np.array, mode='valid') -> np.array:
    """
    Se espera implementar una versión propia de la función correlate2d de scipy.signal
    para comprender el funcionamiento de esta operación.
    """
    return correlate2d(matrix1, matrix2, mode=mode)


def convolve(matrix1: np.array, matrix2:np.array, mode='valid') -> np.array:
    """
    Se espera implementar una versión propia de la función convolve2d de scipy.signal
    para comprender el funcionamiento de esta operación.
    """
    return convolve2d(matrix1, matrix2, mode=mode)


def preprocess(x_train, y_train, x_test, y_test, digits=False):

    if not digits:    
        x_train_channels = x_train.transpose(0, 3, 1, 2)
        x_test_channels = x_test.transpose(0, 3, 1, 2)
    else:
        x_train_channels = np.array([np.array([x]) for x in x_train])
        x_test_channels = np.array([np.array([x]) for x in x_test])

    y_train_encoded = np.zeros((y_train.size, y_train.max() + 1))
    y_train_encoded[np.arange(y_train.size), y_train.flatten()] = 1

    y_test_encoded = np.zeros((y_test.size, y_test.max() + 1))
    y_test_encoded[np.arange(y_test.size), y_test.flatten()] = 1

    x_train_channels = x_train_channels / 255
    x_test_channels = x_test_channels / 255

    # Make validation set
    x_train_channels, x_val, y_train_encoded, y_val = train_test_split(x_train_channels, y_train_encoded, 
                                                                       test_size=0.2, random_state=42)

    return x_train_channels, y_train_encoded, x_test_channels, y_test_encoded, x_val, y_val


if __name__ == '__main__':
    a = np.array([[
      [  20,  200,   -5,   23],
      [ -13,  134,  119,  100],
      [ 120,   32,   49,   25],
      [-120,   20,    9,   24]
    ]])
    
    from nn_layers import MaxPoolLayer

    max_pool_layer = MaxPoolLayer(2)

    f = max_pool_layer.forward(a)
    print(f)
    b = max_pool_layer.backward(f, 0.1)
    print(b)
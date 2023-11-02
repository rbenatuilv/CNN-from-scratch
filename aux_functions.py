import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d
from torch.nn.functional import max_pool2d


def correlate(matrix1: np.array, matrix2:np.array, mode='valid') -> np.array:
    """
    Se espera implementar una versión propia de la función correlate2d de scipy.signal
    para comprender el funcionamiento de esta operación.
    """

    return correlate2d(matrix1, matrix2, mode=mode)

def max_pool(image: np.array, kernel_size: int) -> np.array:
    """
    Se espera implementar una versión propia de la función max_pool2d de pytorch
    para comprender el funcionamiento de esta operación.
    """
    
    return np.array(max_pool2d(image, kernel_size=kernel_size))

import numpy as np
from scipy.signal import correlate2d
from scipy.signal import convolve2d
import numpy as np
from skimage.measure import block_reduce



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
    
    depth, height, width = image.shape
    output = np.zeros((depth, height // kernel_size, width // kernel_size))
    for i in range(depth):
        for j in range(height // kernel_size):
            for k in range(width // kernel_size):
                output[i][j][k] = np.max(image[i][j * kernel_size:(j + 1) * kernel_size, k * kernel_size:(k + 1) * kernel_size])

    return output
    
if __name__ == '__main__':
    a = np.array([[
      [  20,  200,   -5,   23],
      [ -13,  134,  119,  100],
      [ 120,   32,   49,   25],
      [-120,   20,    9,   24],
    ]])
    

    print(max_pool(a, 2))

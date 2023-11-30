import numpy as np


def cross_entropy_loss(prediction, label):
    """
    Implementación de la función de pérdida de entropía cruzada.
    """
    return -np.sum(np.multiply(label.T, np.log(np.clip(prediction, 1e-15, 1))))

def delta_cross_entropy(prediction, label):
    """
    Derivada de la función de pérdida de entropía cruzada.
    """
    label = label.reshape(prediction.shape)

    return prediction - label

def mse_loss(prediction, label):
    """
    Implementación de la función de pérdida de error cuadrático medio.
    """
    return np.sum(np.power(prediction - label, 2)) / prediction.shape[0]

def delta_mse(prediction, label):
    """
    Derivada de la función de pérdida de error cuadrático medio.
    """
    return 2 * (prediction - label) / prediction.shape[0]

LOSS = {
    'cross_entropy': {'loss': cross_entropy_loss, 'delta': delta_cross_entropy},
    'mse': {'loss': mse_loss, 'delta': delta_mse}
}

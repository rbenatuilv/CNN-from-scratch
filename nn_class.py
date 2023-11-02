from nn_layers import CNNLayer, MaxPoolLayer, Activation
from loss_functs import LOSS


class ConvNeuralNet:
    """
    Esta clase se encarga de definir la estructura de la red neuronal convolucional.
    """
    
    def __init__(self, layers, loss, lr):
        self.set_layers(layers)

        try:
            self.loss = LOSS[loss]
        except KeyError:
            raise KeyError(f'Loss function {loss} not implemented')
        
        self.lr = lr

    def set_layers(self, layers):
        """
        Esta función se encarga de inicializar las capas de la red neuronal.
        """
        self.layers = []
        for layer in layers:
            if layer['type'] == 'conv':
                self.layers.append(CNNLayer(layer['input_shape'], layer['kernel_size'], layer['depth']))
            elif layer['type'] == 'maxpool':
                self.layers.append(MaxPoolLayer(layer['kernel_size']))
            elif layer['type'] == 'activation':
                self.layers.append(Activation(layer['activation']))
            else:
                raise KeyError(f'Layer type {layer["type"]} not implemented')

    def fit(self, input_data, target_data, epochs, batch_size):
        """
        Esta función se encarga de entrenar la red neuronal.
        """
        # TODO: Implementar fit
        pass

    def predict(self, input_data):
        """
        Esta función se encarga de realizar la predicción de la red neuronal.
        """
        # TODO: Implementar predict
        pass

from nn_layers import CNNLayer, MaxPoolLayer, Activation, ReshapeLayer, DenseLayer
from loss_functs import LOSS
from tqdm import tqdm


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
        layer_types = {
            'conv': CNNLayer,
            'maxpool': MaxPoolLayer,
            'activation': Activation,
            'reshape': ReshapeLayer,
            'dense': DenseLayer
        }

        self.layers = []
        for elem in layers:
            layer_type, content = elem

            try:
                layer = layer_types[layer_type]
                self.layers.append(layer(**content))
            except KeyError:
                raise KeyError(f'Layer type {layer["type"]} not implemented')

    def fit(self, x_train, y_train, epochs):
        """
        Esta función se encarga de entrenar la red neuronal.
        """
        for e in range(epochs):
            error = 0
            for x, y in tqdm(zip(x_train, y_train)):
                pred = self.predict(x)
                error += self.loss['loss'](pred, y)
                gradient = self.loss['delta'](pred, y)

                for layer in reversed(self.layers):
                    # gradient = layer.backward(gradient, self.lr)
                    # NOTE: Implementar backwards
                    pass
            
            error /= len(x_train)
            print(f'Epoch {e + 1} error: {error}')
        
    def predict(self, input_data):
        """
        Esta función se encarga de realizar la predicción de la red neuronal.
        """
        output_pred = input_data
        for layer in self.layers:
            output_pred = layer.forward(output_pred)
        return output_pred


if __name__ == '__main__':
    from keras.datasets import cifar10
    import numpy as np
    

    layers = [
        ('conv',
            {
                'input_shape': (3, 32, 32),
                'kernel_size': 3,
                'depth': 5
            }
        ),
        ('maxpool',
            {
                'kernel_size': 2
            }
        ),
        ('activation',
            {
                'activation': 'relu'
            }
        ),
        ('reshape',
            {
                'input_shape': (5, 15, 15),
                'output_shape': ((5 * 15 * 15), 1)
            }
        ),
        ('dense',
            {
                'input_shape': ((5 * 15 * 15), 1),
                'neurons': 10
            }
        ),
        ('activation',
            {
                'activation': 'softmax'
            }
        )
    ]

    loss = 'cross_entropy'
    lr = 0.02

    model = ConvNeuralNet(layers, loss, lr)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train_channels = x_train.transpose(0, 3, 1, 2)
    x_test_channels = x_test.transpose(0, 3, 1, 2)

    y_train_encoded = np.zeros((y_train.size, y_train.max() + 1))
    y_train_encoded[np.arange(y_train.size), y_train.flatten()] = 1

    y_test_encoded = np.zeros((y_test.size, y_test.max() + 1))
    y_test_encoded[np.arange(y_test.size), y_test.flatten()] = 1

    # Normalize
    x_train_channels = x_train_channels / 255
    x_test_channels = x_test_channels / 255

    example = x_train_channels[0]
    # pred = model.predict(example)

    model.fit(x_train_channels, y_train_encoded, 1)

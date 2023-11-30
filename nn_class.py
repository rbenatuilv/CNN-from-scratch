from nn_layers import CNNLayer, MaxPoolLayer, Activation, ReshapeLayer, DenseLayer
from loss_functs import LOSS
from tqdm import tqdm
from aux_functions import preprocess
import math


class ConvNeuralNet:
    """
    Esta clase se encarga de definir la estructura de la red neuronal convolucional.
    """
    
    def __init__(self, layers, loss, lr, input_shape):

        self.input_shape = input_shape
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

        input_shape = self.input_shape
        self.layers = []

        for elem in tqdm(layers, desc='Setting layers:', total=len(layers)):
            layer_type, content = elem
            content['input_shape'] = input_shape

            try:
                layer = layer_types[layer_type]
                self.layers.append(layer(**content))
                input_shape = self.layers[-1].get_output_shape()

            except KeyError:
                raise KeyError(f'Layer type {layer["type"]} not implemented')

    def fit(self, x_train, y_train, x_val, y_val, epochs, patience=2, decay_rate=1):
        """
        Esta función se encarga de entrenar la red neuronal.
        """
        best_error = float('inf')
        best_weights = None
        no_improvement_epochs = 0

        train_errors = []
        val_errors = []

        for e in range(epochs):
            train_error = 0
            for x, y in tqdm(zip(x_train, y_train), desc=f'Epoch {e + 1}', total=len(x_train)):
                pred = self.predict(x)
                train_error += self.loss['loss'](pred, y)
                gradient = self.loss['delta'](pred, y)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.lr)
            
            train_error /= len(x_train)
            train_errors.append(train_error)

            val_error = np.mean([self.loss['loss'](self.predict(x), y) 
                                 for x, y in tqdm(zip(x_val, y_val), desc='Validation', total=len(x_val))])
            val_errors.append(val_error)

            print(f'Epoch {e + 1} train error: {train_error}, validation error: {val_error}\n')

            if val_error < best_error:
                best_error = val_error
                best_weights = self.get_weights()
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= patience:
                print(f'Early stopping on epoch {e + 1}')
                self.set_weights(best_weights)
                break

            self.lr *= math.exp(-decay_rate * (e + 1))

        return train_errors, val_errors
    
    def set_weights(self, weights):
        """
        Esta función se encarga de asignar los pesos a la red neuronal.
        """
        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)

    def get_weights(self):
        """
        Esta función se encarga de obtener los pesos de la red neuronal.
        """
        return [layer.get_weights() for layer in self.layers]


    def predict(self, input_data):
        """
        Esta función se encarga de realizar la predicción de la red neuronal.
        """
        output_pred = input_data
        for layer in self.layers:
            output_pred = layer.forward(output_pred)
        return output_pred
    
    def save(self, path):
        """
        Esta función se encarga de guardar los pesos de la red neuronal.
        """
        import pickle

        with open(path, 'wb') as f:
            pickle.dump([layer.get_weights() for layer in self.layers], f)


if __name__ == '__main__':
    from keras.datasets import cifar10
    import numpy as np
    import matplotlib.pyplot as plt
    

    layers = [
        ('conv', {'kernel_size': 3, 'depth': 5}),
        ('maxpool', {'kernel_size': 2}),
        ('activation', {'activation': 'relu'}),
        ('conv', {'kernel_size': 2, 'depth': 3}),
        ('maxpool', {'kernel_size': 2}),
        ('activation', {'activation': 'relu'}),
        ('reshape', {'output_shape': 'flatten'}),
        ('dense', {'neurons': 10}),
        ('activation', {'activation': 'softmax'})
    ]

    loss = 'cross_entropy'
    lr = 0.001
    input_shape = (3, 32, 32)

    model = ConvNeuralNet(layers, loss, lr, input_shape)

    if input('Load weights? [y/n] ') == 'y':
        import pickle

        with open('weights.pkl', 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train_channels, y_train_encoded, x_test_channels, y_test_encoded, x_val, y_val = preprocess(x_train, y_train, x_test, y_test)

    t_err, val_err = model.fit(x_train_channels, y_train_encoded, x_val, y_val, epochs=10, patience=3)
    model.save('weights.pkl')

    
    fig, ax = plt.subplots()
    ax.plot(t_err, label='Train error')
    ax.plot(val_err, label='Validation error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

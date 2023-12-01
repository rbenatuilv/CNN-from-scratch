from nn_layers import CNNLayer, MaxPoolLayer, Activation, ReshapeLayer, DenseLayer, Dropout
from loss_functs import LOSS
from tqdm import tqdm
import math
import numpy as np


class ConvNeuralNet:
    """
    Esta clase se encarga de definir la estructura de la red neuronal convolucional.
    """
    
    def __init__(self, layers, loss, lr, input_shape, name=None):

        self.input_shape = input_shape
        self.set_layers(layers)
        self.name = name

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
            'dense': DenseLayer,
            'dropout': Dropout
        }

        input_shape = self.input_shape
        self.layers = []

        for elem in tqdm(layers, desc='Setting layers', total=len(layers)):
            layer_type, content = elem
            content['input_shape'] = input_shape

            try:
                layer = layer_types[layer_type]
                self.layers.append(layer(**content))
                input_shape = self.layers[-1].get_output_shape()

            except KeyError:
                raise KeyError(f'Layer type {layer["type"]} not implemented')

    def fit(self, x_train, y_train, validation_data, epochs, patience=2, decay_rate=0.01, verbose=True):
        """
        Esta función se encarga de entrenar la red neuronal.
        """
        x_val, y_val = validation_data

        best_error = float('inf')
        best_weights = None
        no_improvement_epochs = 0

        train_errors = [0]
        val_errors = [0]

        for e in range(epochs):
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            train_error = 0
            for x, y in tqdm(zip(x_train, y_train), desc=f'Epoch {e + 1}', total=len(x_train), disable=not verbose):
                pred = self.predict(x)
                train_error += self.loss['loss'](pred, y)
                gradient = self.loss['delta'](pred, y)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.lr)
            
            train_error /= len(x_train)

            if train_error > train_errors[-1]:
                self.lr *= math.exp(-decay_rate * (e + 1))

            train_errors.append(train_error)

            val_error = np.mean([self.loss['loss'](self.predict(x), y) 
                                 for x, y in tqdm(zip(x_val, y_val), desc='Validation', 
                                                  total=len(x_val), disable=not verbose)])
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
        
        train_errors = train_errors[1:]
        val_errors = val_errors[1:]

        return train_errors, val_errors
    
    def predict(self, input_data):
        """
        Esta función se encarga de realizar la predicción de la red neuronal.
        """
        output_pred = input_data
        for layer in self.layers:
            output_pred = layer.forward(output_pred)
        return output_pred
    
    def check_precision(self, x_test, y_test):
        """
        Esta función se encarga de calcular la precisión de la red neuronal.
        """
        correct = 0
        for x, y in tqdm(zip(x_test, y_test), desc='Testing', total=len(x_test)):
            pred = self.predict(x)
            if np.argmax(pred) == np.argmax(y):
                correct += 1
        return correct / len(x_test)

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
    
    def save(self):
        """
        Esta función se encarga de guardar los pesos de la red neuronal.
        """

        if self.name is None:
            path = 'weights.pkl'
        else:
            path = f'weights_{self.name}.pkl'

        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self.get_weights(), f)

    def load(self):
        """
        Esta función se encarga de cargar los pesos de la red neuronal.
        """
        if self.name is None:
            path = 'weights.pkl'
        else:
            path = f'weights_{self.name}.pkl'

        import pickle

        with open(path, 'rb') as f:
            weights = pickle.load(f)
            self.set_weights(weights)

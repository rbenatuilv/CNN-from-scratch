from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from nn_class import ConvNeuralNet
from aux_functions import preprocess
import math
# Import mnist dataset
from keras.datasets import mnist


layers = [
    ('conv', {'kernel_size': 7, 'depth': 5}),
    ('dropout', {'rate': 0.5}),
    ('maxpool', {'kernel_size': 2}),
    ('activation', {'activation': 'relu'}),
    ('conv', {'kernel_size': 4, 'depth': 3}),
    ('dropout', {'rate': 0.5}),
    ('maxpool', {'kernel_size': 2}),
    ('activation', {'activation': 'relu'}),
    ('reshape', {'output_shape': 'flatten'}),
    ('dense', {'neurons': 10}),
    ('activation', {'activation': 'softmax'})
]

loss = 'cross_entropy'
lr = 0.1
input_shape = (3, 32, 32)

model = ConvNeuralNet(layers, loss, lr, input_shape)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_channels, y_train_encoded, x_test_channels, y_test_encoded, x_val, y_val = preprocess(x_train, y_train, x_test, y_test, digits=False)

# print(x_train_channels[0].shape)

x_train = x_train_channels[:1]
y_train = y_train_encoded[:1]

prev_error = 0
for _ in range(1000):
    train_error = 0
    for x, y in zip(x_train, y_train):
        pred = model.predict(x)
        train_error += model.loss['loss'](pred, y)

        gradient = model.loss['delta'](pred, y)
        for layer in reversed(model.layers):
            gradient = layer.backward(gradient, model.lr)

    train_error /= len(x_train)

    # step decay
    if train_error > prev_error:
        model.lr *= math.exp(-0.1)

    prev_error = train_error

    print(train_error)
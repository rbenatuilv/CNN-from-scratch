from keras.datasets import cifar10
import matplotlib.pyplot as plt
from nn_class import ConvNeuralNet
from aux_functions import preprocess


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
lr = 0.1
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

print(model.check_precision(x_test_channels, y_test_encoded))
if input('Save weights? [y/n] ') == 'y':
    model.save('weights.pkl')



fig, ax = plt.subplots()
ax.plot(t_err, label='Train error')
ax.plot(val_err, label='Validation error')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.legend()
plt.show()

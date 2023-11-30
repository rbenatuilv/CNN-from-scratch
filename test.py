from nn_layers import CNNLayer, MaxPoolLayer, Activation, ReshapeLayer, DenseLayer
import pickle


# Load weights
with open('weights.pkl', 'rb') as f:
    weights = pickle.load(f)

print(weights)
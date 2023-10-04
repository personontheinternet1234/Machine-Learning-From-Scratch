# NeuralNetwork.py

import numpy as np
import random
from tqdm import tqdm  # maybe won't use

# classes and definitions

def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def sigmoid_prime(values):
    # output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


def relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def relu_prime(values):
    return np.where(values > 0, 1, 0.1)


# def make_vertical(vector):
#     ...


# network presets

# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["1", "2"]

# neural network structure
layer_sizes = [2, 3, 3, 2]

# learning presets
learn = True  # add this functionality, add ability to choose original weights and biases
non_linearity = "sigmoid"  # add this functionality
error_analysis = "SSR"  # add this functionality
error_report_type = "SSR"  # add this functionality
epochs = 100
return_rate = 1
learning_rate = 0.01

# if set network
set_weights = [

]

set_biases = [

]

# training data set
input_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_training = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

# network formation

layers = len(layer_sizes)
# if learn: indent afterwards
# instantiate weights and biases
weights = []
biases = []

for i in range(layers - 1):
    weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]))
    biases.append(np.zeros((layer_sizes[i + 1], 1)))

# training loop
for epoch in range(epochs):
    # choose from training set
    training_choice = random.randint(0, len(input_training) - 1)

    # reformat inputs and outputs
    activation = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    expected_values = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))

    # forward pass
    activations = [activation]
    for layer in range(layers - 1):
        activation = relu(np.matmul(weights[layer], activation) + biases[layer])
        activations.append(activation)

    # calculate gradients
    d_activations = []
    d_weights = []
    d_biases = []
    
    # error with respect to last layer
    d_activations.insert(0, -2 * np.subtract(expected_values, activations[-1]))

    for layer in range(layers - 1, 0, -1):  # start at last hidden layer, go back until layer = 0
        # gradient of biases
        print(activations[layer])
        d_b = relu_prime(np.matmul(weights[layer - 1], activations[layer - 1]) + biases[layer - 1] * d_activations[0])
        d_biases.insert(0, d_b)

        # gradient of weights
        x = np.resize(biases[0], (len(activations[layer - 1]), len(activations[layer]))).T,
        y = np.resize(activations[layer - 1].T, (len(activations[layer]), len(activations[layer - 1])))
        d_w = np.multiply(x, y)
        d_weights.insert(0, d_w)

        # gradient of activation from before last layer
        upstream = np.resize(d_biases[0], (len(activations[layer - 1]), len(activations[layer])))
        totals = np.sum(np.multiply(upstream, weights[layer - 1].T), axis=1)
        d_a = np.reshape(totals, (len(activations[layer - 1]), 1))
        d_activations.insert(0, d_a)

    # error report
    if epoch % return_rate == 0:
        error = 0
        for j in range(len(activations[-1])):
            error += (expected_values[j] - activations[-1][j]) ** 2
        print(f"({round((epoch / epochs) * 100)}%) Error: {error}")

# else
if not learn:
    # use set weights and biases
    weights = set_weights
    biases = set_biases

# finalized network application

while True:
    # get inputs
    print("")
    inputs = []
    for activation in range(layer_sizes[0]):
        inputs.append(float(input(f"{input_index[activation]}: ")))

    # forward pass
    activations = np.reshape(inputs, (len(inputs), 1))
    for layer in range(layers - 1):
        activations = relu(np.matmul(weights[layer], activations) + biases[layer])

    # result
    print("")
    print(activations)
    output_number = np.nanargmax(np.where(activations == activations.max(), activations, np.nan))
    print(f"Predicted: {output_index[output_number]}")

# yay!

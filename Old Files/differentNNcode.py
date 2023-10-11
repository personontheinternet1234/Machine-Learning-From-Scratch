import random

import numpy as np

# functions

def leaky_relu(inputs):
    output = np.maximum(0.1 * inputs, inputs)
    return output

def leaky_relu_prime(inputs):
    output = np.where(inputs > 0, 1, 0.1)
    return output

def xavier_initialize(length, width):
    matrix = np.random.randn(length, width) * np.sqrt(2 / length)
    return matrix

def zeros_initialize(length, width):
    matrix = np.zeros((length, width))
    return matrix

def forward_pass(inputs, weights, biases):
    activations = [inputs]
    for layer in range(len(weights)):
        activations.append(leaky_relu(np.matmul(weights[layer], activations[-1]) + biases[layer]))
    return activations

# def gradient(values, activations, weights, biases):
#     ...

def optimize(initial, gradient, learning_rate):
    output = []
    for connection in range(len(initial)):
        output.append(np.subtract(initial[connection], np.multiply(learning_rate, gradient[connection])))
    return output

def make_vector(list):
    vector = np.reshape(np.array(list), (len(list), 1))
    return vector

def argmax(inputs):
    index = np.nanargmax(np.where(inputs == inputs.max(), inputs, np.nan))
    return index

def calculate_error(expected, actual):
    error = np.sum(np.subtract(expected, actual) ** 2, axis=0)
    return error

# network presets

# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["checkered", "non-checkered"]

# learning presets
learn = True
load = True
save = False
epochs = 1
return_rate = 1000
learning_rate = 0.01

# neural network structure
layer_sizes = [2, 3, 2]

# if set network
set_weights = [
    np.array([[1,1],
              [1,1],
              [1,1]]),
    np.array([[1, 1, 1],
              [1, 1, 1]])
]

set_biases = [
    np.array([[0],
              [0],
              [0]]),
    np.array([[0],
              [0]]),
]

# training data set
input_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_training = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
]

# network formation

input('Press "Enter" to start.')

layers = len(layer_sizes)
if learn:
    # instantiate weights and biases
    weights = []
    biases = []
    if load:
        # use set weights and biases
        weights = set_weights
        biases = set_biases
    else:
        # use random weights and zeros for biases
        for connection in range(layers - 1):
            weights.append(xavier_initialize(layer_sizes[connection + 1], layer_sizes[connection]))
            biases.append(zeros_initialize(layer_sizes[connection + 1], 1))
    
    # training loop
    for epoch in range(epochs):
        # choose from training set
        training_choice = random.randint(0, len(input_training) - 1)  # maybe optimize?
        training_choice = 1
        
        # reformat inputs and outputs
        activation = make_vector(input_training[training_choice])
        expected_values = make_vector(output_training[training_choice])
        
        # forward pass
        activations = forward_pass(activation, weights, biases)
        
        # calculate gradients  # make function
        d_activations = []
        d_weights = []
        d_biases = []
        
        d_activations.insert(0, np.multiply(-2, np.subtract(expected_values, activations[-1])))
        
        for connection in range(layers - 1):
            d_bias = leaky_relu_prime(np.matmul(weights[-connection - 1], activations[-connection - 2]) + biases[-connection - 1]) * d_activations[0]
            d_biases.insert(0, d_bias)
            d_weight = np.multiply(np.resize(d_biases[0], (layer_sizes[-connection - 2], layer_sizes[-connection - 1])).T, np.resize(activations[-connection - 2], (layer_sizes[-connection - 1], layer_sizes[-connection - 2])))
            d_weights.insert(0, d_weight)
            d_activation = np.reshape(np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[-connection - 2], layer_sizes[-connection - 1])), weights[-connection - 1].T), axis=1), (layer_sizes[-connection - 2], 1))
            d_activations.insert(0, d_activation)
        
        # optimize weights and biases
        weights = optimize(weights, d_weights, learning_rate)
        biases = optimize(biases, d_biases, learning_rate)
        print(d_weights)
        print(d_biases)
        
        # error report
        if epoch % return_rate == 0:
            error = 0
            for test_case in range(len(input_training)):
                expected_values = make_vector(output_training[test_case])
                actual_values = forward_pass(make_vector(input_training[test_case]), weights, biases)[-1]

                error += calculate_error(expected_values, actual_values)
            print(f"({round((epoch / epochs) * 100)}%) MSE: {error[0] / len(input_training)}")
else:
    # instantiate weights and biases
    weights = []
    biases = []
    if load:
        # use set weights and biases
        weights = set_weights
        biases = set_biases
    else:
        # use random weights and zeros for biases
        for connection in range(layers - 1):
            weights.append(xavier_initialize(layer_sizes[connection + 1], layer_sizes[connection]))
            biases.append(zeros_initialize(layer_sizes[connection + 1], 1))

# finalized network application

# print optimized weights and biases
if save:
    for connection in range(len(weights)):
        print("")
        print(f"Weights (layer {connection}):")
        print(weights[connection])
        print(f"Bias (layer {connection}):")
        print(biases[connection])

# print final error
error = 0
for test_case in range(len(input_training)):
    expected_values = make_vector(output_training[test_case])
    actual_values = forward_pass(make_vector(input_training[test_case]), weights, biases)[-1]
    error += calculate_error(expected_values, actual_values)
print("")
print(f"Final MSE: {error[0] / len(input_training)}")

# user input
while True:
    # get inputs
    print("")
    inputs = []
    for node in range(layer_sizes[0]):
        inputs.append(float(input(f"{input_index[node]}: ")))
    
    # forward pass
    inputs = make_vector(inputs)
    results = (forward_pass(inputs, weights, biases))[-1]
    
    # result
    print("")
    print(results)
    index = argmax(results)
    print(f"Predicted: {output_index[index]}")

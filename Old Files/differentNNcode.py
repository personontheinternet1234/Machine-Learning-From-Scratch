""" neural network without libraries
    Isaac Park Verbrugge & Christian Host-Madsen """

import ast
import random

import numpy as np

""" functions """


def leaky_relu(inputs):
    # leaky relu equation
    output = np.maximum(0.1 * inputs, inputs)
    return output


def leaky_relu_prime(inputs):
    # leaky relu' equation
    output = np.where(inputs > 0, 1, 0.1)
    return output


def xavier_initialize(length, width):
    # xavier initialization
    array = np.random.randn(length, width) * np.sqrt(2 / length)
    return array


def zeros_initialize(length, width):
    # zeros initialization
    array = np.zeros((length, width))
    return array


def forward_pass(inputs, weights, biases):
    # pass inputs through network
    activations = [inputs]
    for layer in range(len(weights)):
        activations.append(leaky_relu(np.matmul(weights[layer], activations[-1]) + biases[layer]))
    return activations


def find_gradients(expected, activations, weights, biases):
    # initialize gradient lists
    d_activations = []
    d_weights = []
    d_biases = []
    layers = len(activations)

    # calculate gradients
    d_activations.insert(0, np.multiply(-2, np.subtract(expected, activations[-1])))
    for connection in range(layers - 2):
        d_biases.insert(0, leaky_relu_prime(np.matmul(weights[-connection - 1], activations[-connection - 2]) + biases[-connection - 1]) * d_activations[0])
        d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[-connection - 2], layer_sizes[-connection - 1])).T, np.resize(activations[-connection - 2], (layer_sizes[-connection - 1], layer_sizes[-connection - 2]))))
        d_activations.insert(0, np.reshape(np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[-connection - 2], layer_sizes[-connection - 1])), weights[-connection - 1].T), axis=1), (layer_sizes[-connection - 2], 1)))
    d_biases.insert(0, leaky_relu_prime(np.matmul(weights[0], activations[0]) + biases[0]) * d_activations[0])
    d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[0], layer_sizes[1])).T, np.resize(activations[0], (layer_sizes[1], layer_sizes[0]))))

    # return gradients
    gradients = {
        "d_weights": d_weights,
        "d_biases": d_biases,
    }
    return gradients


def optimize(initial, gradient, learning_rate):
    # adjust weights and biases
    output = []
    for connection in range(len(initial)):
        output.append(np.subtract(initial[connection], np.multiply(learning_rate, gradient[connection])))
    return output


def make_vector(list):
    # turn a list into a vector
    vector = np.reshape(np.array(list), (len(list), 1))
    return vector


def argmax(inputs):
    # argmax
    index = np.nanargmax(np.where(inputs == inputs.max(), inputs, np.nan))
    return index


def calculate_error(expected, actual):
    # SSR calculation
    error = np.sum(np.subtract(expected, actual) ** 2, axis=0)
    return error


""" network presets """

# user indexes
input_index = ["a(0)0", "a(0)1"]
output_index = ["checkered", "non-checkered"]

# learning presets
learn = True
load = False
save = False
epochs = 100000
return_rate = 1000
learning_rate = 0.1

# neural network structure
layer_sizes = [2, 3, 2]

# training set
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

""" network formation """

input('Press "Enter" to start.')

layers = len(layer_sizes)
if learn:
    # instantiate weights and biases
    weights = []
    biases = []
    if load:
        # load weights and biases
        with open("saved/weights", "r") as weights_f:
            for line in weights_f:
                weights.append(np.array(ast.literal_eval(line)))
        with open("saved/biases", "r") as biases_f:
            for line in biases_f:
                biases.append(make_vector(ast.literal_eval(line)))
    else:
        # generate weights and biases
        for connection in range(layers - 1):
            weights.append(xavier_initialize(layer_sizes[connection + 1], layer_sizes[connection]))
            biases.append(zeros_initialize(layer_sizes[connection + 1], 1))

    # format training data
    for data in range(len(input_training)):
        input_training[data] = make_vector(input_training[data])
    for data in range(len(output_training)):
        output_training[data] = make_vector(output_training[data])

    # begin training loop
    for epoch in range(epochs):
        # choose from training set
        training_choice = random.randint(0, len(input_training) - 1)

        # set inputs and outputs
        activation = input_training[training_choice]
        expected = output_training[training_choice]

        # forward pass
        activations = forward_pass(activation, weights, biases)

        # calculate gradients
        gradients = find_gradients(expected, activations, weights, biases)

        # optimize weights and biases
        weights = optimize(weights, gradients["d_weights"], learning_rate)
        biases = optimize(biases, gradients["d_biases"], learning_rate)

        # error report
        if epoch % return_rate == 0:
            error = 0
            for test_case in range(len(input_training)):
                expected = output_training[test_case]
                actual_values = forward_pass(input_training[test_case], weights, biases)[-1]
                error += calculate_error(expected, actual_values)
            print(f"({round((epoch / epochs) * 100)}%) MSE: {error[0] / len(input_training)}")

    # print final error
    error = 0
    for test_case in range(len(input_training)):
        expected = output_training[test_case]
        actual_values = forward_pass(input_training[test_case], weights, biases)[-1]
        error += calculate_error(expected, actual_values)
    print("")
    print(f"Final MSE: {error[0] / len(input_training)}")
else:
    # instantiate weights and biases
    weights = []
    biases = []
    if load:
        # load weights and biases
        with open("saved/weights", "r") as weights_f:
            for line in weights_f:
                weights.append(np.array(ast.literal_eval(line)))
        with open("saved/biases", "r") as biases_f:
            for line in biases_f:
                biases.append(make_vector(ast.literal_eval(line)))
    else:
        # generate weights and biases
        for connection in range(layers - 1):
            weights.append(xavier_initialize(layer_sizes[connection + 1], layer_sizes[connection]))
            biases.append(zeros_initialize(layer_sizes[connection + 1], 1))
if save:
    # save optimized weights and biases
    with open("saved/weights", "w") as file:
        for array in range(len(weights)):
            file.write(str(weights[array].tolist()) + "\n")
    with open("saved/biases", "w") as file:
        for array in range(len(biases)):
            file.write(str(biases[array].tolist()) + "\n")

""" finalized network application """

while True:
    # get inputs
    print("")
    inputs = []
    for node in range(layer_sizes[0]):
        inputs.append(float(input(f"{input_index[node]}: ")))

    # forward pass
    inputs = make_vector(inputs)
    results = (forward_pass(inputs, weights, biases))[-1]

    # return result
    print("")
    print(results)
    index = argmax(results)
    print(f"Predicted: {output_index[index]}")

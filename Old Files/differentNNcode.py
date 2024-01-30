""" neural network without libraries
    Isaac Park Verbrugge & Christian Host-Madsen """

import ast
import random
import time

import numpy as np

""" functions """


def leaky_relu(inputs):
    # leaky relu equation
    output = np.maximum(0.1 * inputs, inputs)
    return output


def leaky_relu_prime(inputs):
    # leaky relu equation
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
    for connection in range(-1, -layers + 1, -1):
        d_biases.insert(0, leaky_relu_prime(np.matmul(weights[connection], activations[connection - 1]) + biases[connection]) * d_activations[0])
        d_weights.insert(0, np.multiply(np.resize(d_biases[0], (layer_sizes[connection - 1], layer_sizes[connection])).T, np.resize(activations[connection - 1], (layer_sizes[connection], layer_sizes[connection - 1]))))
        d_activations.insert(0, np.reshape(np.sum(np.multiply(np.resize(d_biases[0], (layer_sizes[connection - 1], layer_sizes[connection])), weights[connection].T), axis=1), (layer_sizes[connection - 1], 1)))
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
    layer_size = len(initial)
    for connection in range(layer_size):
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
return_rate = 10000
learning_rate = 0.1

# neural network structure
layer_sizes = [2, 3, 2]

# file locations
input_data_location = "data/input_data.txt"
output_data_location = "data/output_data.txt"


""" network formation """

input('Press "Enter" to start.')
start_time = time.time()

# load training set
input_training = []
output_training = []
with open("data/input_data.txt", "r") as f:
    for line in f:
        input_training.append(ast.literal_eval(line))
with open("data/output_data.txt", "r") as f:
    for line in f:
        output_training.append(ast.literal_eval(line))

# set values
layers = len(layer_sizes)
input_len = len(input_training)
output_len = len(output_training)

# format training data
for data in range(input_len):
    input_training[data] = make_vector(input_training[data])
for data in range(output_len):
    output_training[data] = make_vector(output_training[data])

# instantiate weights and biases
weights = []
biases = []
if load:
    # load weights and biases
    with open("saved/weights.txt", "r") as f:
        for line in f:
            weights.append(np.array(ast.literal_eval(line)))
    with open("saved/biases.txt", "r") as f:
        for line in f:
            biases.append(make_vector(ast.literal_eval(line)))
    weights_len = len(weights)
else:
    # generate weights and biases
    for connection in range(layers - 1):
        weights.append(xavier_initialize(layer_sizes[connection + 1], layer_sizes[connection]))
        biases.append(zeros_initialize(layer_sizes[connection + 1], 1))

if learn:
    # begin training loop
    for epoch in range(epochs):
        # choose from training set
        training_choice = random.randint(0, input_len - 1)

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

        # loss report
        if epoch % return_rate == 0:
            error = 0
            for test_case in range(input_len):
                expected = output_training[test_case]
                actual_values = forward_pass(input_training[test_case], weights, biases)[-1]
                error += calculate_error(expected, actual_values)
            print(f"{round((epoch / epochs) * 100)}% - MSE: {error[0] / input_len}")

end_time = time.time()
# calculate loss
error = 0
for test_case in range(input_len):
    expected = output_training[test_case]
    actual_values = forward_pass(input_training[test_case], weights, biases)[-1]
    error += calculate_error(expected, actual_values)
# calculate accuracy
correct = 0
for test_case in range(input_len):
    if argmax(output_training[test_case]) == argmax(forward_pass(input_training[test_case], weights, biases)[-1]):
        correct += 1
# return results
print("")
print(f"Results - Loss: {error[0] / input_len} - Elapsed Time: {end_time - start_time}s - Accuracy: {correct / len(input_training) * 100}%")

if save:
    # save optimized weights and biases
    with open("saved/weights.txt", "w") as f:
        for array in range(len(weights)):
            f.write(str(weights[array].tolist()) + "\n")
    with open("saved/biases.txt", "w") as f:
        for array in range(len(biases)):
            f.write(str(biases[array].tolist()) + "\n")

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

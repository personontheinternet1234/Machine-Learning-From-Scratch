import random

import numpy as np


# NeuralNet.py
# Isaac Verbrugge & Christian Host-Madsen
# (program info)

# function definitions


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


def forward_pass(a0, w0, b0, w1, b1):
    # forward pass
    a1 = relu(np.matmul(w0, a0) + b0)
    a2 = relu(np.matmul(w1, a1) + b1)

    return a2


# def backpropagate():
# ...


# code

# user output index
user_outputs = ["1", "2"]

# neural network structure
input_size = 2
hidden_1_size = 3
output_size = 2

# learning presets
learning_rate = 0.1

# training processed_data
training_input = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.5, 0.5]
]
training_output = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0],
    [0.5, 0.5]
]

# instantiate weights and biases
w0 = np.array([[1, 1], [1, 1], [1, 1]])
b0 = np.array([[0], [0], [0]])
w1 = np.array([[1, 1, 1], [1, 1, 1]])
b1 = np.array([[0], [0]])

# training loop
for i in range(10000):
    # choose from training set
    training_choice = random.randint(0, len(training_input) - 1)

    # reformat inputs and outputs
    a0 = np.reshape(np.array(training_input[training_choice]), (len(training_input[training_choice]), 1))
    c = np.reshape(np.array(training_output[training_choice]), (len(training_output[training_choice]), 1))

    # calculate outputs
    a1 = relu(np.matmul(w0, a0) + b0)
    a2 = relu(np.matmul(w1, a1) + b1)

    # calculate gradients

    # second layer
    d_a2 = -2 * np.subtract(c, a2)
    d_b1 = relu_prime(np.matmul(w1, a1) + b1) * d_a2
    d_w1 = np.multiply(np.resize(d_b1, (hidden_1_size, output_size)).T, np.resize(a1, (output_size, hidden_1_size)))

    # first layer
    d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b1, (hidden_1_size, output_size)), w1.T), axis=1),
                      (hidden_1_size, 1))
    d_b0 = relu_prime(np.matmul(w0, a0) + b0) * d_a1
    d_w0 = np.multiply(np.resize(d_b0, (input_size, hidden_1_size)).T, np.resize(a0, (hidden_1_size, input_size)))

    # optimize weights and biases

    w0 = np.subtract(w0, learning_rate * d_w0)
    b0 = np.subtract(b0, learning_rate * d_b0)
    w1 = np.subtract(w1, learning_rate * d_w1)
    b1 = np.subtract(b1, learning_rate * d_b1)

    # error report every few epochs

    print(a2)
    print(c)
    print("----------")

while True:
    print("")
    inputs = []
    for i in range(input_size):
        inputs.append(float(input("a(0)" + str(i) + ": ")))  # gets inputs

    # forward pass
    a0 = np.reshape(inputs, (len(inputs), 1))
    a2 = forward_pass(a0, w0, b0, w1, b1)

    # result
    print("")
    print(a2)
    output_index = np.nanargmax(np.where(a2 == a2.max(), a2, np.nan))
    print(f"Predicted: {user_outputs[output_index]}")

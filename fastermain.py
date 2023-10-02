import numpy as np
import random
from tqdm import tqdm

actual_outputs = ["0", "1"]

# float_possibilities = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]


def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def sigmoid_prime(values):
    output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    return output


def relu(x):  # Leaky Relu function, with a slope of 0.1 for neg values built for use with matrices
    return np.maximum(0.1 * x, x)


def derivative_relu(scalar_passed_to_relu):  # derivative of our leaky relu
    return 1 if (scalar_passed_to_relu > 0) else 0.1


input_training = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]
output_training = [
    [1, 1],
    [0, 1],
    [1, 0],
    [0, 0]
]

# add hidden layers when necessary
input_size = 2
hidden_layers = 2  # update for this
hidden_sizes = [3, 2]  # update for this
hidden_1_size = 3
output_size = 2
epochs = 10000
learning_rate = 0.001

w0 = np.random.randn(hidden_1_size, input_size)
b1 = np.zeros((hidden_1_size, 1))
w1 = np.random.randn(output_size, hidden_1_size)  # going_to x from
b2 = np.zeros((output_size, 1))

for i in range(epochs):

    # forward pass

    # choose from training set
    training_choice = random.randint(0, len(input_training) - 1)

    # reformat inputs and outputs
    a0 = np.reshape(input_training[training_choice], (len(input_training[training_choice]), 1))
    c = np.reshape(output_training[training_choice], (len(output_training[training_choice]), 1))

    # calculate outputs
    a1 = sigmoid(np.matmul(w0, a0) + b1)
    a2 = sigmoid(np.matmul(w1, a1) + b2)

    # calculate gradients

    # second layer
    d_a2 = -2 * np.subtract(c, a2)
    d_b2 = sigmoid_prime(d_a2)
    d_w1 = np.multiply(np.resize(d_b2, (hidden_1_size, output_size)).T, np.resize(a1.T, (output_size, hidden_1_size)))

    # pseudo
    # [
    #     [1, 2, 4.5]
    #     [4.5, 2, 3]
    # ]
    #
    # [
    #     [1]
    #     [2]
    # ]
    # g_activation = connection.weight * g_upstream[connectionindex][0]

    # d_a1_0 = np.sum(np.multiply(w1[0], d_b2))

    # first layer
    d_a1 = np.reshape(np.sum(np.multiply(w1, d_b2), axis=0), (len(a1), 1))
    d_b1 = sigmoid_prime(d_a1)
    d_w0 = np.multiply(np.resize(d_b1, (input_size, hidden_1_size)).T, np.resize(a0.T, (hidden_1_size, input_size)))

    # optimize weights and biases

    w0 = np.subtract(w0, learning_rate * d_w0)
    b1 = np.subtract(b1, learning_rate * d_b1)
    w1 = np.subtract(w1, learning_rate * d_w1)
    b2 = np.subtract(b2, learning_rate * d_b2)

    # error report

    if i % 1000 == 0:
        error = 0
        for j in range(len(a2)):
            error += (c[j] - a2[j]) ** 2
        print("SSR: " + str(error))

# return optimized weights and biases
print("")
print("w0:")
print(w0)
print("b1:")
print(b1)
print("w1:")
print(w1)
print("b2:")
print(b2)

# allows user to plug in values
while True:
    print("")
    inputs = []
    for i in range(input_size):
        inputs.append(float(input("a(0)" + str(i) + ": ")))  # gets inputs

    # calculates
    a0 = np.reshape(inputs, (len(inputs), 1))
    a1 = sigmoid(np.dot(w0, a0) + b1)
    a2 = sigmoid(np.dot(w1, a1) + b2)

    # prints result
    print("")
    print(a2)
    output_index = np.nanargmax(np.where(a2 == a2.max(), a2, np.nan))
    print(actual_outputs[output_index])
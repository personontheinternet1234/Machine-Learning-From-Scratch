import numpy as np
import random
from tqdm import tqdm

def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def sigmoid_prime(values):
    output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
    return output


def relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def relu_prime(values):
    if values > 0:
        return 1
    else:
        return 0.1

# training data
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

# user output index
user_outputs = ["0", "1"]

# add hidden layers when necessary
input_size = 2
hidden_layers = 2  # update for this later
hidden_sizes = [3, 2]  # update for this later
hidden_1_size = 3
output_size = 2
epochs = 1
learning_rate = 0.001

# instantiate weights and biases
w0 = np.random.randn(hidden_1_size, input_size)
b1 = np.zeros((hidden_1_size, 1))
w1 = np.random.randn(output_size, hidden_1_size)
b2 = np.zeros((output_size, 1))

# training loop
for i in range(epochs):

    # forward pass # this works i think

    # choose from training set
    training_choice = random.randint(0, len(input_training) - 1)

    # reformat inputs and outputs
    a0 = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    c = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))
    
    # calculate outputs
    a1 = sigmoid(np.matmul(w0, a0) + b1)
    a2 = sigmoid(np.matmul(w1, a1) + b2)

    # calculate gradients

    # second layer
    d_a2 = -2 * np.subtract(c, a2)
    d_b2 = sigmoid_prime(np.matmul(w1, a1) + b2) * d_a2
    d_w1 = np.multiply(np.resize(d_b2, (hidden_1_size, output_size)).T, np.resize(a1.T, (output_size, hidden_1_size)))

    # first layer
    d_a1 = np.reshape( np.sum(np.multiply(np.resize(d_b2, (hidden_1_size, output_size)), w1.T), axis=1), (hidden_1_size, 1) )
    print(d_a1)
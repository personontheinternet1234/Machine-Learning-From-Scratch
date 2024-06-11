import numpy as np
import random
from tqdm import tqdm

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


def forward_pass(a0, w0, b1, w1, b2):
    # forward pass
    a1 = relu(np.matmul(w0, a0) + b1)
    a2 = relu(np.matmul(w1, a1) + b2)
    
    return a2

# def backpropagate():
    # ...    


# code

# user output index
user_outputs = ["1", "2"]

# neural network structure
input_size = 2
hidden_layers = 2  # update for this later
hidden_sizes = [3, 2]  # update for this later
hidden_1_size = 3  # phase this out
output_size = 2

# learning presets
learn = True  # add this functionality, add ability to choose original weights and biases
epochs = 100000
return_rate = 1000
learning_rate = 0.01

# training data
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

# instantiate weights and biases
w0 = np.random.randn(hidden_1_size, input_size)
b1 = np.zeros((hidden_1_size, 1))
w1 = np.random.randn(output_size, hidden_1_size)
b2 = np.zeros((output_size, 1))

# training loop
for i in range(epochs):
    # choose from training set
    training_choice = random.randint(0, len(input_training) - 1)
    
    # reformat inputs and outputs
    a0 = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    c = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))
    
    # calculate outputs
    a1 = relu(np.matmul(w0, a0) + b1)
    a2 = relu(np.matmul(w1, a1) + b2)

    # calculate gradients

    # second layer
    d_a2 = -2 * np.subtract(c, a2)
    d_b2 = relu_prime(np.matmul(w1, a1) + b2) * d_a2
    d_w1 = np.multiply(np.resize(d_b2, (hidden_1_size, output_size)).T, np.resize(a1.T, (output_size, hidden_1_size)))
    
    # first layer
    d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b2, (hidden_1_size, output_size)), w1.T), axis=1), (hidden_1_size, 1))
    d_b1 = relu_prime(np.matmul(w0, a0) + b1) * d_a1
    d_w0 = np.multiply(np.resize(d_b1, (input_size, hidden_1_size)).T, np.resize(a0.T, (hidden_1_size, input_size)))
    
    # optimize weights and biases

    w0 = np.subtract(w0, learning_rate * d_w0)
    b1 = np.subtract(b1, learning_rate * d_b1)
    w1 = np.subtract(w1, learning_rate * d_w1)
    b2 = np.subtract(b2, learning_rate * d_b2)

    # error report every few epochs

    if i % return_rate == 0:
        error = 0
        for j in range(len(a2)):
            error += (c[j] - a2[j]) ** 2
        print(f"Error: {error} ({round(i / epochs * 100)}%)")

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

# allow user to test optimized network
while True:
    print("")
    inputs = []
    for i in range(input_size):
        inputs.append(float(input("a(0)" + str(i) + ": ")))  # gets inputs

    # forward pass
    a0 = np.reshape(inputs, (len(inputs), 1))
    a2 = forward_pass(a0, w0, b1, w1, b2)
  
    # result
    print("")
    print(a2)
    output_index = np.nanargmax(np.where(a2 == a2.max(), a2, np.nan))
    print(f"Predicted: {user_outputs[output_index]}")

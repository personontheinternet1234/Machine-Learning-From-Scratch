import random

import numpy as np


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


# X & Y
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
Y = [[350, 514], [712, 1068], [1024, 1550]]

# W & B
W0 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]).T
W1 = np.array([[4, 6], [5, 8], [2, 3], [5, 7]]).T
B0 = np.array([[5], [6], [7], [8]])
B1 = np.array([[4], [9]])

# choose from training set
tc = random.randint(0, len(X) - 1)

# reformat inputs and outputs
A0 = np.reshape(np.array(X[tc]), (len(X[tc]), 1))
c = np.reshape(np.array(X[tc]), (len(X[tc]), 1))

# calculate outputs
a1 = relu(np.matmul(w0, a0) + b0)
a2 = relu(np.matmul(w1, a1) + b1)

# calculate gradients

# second layer
d_a2 = -2 * np.subtract(c, a2)
d_b1 = relu_prime(np.matmul(w1, a1) + b1) * d_a2
d_w1 = np.multiply(np.resize(d_b1, (hidden_1_size, output_size)).T, np.resize(a1, (output_size, hidden_1_size)))

# first layer
d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b1, (hidden_1_size, output_size)), w1.T), axis=1), (hidden_1_size, 1))
d_b0 = relu_prime(np.matmul(w0, a0) + b0) * d_a1
d_w0 = np.multiply(np.resize(d_b0, (input_size, hidden_1_size)).T, np.resize(a0, (hidden_1_size, input_size)))

# optimize weights and biases

w0 = np.subtract(w0, learning_rate * d_w0)
b0 = np.subtract(b0, learning_rate * d_b0)
w1 = np.subtract(w1, learning_rate * d_w1)
b1 = np.subtract(b1, learning_rate * d_b1)
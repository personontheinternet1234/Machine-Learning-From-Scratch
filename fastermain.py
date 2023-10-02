import numpy as np
import random
from tqdm import tqdm

# class NonLinearity:
    
#     def __init__(self, values):
#         self.values = values
        
#     class sigmoid(self, values):
#         self.values = values

#         define forward(self):
#             output = 1 / (1 + np.exp(-1 * values))
#             return output

#         define prime(self):
#             output = 1 / (1 + np.exp(-1 * values)) * (1 - 1 / (1 + np.exp(-1 * values)))
#             return output
        
        


# class Error:
#     ...


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
    if values > 0:
        return 1
    else:
        return 0.1

# training data
input_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
output_training = [
    [0, 0],
    [0, 1],
    [1, 0],
    [0.5, 0.5]
]

# user output index
user_outputs = ["0", "1"]

# add hidden layers when necessary
input_size = 2
hidden_layers = 2  # update for this later
hidden_sizes = [3, 2]  # update for this later
hidden_1_size = 3
output_size = 2
epochs = 10000
learning_rate = 0.01

# instantiate weights and biases
w0 = np.random.randn(hidden_1_size, input_size)
b1 = np.zeros((hidden_1_size, 1))
w1 = np.random.randn(output_size, hidden_1_size)  # to x from
b2 = np.zeros((output_size, 1))

def forward(training_choice):
    # forward pass # this works i think
    global a1
    global a2
    global a0
    global c
    # reformat inputs and outputs
    a0 = np.reshape(np.array(input_training[training_choice]), (len(input_training[training_choice]), 1))
    c = np.reshape(np.array(output_training[training_choice]), (len(output_training[training_choice]), 1))
    
    # calculate outputs
    a1 = sigmoid(np.matmul(w0, a0) + b1)
    a2 = sigmoid(np.matmul(w1, a1) + b2)

    return a2

def backward():  # calculate gradients
    global w0
    global b1
    global w1
    global b2

    # second layer
    d_a2 = -2 * np.subtract(c, a2)
    d_b2 = sigmoid_prime(np.matmul(w1, a1) + b2) * d_a2
    d_w1 = np.multiply(np.resize(d_b2, (hidden_1_size, output_size)).T, np.resize(a1.T, (output_size, hidden_1_size)))

    # first layer
    d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b2, (hidden_1_size, output_size)), w1.T), axis=1), (hidden_1_size, 1))
    d_b1 = sigmoid_prime(np.matmul(w0, a0) + b1) * d_a1
    d_w0 = np.multiply(np.resize(d_b1, (input_size, hidden_1_size)).T, np.resize(a0.T, (hidden_1_size, input_size)))

    # optimize weights and biases

    w0 = np.subtract(w0, learning_rate * d_w0)
    # print(d_w0)
    # print(w0)
    b1 = np.subtract(b1, learning_rate * d_b1)
    # print(d_b1)
    # print(b1)
    w1 = np.subtract(w1, learning_rate * d_w1)
    # print(d_w1)
    # print(w1)
    b2 = np.subtract(b2, learning_rate * d_b2)
    # print(d_b2)
    # print(b2)

# training loop
for i in range(epochs):
    # choose from training set
    training_choice = random.randint(0, len(input_training) - 1)
    
    forward(training_choice)
    backward()

    # MSE
    if i % 100 == 0:
        error = 0
        for point in range(len(input_training) - 1):
            calculatedoutputs = forward(point)
            error += (output_training[point] - calculatedoutputs) ** 2
        error /= len(input_training)
        error = np.sum(error)
        print(f"MSE: {error}")


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
    a1 = sigmoid(np.matmul(w0, a0) + b1)
    a2 = sigmoid(np.matmul(w1, a1) + b2)
  
    # result
    print("")
    print(a2)
    output_index = np.nanargmax(np.where(a2 == a2.max(), a2, np.nan))
    print(user_outputs[output_index])

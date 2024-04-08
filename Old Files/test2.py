import numpy as np


def relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_relu(values):
    return np.where(values > 0, 1, 0.1)


def tb():
    print("---------------")


lr = 0.1
tc = 0
ins = 3
hls = 4
ots = 2

X = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
Y = [np.array([350, 514]), np.array([712, 1068]), np.array([1024, 1550])]

W0 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
W1 = np.array([[4, 6], [5, 8], [2, 3], [5, 7]])
B0 = np.array([5, 6, 7, 8])
B1 = np.array([4, 9])

# instantiate weights and biases

# f
A1 = relu(np.matmul(X, W0) + B0)
A2 = relu(np.matmul(A1, W1) + B1)

A1_s = relu(np.matmul(X[tc], W0) + B0)
A2_s = relu(np.matmul(A1_s, W1) + B1)

# b
# dA2 = -2 * np.subtract(Y, A2)
dA2 = -2 * (Y - A2)
dB1 = np.sum(d_relu(np.matmul(A1, W1) + B1) * dA2, axis=0)
dW1 = np.resize(dB1, (hls, ots)).T * np.resize(A1, (ots, hls))
# dW1 = np.multiply(np.resize(dB1, (hls, ots)).T, np.resize(A1, (ots, hls)))
print(dB1)
print(np.resize(dB1, (len(dB1), 1)))
print()
print(np.matmul(np.sum(A1, axis=0), np.resize(dB1, (len(dB1), 1))))
print(dW1)
print(W1)
tb()

# dA2_s = -2 * np.subtract(Y[tc], A2_s)
dA2_s = -2 * (Y[tc] - A2_s)
dB1_s = d_relu(np.matmul(A1_s, W1) + B1) * dA2_s
tb()
print(dB1_s)
# dW1_s = np.multiply(np.resize(dB1_s, (hls, ots)).T, np.resize(A1_s, (ots, hls)))
dW1_s = np.resize(dB1_s, (hls, ots)).T * np.resize(A1_s, (ots, hls))

dA1 = np.reshape(np.sum(np.resize(dB1, (hls, ots)) * W1, axis=1), (hls, 1))
dB0 = d_relu(np.matmul(W0, X) + B0) * dA1
dW0 = np.resize(dB0, (ins, hls)).T * np.resize(A0, (hls, ins))

dA1_s = np.reshape(np.sum(np.resize(dB1_s, (hls, ots)) * W1, axis=1), (hls, 1))
dB0_s = d_relu(np.matmul(W0, X[tc]) + B0) * dA1_s


print(dW1)
tb()
print(dW1_s)

# # second layer
# d_a2 = -2 * np.subtract(c, a2)
# d_b1 = d_relu(np.matmul(w1, a1) + b1) * d_a2
# d_w1 = np.multiply(np.resize(d_b1, (hidden_1_size, output_size)).T, np.resize(a1, (output_size, hidden_1_size)))
#
# # first layer
# d_a1 = np.reshape(np.sum(np.multiply(np.resize(d_b1, (hidden_1_size, output_size)), w1.T), axis=1), (hidden_1_size, 1))
# d_b0 = d_relu(np.matmul(w0, a0) + b0) * d_a1
# d_w0 = np.multiply(np.resize(d_b0, (input_size, hidden_1_size)).T, np.resize(a0, (hidden_1_size, input_size)))
#
# # optimize weights and biases
#
# w0 = np.subtract(w0, learning_rate * d_w0)
# b0 = np.subtract(b0, learning_rate * d_b0)
# w1 = np.subtract(w1, learning_rate * d_w1)
# b1 = np.subtract(b1, learning_rate * d_b1)

import numpy as np


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def sigmoid(values):
    output = 1 / (1 + np.exp(-1 * values))
    return output


def d_sigmoid(values):
    output = sigmoid(values) * (1 - sigmoid(values))
    return output


def tb():
    print("---------------")


# X & Y
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
Y = [[350, 514], [712, 1068], [1024, 1550]]

# W & B
W0 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]).T
W1 = np.array([[4, 6], [5, 8], [2, 3], [5, 7]]).T
B0 = np.array([[5], [6], [7], [8]])
B1 = np.array([[4], [9]])

# params
lr = 0.1
tc = 0
ins = 3
hls = 4
ots = 2

# reformat
A0 = np.reshape(np.array(X[tc]), (len(X[tc]), 1))
c = np.reshape(np.array(Y[tc]), (len(Y[tc]), 1))

# f
A1 = sigmoid(np.matmul(W0, A0) + B0)
A2 = sigmoid(np.matmul(W1, A1) + B1)
# tb()
# print(A1.T)
# tb()
# print(A2.T)
# tb()

# for i in range(3):
#     print("")

# b
# l2
# tb()
dA2 = -2 * np.subtract(c, A2)
# print(dA2.T)
tb()
dB1 = d_sigmoid(np.matmul(W1, A1) + B1) * dA2
print(dB1.T)
tb()
dW1 = np.multiply(np.resize(dB1, (hls, ots)).T, np.resize(A1, (ots, hls)))
# print(dW1.T)
# tb()
# l1
dA1 = np.reshape(np.sum(np.multiply(np.resize(dB1, (hls, ots)), W1.T), axis=1), (hls, 1))
# print(np.sum(np.multiply(np.resize(dB1, (hls, ots)), W1.T), axis=1))
# print(dA1.T)
# tb()
dB0 = d_sigmoid(np.matmul(W0, A0) + B0) * dA1
print(dB0.T)
tb()
dW0 = np.multiply(np.resize(dB0, (ins, hls)).T, np.resize(A0, (hls, ins)))
# print(dW0.T)
# tb()

# for i in range(3):
#     print("")

# optimize
W0 = np.subtract(W0, lr * dW0)
B0 = np.subtract(B0, lr * dB0)
W1 = np.subtract(W1, lr * dW1)
B1 = np.subtract(B1, lr * dB1)

# return
# tb()
# print(W0.T)
# tb()
# print(B0.T)
# tb()
# print(W1.T)
# tb()
# print(B1.T)
# tb()

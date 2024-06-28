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


lr = 0.1
tc = 0
ins = 3
hls = 4
ots = 2

X = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), np.array([[7, 8, 9]])]
Y = [np.array([[350, 514]]), np.array([[712, 1068]]), np.array([[1024, 1550]])]

W0 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
W1 = np.array([[4, 6], [5, 8], [2, 3], [5, 7]])
B0 = np.array([5, 6, 7, 8])
B1 = np.array([4, 9])

# f
A0 = X[tc]
A1 = sigmoid(A0 @ W0 + B0)
A2 = sigmoid(A1 @ W1 + B1)
E = Y[tc]

# b
# l2
dA2 = -2 * (E - A2)
print(d_sigmoid(A1 * W1 + B1) * dA2)
print(d_sigmoid(A1 @ W1 + B1) @ dA2.T)
dB1 = d_sigmoid(A1 @ W1 + B1) @ dA2
dW1 = A1.T @ dB1
# l1
dA1 = np.array([np.sum(W1 @ dB1, axis=1)])
dB0 = d_sigmoid(A0 @ W0 + B0) @ dA1
dW0 = A0.T @ dB0

# optimize
W0 = W0 - lr * dW0
B0 = B0 - lr * dB0
W1 = W1 - lr * dW1
B1 = B1 - lr * dB1

# return
# tb()
# print(A0)
# tb()
# print(A1)
# tb()
# print(A2)
# tb()
# print(E)
# tb()
# for i in range(3):
#     print("")
# tb()
# print(dA2)
tb()
print(dB1)
tb()
# print(dW1)
# tb()
# print(dA1)
# tb()
print(dB0)
tb()
# print(dW0)
# tb()
# for i in range(3):
#     print("")
# tb()
# print(W0)
# tb()
# print(B0)
# tb()
# print(W1)
# tb()
# print(B1)
# tb()

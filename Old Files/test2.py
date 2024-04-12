import numpy as np


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
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

# f
A1_s = l_relu(np.matmul(W0.T, np.reshape(X[tc], (3, 1))) + np.reshape(B0, (4, 1))).T
A2_s = l_relu(np.matmul(A1_s, W1) + B1)

tb()
print(A1_s)
tb()
print(A2_s)
tb()

for i in range(3):
    print("")

# b
# l2
tb()
dA2 = -2 * np.subtract(Y[tc], A2_s)
print(dA2)
tb()
dB1 = dA2
print(dB1)
tb()
dW1 = A1_s.T * dB1
print(dW1)
tb()
# l1
dA1 = np.reshape(np.sum(W1 * dB1, axis=1), (1, hls))
print(dA1)
tb()
dB0 = dA1
print(dB0)
tb()
dW0 = np.multiply(np.resize(X[tc], (len(X[tc]), 1)), dB0)
print(dW0)
tb()

for i in range(3):
    print("")

# optimize
W0 = np.subtract(W0, lr * dW0)
B0 = np.subtract(B0, lr * dB0)
W1 = np.subtract(W1, lr * dW1)
B1 = np.subtract(B1, lr * dB1)

# return
tb()
print(W0)
tb()
print(B0)
tb()
print(W1)
tb()
print(B1)
tb()

import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


def d_l_relu(values):
    return np.where(values > 0, 1, 0.1)


def tb():
    print("---------------")


lr = 0.01
ep = 10000
ins = 2
hls = 3
ots = 2

X = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
Y = [np.array([[1, 1]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[0, 0]])]

W0 = np.random.randn(ins, hls)
B0 = np.zeros((1, hls))
W1 = np.random.randn(hls, ots)
B1 = np.zeros((1, ots))

t1 = time.time()
Ll = []
for i in tqdm(range(ep), ncols=150):
    # f
    tc = int(np.random.rand() * len(X))
    A0_s = X[tc]
    A1_s = l_relu(np.matmul(A0_s, W0) + B0)
    A2_s = l_relu(np.matmul(A1_s, W1) + B1)
    E = Y[tc]

    # b
    # l2
    dA2 = -2 * (E - A2_s)
    dB1 = dA2
    dW1 = A1_s.T * dB1
    # l1
    dA1 = np.array([np.sum(W1 * dB1, axis=1)])
    dB0 = dA1
    dW0 = A0_s.T * dB0

    # o
    W0 = W0 - lr * dW0
    B0 = B0 - lr * dB0
    W1 = W1 - lr * dW1
    B1 = B1 - lr * dB1

    # e
    X1 = l_relu(np.matmul(X, W0) + B0)
    X2 = l_relu(np.matmul(X1, W1) + B1)
    L = np.sum(np.subtract(Y, X2) ** 2) / len(X)

    # s
    Ll.append(L)
t2 = time.time()

print(t2 - t1)
plt.plot(range(ep), Ll, color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.show()

while True:
    print("")
    inputs = []
    for i in range(ins):
        inputs.append(float(input(f"{str(i)}: ")))

    # f
    A0_s = np.array(inputs)
    A1_s = l_relu(np.matmul(A0_s, W0) + B0)
    A2_s = l_relu(np.matmul(A1_s, W1) + B1)

    # r
    print("")
    print(A2_s)

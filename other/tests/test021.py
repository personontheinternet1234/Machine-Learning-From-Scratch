import random

import numpy as np


def g1(v):
    output = np.maximum(0.1 * v, v)
    return output


def dg1(v):
    return np.where(v > 0, 1, 0.1)


def g2(v):
    return np.exp(v) / np.sum(np.exp(v))


def dg2(v):
    return v


def j(yh, yl):
    return np.sum((yl - yh) ** 2)


def dj(yh, yl):
    return -2 * (yl - yh)


def tb():
    print('----------')


# s
l1 = 3
l2 = 4
l3 = 3
it = 10000
lr = 0.1
bs = 8

# t
x = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
y = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
# fm
for i in range(len(x)):
    x[i] = [x[i]]
    y[i] = [y[i]]
x = np.array(x)
y = np.array(y)

# p
w1 = np.random.randn(l1, l2)
b1 = np.zeros((1, l2))
w2 = np.random.randn(l2, l3)
b2 = np.zeros((1, l3))

for i in range(it):
    # ch
    tc = random.randint(bs, 8)
    a1 = x[tc-bs:tc]
    y1 = y[tc-bs:tc]

    # f
    a2 = g1(a1 @ w1) + b1
    a3 = g1(a2 @ w2) + b2
    c = j(a3, y1)

    # b
    # l3
    da3 = dj(a3, y1)
    # l2
    db2 = dg1(a2 @ w2 + b2) * da3
    dw2 = np.reshape(a2, (bs, l2, 1)) * db2
    da2 = np.reshape(np.sum(w2 * db2, axis=2, keepdims=True), (bs, 1, l2))
    # l1
    db1 = dg1(a1 @ w1 + b1) * da2
    dw1 = np.reshape(a1, (bs, l1, 1)) * db1

    # o
    b2 -= lr * np.sum(db2, axis=0) / bs
    w2 -= lr * np.sum(dw2, axis=0) / bs
    b1 -= lr * np.sum(db1, axis=0) / bs
    w1 -= lr * np.sum(dw1, axis=0) / bs

    # # r
    print(c)
    tb()

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
    return


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
it = 1000
lr = 0.1

# t
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]])

# f
w1 = np.random.randn(l1, l2)
b1 = np.zeros(l2)
w2 = np.random.randn(l2, l3)
b2 = np.zeros(l3)

tc = random.randint(0, 7)

for i in range(it):
    # f
    a1 = x[tc]
    y1 = y[tc]
    a2 = g1(a1 @ w1 + b1)
    a3 = g1(a2 @ w2 + b2)
    c = j(a3, y1)
    print(c)

    # b
    # l3
    da3 = dj(a3, y1)
    dal = dg1(a2 @ w2 + b2)
    db2 = dal * da3
    dw2 =

    # o
    b2 -= lr * db2

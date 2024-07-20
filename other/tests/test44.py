import random
import time

import numpy as np

from gardenpy import (
    Tensor,
    nabla,
    chain,
    Activators,
    Losses
)


def g(v):
    output = np.maximum(0.1 * v, v)
    return output


def dg(v):
    return np.where(v > 0, 1, 0.1)


def j(yh, yl):
    return np.sum((yl - yh) ** 2)


def dj(yh, yl):
    return -2 * (yl - yh)


def tb():
    print('----------')


# object instantiation
gtens = Activators('lrelu', beta=0.1)
gt = gtens.activate
dgt = gtens.d_activate
jtens = Losses('ssr')
jt = jtens.loss
djt = jtens.d_loss


# s
l1 = 3
l2 = 4
l3 = 3
it = 1
lr = 0.1

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

w1t = Tensor(w1)
b1t = Tensor(b1)
w2t = Tensor(w2)
b2t = Tensor(b2)

s = time.time()
for i in range(it):
    # choice
    tc = random.randint(0, 7)

    # data instantiation
    a1 = x[tc]
    y1 = y[tc]

    # data instantiation tensors
    a1t = Tensor(a1)
    y1t = Tensor(y1)

    # layer 1
    a2 = g(a1 @ w1 + b1)
    a2t = gt(a1t @ w1t + b1t)
    print(a2)
    print()
    print(a2t)
    tb()

    # layer 2
    a3 = g(a2 @ w2 + b2)
    a3t = gt(a2t @ w2t + b2t)
    print(a3)
    print()
    print(a3t)
    tb()

    # layer 3
    c = j(a3, y1)
    ct = jt(y1t, a3t)
    print(c)
    print()
    print(ct)
    tb()

    # back layer 3
    da3 = dj(a3, y1)
    da3t = nabla(ct, a3t)
    print(da3)
    print()
    print(da3t)
    tb()

    # back layer 2
    db2 = dg(a2 @ w2 + b2) * da3
    db2t = chain(da3t, nabla(a3t, b2t))

    dw2 = a2.T * db2

    da2 = np.sum(w2 * db2, axis=1)


    # back layer 1
    db1 = dg(a1 @ w1 + b1) * da2

    dw1 = a1.T * db1


    # o
    b2 -= lr * db2
    w2 -= lr * dw2
    b1 -= lr * db1
    w1 -= lr * dw1

    # r
    # print(c)
    # tb()

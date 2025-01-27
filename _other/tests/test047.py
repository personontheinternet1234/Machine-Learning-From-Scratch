import numpy as np

from gardenpy import (
    nabla,
    Initializers,
    Losses,
    Activators
)


def tb():
    print('----------')


def g(v):
    output = np.maximum(0.01 * v, v)
    return output


def dg(v):
    return np.where(v > 0, 1, 0.01)


def j(yh, yl):
    return np.sum((yl - yh) ** 2)


def dj(yh, yl):
    return -2 * (yl - yh)


init = Initializers(algorithm='xavier')
act = Activators(algorithm='lrelu')
loss = Losses(algorithm='ssr')

# objects

x = init.initialize(1, 5)
y = init.initialize(1, 5)

xarr = x.to_array()
yarr = y.to_array()

# forward

yhat = act.activate(x)
los = loss.loss(yhat, y)

yhatarr = g(xarr)
losarr = j(yhatarr, yarr)

# backward

deriv = nabla(los, yhat)

derivarr = dj(yhatarr, yarr)

deriv2 = nabla(yhat, x)

deriv2arr = dg(xarr)

# result testing

tb()

print(deriv)
print(derivarr)

tb()

print(deriv2)
print(deriv2arr)

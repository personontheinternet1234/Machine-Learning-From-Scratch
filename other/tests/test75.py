import numpy as np
from functional.algorithms import Initializers, Losses, Optimizers
from functional.operators import tensor, nabla
from functional.objects import Tensor

init = Initializers('gaussian').initialize
criterion = Losses('ssr').loss
optim = Optimizers('adam').optimize

x = tensor([[1, 0.5]])
w = tensor([[0.5, 0.5]])
y = tensor([[0.5, 1]])

for i in range(1000):
    yhat = x * w
    loss = criterion(yhat, y)
    grad_w = nabla(w, loss)
    w -= grad_w * np.array([[0.01]])
    print(yhat)
    Tensor.zero_grad(x, w, y)

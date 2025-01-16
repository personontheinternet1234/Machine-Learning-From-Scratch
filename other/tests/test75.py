import numpy as np
from functional.algorithms import Initializers, Losses, Optimizers
from functional.operators import tensor, nabla
from functional.objects import Tensor

init = Initializers('gaussian').initialize
criterion = Losses('ssr').loss
optim = Optimizers('sgd', alpha=0.01).optimize

x = tensor([[1, 0.5]])
w = init(1, 2)
y = tensor([[0.5, 1]])

for i in range(1000):
    yhat = x * w
    loss = criterion(yhat, y)
    grad_w = nabla(w, loss)
    itm_id = w.id
    w = optim(w, grad_w)
    print(f"Loss: {str(loss)[2:-2]}")
    Tensor.zero_grad(x, w, y)

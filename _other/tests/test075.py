from gardenpy.functional.algorithms import Initializers, Losses, Optimizers
from gardenpy.functional.operators import tensor, nabla
from gardenpy.functional.objects import Tensor

init = Initializers('gaussian')
criterion = Losses('ssr')
optim = Optimizers('rmsp', alpha=1e-2)

x = tensor([[1, 0.5]])
w = init(1, 2)
y = tensor([[0.5, 1]])
print_loss = True

for i in range(1_000):
    yhat = x * w
    loss = criterion(yhat, y)
    grad_w = nabla(w, loss)
    w = optim(w, grad_w)
    if print_loss:
        print(f"Loss: {str(loss)[2:-2]}")
    Tensor.zero_grad(x, w, y)

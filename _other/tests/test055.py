import time

from gardenpy import (
    Tensor,
    nabla,
    chain,
    Initializers,
    Activators,
    Losses,
    Optimizers
)
from gardenpy.utils import (
    ansi,
    progress,
    convert_time
)
from gardenpy.utils.helpers import print_contributors

##########

max_iter = 100000

##########

w_init = Initializers(algorithm='xavier')
b_init = Initializers(algorithm='uniform', value=0.0)

act1 = Activators(algorithm='lrelu')
act2 = Activators(algorithm='lrelu')
loss = Losses(algorithm='ssr')
optim = Optimizers(algorithm='rmsp', gamma=1e-2)

##########
x = Tensor([[1.0, 1.0, 1.0]])
y = Tensor([[0.1, 0.1, 0.1]])

w1 = w_init.initialize(3, 4)
w2 = w_init.initialize(4, 3)

b1 = b_init.initialize(1, 4)
b2 = b_init.initialize(1, 3)

g1 = act1.activate
g2 = act2.activate
j = loss.loss
step = optim.optimize

##########

yhat = None
ep = 0

print()
print_contributors()
print(f"\n{ansi['bold']}Training{ansi['reset']}")

start = time.time()
print(f"Epoch {ansi['white']}{ep + 1}{ansi['reset']}")

##########

for i in range(max_iter):
    alpha1 = x @ w1 + b1
    a2 = g1(alpha1)

    alpha2 = a2 @ w2 + b2
    yhat = g2(alpha2)

    loss = j(yhat, y)

    ##########

    grad_alpha2 = nabla(loss, alpha2)
    grad_b2 = chain(grad_alpha2, nabla(alpha2, b2))
    # grad_w2 = chain(grad_alpha2, nabla(alpha2, w2))

    # grad_alpha1 = chain(grad_alpha2, nabla(alpha2, alpha1))
    # grad_b1 = chain(grad_alpha1, nabla(alpha1, b1))
    # grad_w1 = chain(grad_alpha1, nabla(alpha1, w1))

    ##########

    print(b2)
    print(grad_b2)
    b2 = step(b2, grad_b2)
    # w2 = step(w2, grad_w2)
    # b1 = step(b1, grad_b1)
    # w1 = step(w1, grad_w1)

    ##########

    elapsed = time.time() - start
    desc = (
        f"{str(i + 1).zfill(len(str(max_iter)))}{ansi['white']}it{ansi['reset']}/{max_iter}{ansi['white']}it{ansi['reset']}  "
        f"{(100 * (i + 1) / max_iter):05.1f}{ansi['white']}%{ansi['reset']}  "
        f"{round((i + 1) / elapsed, 1)}{ansi['white']}it/s{ansi['reset']}  "
        f"{loss[0]:.3}{ansi['white']}loss{ansi['reset']}  "
        f"{convert_time(elapsed)}{ansi['white']}et{ansi['reset']}  "
        f"{convert_time(elapsed * max_iter / (i + 1) - elapsed)}{ansi['white']}eta{ansi['reset']}"
    )
    progress(i, max_iter, desc=desc)

##########

print(f"\n\n{ansi['bold']}Results{ansi['reset']}")
print(f"Predicted    {ansi['white']}{ansi['italic']}{yhat}{ansi['reset']}")
print(f"Expected     {ansi['white']}{ansi['italic']}{y}{ansi['reset']}")
print()

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
from gardenpy.utils.helper_functions import print_contributors

##########

max_iter = 10000

##########

w_init = Initializers(algorithm='xavier')
b_init = Initializers(algorithm='uniform', value=0.0)

act1 = Activators(algorithm='lrelu')
act2 = Activators(algorithm='lrelu')
loss = Losses(algorithm='ssr')
gamma = 1e-2
alg = 'rmsp'
optim_b2 = Optimizers(algorithm=alg, gamma=gamma)
optim_w2 = Optimizers(algorithm=alg, gamma=gamma)
optim_b1 = Optimizers(algorithm=alg, gamma=gamma)
optim_w1 = Optimizers(algorithm=alg, gamma=gamma)

##########
x = Tensor([[1.0, 0.5, 1.0]])
y = Tensor([[5.0, 5.0, 3.0]])

w1 = w_init.initialize(3, 4)
w2 = w_init.initialize(4, 3)

b1 = b_init.initialize(1, 4)
b2 = b_init.initialize(1, 3)

g1 = act1.activate
g2 = act2.activate
j = loss.loss
step_b2 = optim_b2.optimize
step_w2 = optim_w2.optimize
step_b1 = optim_b1.optimize
step_w1 = optim_w1.optimize

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
    alpha1 = x @ w1
    beta1 = alpha1 + b1
    a2 = g1(beta1)

    alpha2 = a2 @ w2
    beta2 = alpha2 + b2
    yhat = g2(beta2)

    loss = j(yhat, y)

    ##########

    grad_yhat = nabla(loss, yhat)
    grad_beta2 = chain(grad_yhat, nabla(yhat, beta2))
    grad_b2 = chain(grad_beta2, nabla(beta2, b2))
    grad_alpha2 = chain(grad_beta2, nabla(beta2, alpha2))
    grad_w2 = chain(grad_alpha2, nabla(alpha2, w2))

    grad_a2 = chain(grad_alpha2, nabla(alpha2, a2))
    grad_beta1 = chain(grad_a2, nabla(a2, beta1))
    grad_b1 = chain(grad_beta1, nabla(beta1, b1))
    grad_alpha1 = chain(grad_beta1, nabla(beta1, alpha1))
    grad_w1 = chain(grad_alpha1, nabla(alpha1, w1))

    ##########

    b2 = step_b2(b2, grad_b2)
    w2 = step_w2(w2, grad_w2)
    # b1 = step_b1(b1, grad_b1)
    # w1 = step_w1(w1, grad_w1)

    ##########

    elapsed = time.time() - start
    desc = (
        f"{str(i + 1).zfill(len(str(max_iter)))}{ansi['white']}it{ansi['reset']}/{max_iter}{ansi['white']}it{ansi['reset']}  "
        f"{(100 * (i + 1) / max_iter):05.1f}{ansi['white']}%{ansi['reset']}  "
        f"{round((i + 1) / elapsed, 1)}{ansi['white']}it/s{ansi['reset']}  "
        f"{convert_time(elapsed)}{ansi['white']}et{ansi['reset']}  "
        f"{convert_time(elapsed * max_iter / (i + 1) - elapsed)}{ansi['white']}eta{ansi['reset']}  "
        f"{loss[0]:.3}{ansi['white']}loss{ansi['reset']}"
    )
    progress(i, max_iter, desc=desc)

##########

print(f"\n\n{ansi['bold']}Results{ansi['reset']}")
print(f"Predicted    {ansi['white']}{ansi['italic']}{yhat}{ansi['reset']}")
print(f"Expected     {ansi['white']}{ansi['italic']}{y}{ansi['reset']}")
print()

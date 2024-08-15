import random
import time

import numpy as np

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

max_iter = 20000

##########

w_init = Initializers(algorithm='xavier')
b_init = Initializers(algorithm='uniform', value=0.0)

act1 = Activators(algorithm='lrelu')
act2 = Activators(algorithm='softmax')
loss = Losses(algorithm='focal')
alg = 'adam'
hyp = {}
optim_b2 = Optimizers(alg, hyperparameters=hyp, correlator=False)
optim_w2 = Optimizers(alg, hyperparameters=hyp, correlator=False)
optim_b1 = Optimizers(alg, hyperparameters=hyp, correlator=False)
optim_w1 = Optimizers(alg, hyperparameters=hyp, correlator=False)

##########
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0, 1], [1, 0], [1, 0], [0, 1]]
for i in range(len(x)):
    x[i] = Tensor([x[i]])
for i in range(len(y)):
    y[i] = Tensor([y[i]])

w1 = w_init.initialize(2, 3)
w2 = w_init.initialize(3, 2)

b1 = b_init.initialize(1, 3)
b2 = b_init.initialize(1, 2)

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
    tc = random.randint(0, 3)
    tc = 2
    a1 = x[tc]
    y1 = y[tc]
    alpha1 = a1 @ w1
    beta1 = alpha1 + b1
    a2 = g1(beta1)

    alpha2 = a2 @ w2
    beta2 = alpha2 + b2
    yhat = g2(beta2)

    loss = j(yhat, y1)

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
    print(f"\n{b2.get_internals()}")
    print(f"\n{w2.get_internals()}")
    # b1 = step_b1(b1, grad_b1)
    # w1 = step_w1(w1, grad_w1)

    ##########

    a1.tracker_reset()
    y1.tracker_reset()

    grad_b2.tracker_reset()
    grad_w2.tracker_reset()
    grad_b1.tracker_reset()
    grad_w1.tracker_reset()

    b2.tracker_reset()
    w2.tracker_reset()
    b1.tracker_reset()
    w1.tracker_reset()

    ##########

    floss = Tensor([0])
    faccu = 0
    for c in range(len(x)):
        alpha1 = Tensor(x[c].to_array()) @ w1
        beta1 = alpha1 + b1
        a2 = g1(beta1)

        alpha2 = a2 @ w2
        beta2 = alpha2 + b2
        yhat = g2(beta2)

        floss += j(yhat, y[c])
        faccu += np.nanargmax(yhat.to_array())  # this doesn't work

    elapsed = time.time() - start
    faccu /= 0.04
    desc = (
        f"{str(i + 1).zfill(len(str(max_iter)))}{ansi['white']}it{ansi['reset']}/{max_iter}{ansi['white']}it{ansi['reset']}  "
        f"{(100 * (i + 1) / max_iter):05.1f}{ansi['white']}%{ansi['reset']}  "
        f"{round((i + 1) / elapsed, 1)}{ansi['white']}it/s{ansi['reset']}  "
        f"{convert_time(elapsed)}{ansi['white']}et{ansi['reset']}  "
        f"{convert_time(elapsed * max_iter / (i + 1) - elapsed)}{ansi['white']}eta{ansi['reset']}  "
        f"{faccu:05.1f}{ansi['white']}%accu{ansi['reset']}  "
        f"{floss[0]:.5}{ansi['white']}loss{ansi['reset']}"
    )
    progress(i, max_iter, desc=desc)

##########

yhats = []
for case in x:
    yhats.append(g2(g1(case @ w1 + b1) @ w2 + b2))

print(f"\n\n{ansi['bold']}Results{ansi['reset']}")
print(f"Predicted    {ansi['white']}{ansi['italic']}{yhats}{ansi['reset']}")
print(f"Expected     {ansi['white']}{ansi['italic']}{y}{ansi['reset']}")
print()

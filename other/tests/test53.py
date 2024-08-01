import time

import numpy as np

from gardenpy import (
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

max_iter = 100000

##########

W_init = Initializers(algorithm='xavier')
B_init = Initializers(algorithm='uniform', value=0.0)
P_init = Initializers(algorithm='uniform', value=1.0)

act1 = Activators(algorithm='relu')
act2 = Activators(algorithm='softmax')
loss = Losses(algorithm='centropy')
optim = Optimizers(algorithm='adam', gamma=1e-2)

##########

X = P_init.initialize(1, 3)
Y = P_init.initialize(1, 3)
Y *= 0.0
Y[0][0] = 1.0

W1 = W_init.initialize(3, 4)
W2 = W_init.initialize(4, 3)

B1 = B_init.initialize(1, 4)
B2 = B_init.initialize(1, 3)

g1 = act1.activate
g2 = act2.activate
j = loss.loss
step = optim.optimize

##########

Yhat = None
ep = 0

print()
print_contributors()
print(f"\n{ansi['bold']}Training{ansi['reset']}")

start = time.time()
print(f"Epoch {ansi['white']}{ep + 1}{ansi['reset']}")

##########

for i in range(max_iter):
    alpha1 = X @ W1 + B1
    a2 = g1(alpha1)

    alpha2 = a2 @ W2 + B2
    Yhat = g2(alpha2)

    L = j(Yhat, Y)

    ##########

    grad_alpha2 = nabla(L, alpha2)
    grad_B2 = chain(grad_alpha2, nabla(alpha2, B2))
    # grad_W2 = chain(grad_alpha2, nabla(alpha2, W2))

    # grad_alpha1 = chain(grad_alpha2, nabla(alpha2, alpha1))
    # grad_B1 = chain(grad_alpha1, nabla(alpha1, B1))
    # grad_W1 = chain(grad_alpha1, nabla(alpha1, W1))

    ##########

    # W2 = step(W2, grad_W2)
    B2 = step(B2, grad_B2)
    # W1 = step(W1, grad_W1)
    # B1 = step(B1, grad_B1)

    ##########

    accu = 100.0 - 50.0 * np.sum(np.abs(np.argmax(Yhat) - np.argmax(Y)))
    elapsed = time.time() - start
    desc = (
        f"{str(i + 1).zfill(len(str(max_iter)))}{ansi['white']}it{ansi['reset']}/{max_iter}{ansi['white']}it{ansi['reset']}  "
        f"{(100 * (i + 1) / max_iter):05.1f}{ansi['white']}%{ansi['reset']}  "
        f"{L[0]:.3}{ansi['white']}loss{ansi['reset']}  "
        f"{accu:05.1f}{ansi['white']}accu{ansi['reset']}  "
        f"{convert_time(elapsed)}{ansi['white']}et{ansi['reset']}  "
        f"{convert_time(elapsed * max_iter / (i + 1) - elapsed)}{ansi['white']}eta{ansi['reset']}  "
        f"{round((i + 1) / elapsed, 1)}{ansi['white']}it/s{ansi['reset']}"
    )
    progress(i, max_iter, desc=desc)

##########

print(f"\n\n{ansi['bold']}Results{ansi['reset']}")
print(f"Predicted    {ansi['white']}{ansi['italic']}{Yhat}{ansi['reset']}")
print(f"Expected     {ansi['white']}{ansi['italic']}{Y}{ansi['reset']}")
print()

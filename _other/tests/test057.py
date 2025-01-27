import numpy as np

from gardenpy import (
    nabla,
    chain,
    Initializers,
    Losses,
    Activators
)

init = Initializers('gaussian')
act = Activators('lrelu').activate
loss = Losses('ssr').loss
# 5, 4
a1 = init.initialize(1, 5)
w1 = init.initialize(5, 4)
b1 = init.initialize(1, 4)
y = init.initialize(1, 4)

r1 = a1 @ w1 + b1
r2 = act(r1)
r3 = loss(r2, y)

grad_r2 = nabla(r3, r2)
grad_b1 = chain(grad_r2, nabla(r2, b1))
grad_r1 = chain(grad_r2, nabla(r2, r1))
grad_w1 = chain(grad_r1, nabla(r1, w1))
grad_a1 = nabla(r1, a1)

print('\n\nr3')
print(r3)
print('\n\nr2')
print(r2)
print(grad_r2)
print(grad_r2.tracker['org'])
print('\n\nr1')
print(r1)
print(grad_r1)
print(grad_r1.tracker['org'])
print('\n\nb1')
print(b1)
print(grad_b1)
print(grad_b1.tracker['org'])
print('\n\nw1')
print(w1)
print(grad_w1)
print(grad_w1.tracker['org'])
print('\n\na1')
print(a1)
print(grad_a1)
print(grad_a1.tracker['org'])
print('\n\na1 test')
print(a1)
print(grad_a1.to_array().squeeze())
print(np.sum(grad_r1.to_array().T * grad_a1.to_array().squeeze(), axis=0))

import matplotlib

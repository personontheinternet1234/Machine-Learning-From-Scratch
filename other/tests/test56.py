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
a1 = init.initialize(1, 5)
w1 = init.initialize(5, 4)
b1 = init.initialize(1, 4)
y = init.initialize(1, 4)

r1 = a1 @ w1
r2 = r1 + b1
r3 = act(r2)
r4 = loss(r3, y)


grad_r3 = nabla(r4, r3)
grad_r2 = chain(grad_r3, nabla(r3, r2))
grad_b1 = chain(grad_r2, nabla(r2, b1))
grad_r1 = chain(grad_r2, nabla(r2, r1))
grad_w1 = chain(grad_r1, nabla(r1, w1))
grad_a1 = chain(grad_r1, nabla(r1, a1))


print('\n\nr3')
print(r3)
print(grad_r3)
print('\n\nr2')
print(r2)
print(grad_r2)
print('\n\nr1')
print(r1)
print(grad_r1)
print('\n\nb1')
print(b1)
print(grad_b1)
print('\n\nw1')
print(w1)
print(grad_w1)
print('\n\na1')
print(a1)
print(grad_a1)
print('\na1 unchained')
print(nabla(r1, a1))
print('\n\n')

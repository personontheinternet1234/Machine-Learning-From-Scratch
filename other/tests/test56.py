from gardenpy import (
    nabla,
    chain,
    Initializers
)

init = Initializers('gaussian')
a1 = init.initialize(1, 5)
a2 = init.initialize(5, 4)
a3 = init.initialize(1, 4)

r1 = a1 @ a2
r2 = r1 + a3

grad_r1 = nabla(r1, a1)
grad_r2 = nabla(r2, r1)

print(grad_r1)
print(grad_r2)

grad_chain_f = chain(grad_r2, grad_r1)
print(grad_chain_f)
grad_chain_f.reduce()
print(grad_chain_f)

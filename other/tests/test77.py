import gardenpy as gp

init = gp.Initializers('gaussian').initialize

t1 = init(5, 5)
t2 = init(5, 5)
t3 = init(5, 5)
r1 = t1 * t2
r2 = r1 + t3
print(gp.nabla(t1, r2, binary=False))
gp.Tensor.zero_grad(t3)
print(gp.Tensor.get_instances())
t4 = init(5, 5)
print(gp.Tensor.get_instances())

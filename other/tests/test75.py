from gardenpy.utils.algorithms import Initializers
from gardenpy.utils.objects_2 import Tensor

init = Initializers('gaussian').initialize
nabla = Tensor.nabla
chain = Tensor.chain

t1 = init(5, 5)
t2 = init(5, 5)
t3 = init(5, 5)
r1 = t1 * t2
r2 = r1 + t3
nabla(r1, t1)
nabla(r2, t2)
Tensor.zero_grad(t3)
print(Tensor.get_instances())
print(t3.tracker)

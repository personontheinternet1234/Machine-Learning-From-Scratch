import numpy as np

from gardenpy.utils.objects import Tensor
from gardenpy.utils.operations import nabla, chain, test_tracker

t1 = Tensor(np.random.randn(3, 3))
t2 = Tensor(np.random.randn(3, 3))
t3 = Tensor(np.random.randn(3, 3))
t4 = Tensor(np.random.randn(3, 3))
t5 = Tensor(np.random.randn(3, 3))

print(id(t1))
print(id(t2))
print(id(t3))
print(id(t4))
print()

c1 = t1 + t2
c2 = c1 - t3
c3 = c2 * t4
print(id(c1))
print(id(c2))
print(id(c3))

# print()
# print(repr(c1))
# print(repr(c2))
# print(repr(c3))

<<<<<<< Updated upstream
print()
b = test_tracker(c3, t1)
print(b)
print()
print(repr(t1))
=======
print("calculated chains:")
f = test_tracker(c3, t1)
print([id(i) for i in f])
f = test_tracker(c3, c2)
print([id(i) for i in f])
f = test_tracker(c2, t1)
print([id(i) for i in f])
f = test_tracker(c1, t1)
print([id(i) for i in f])
>>>>>>> Stashed changes

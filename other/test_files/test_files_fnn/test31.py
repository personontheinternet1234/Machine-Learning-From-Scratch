import numpy as np

from gardenpy.utils.objects import Tensor
from gardenpy.utils.operations import test_tracker

t1 = Tensor(np.random.randn(3, 3))
t2 = Tensor(np.random.randn(3, 3))
t3 = Tensor(np.random.randn(3, 3))
t4 = Tensor(np.random.randn(3, 3))
t5 = Tensor(np.random.randn(3, 3))

print("'t' ids:")
print(f"t1: {id(t1)}")
print(f"t2: {id(t2)}")
print(f"t3: {id(t3)}")
print(f"t4: {id(t4)}")
print(f"t5: {id(t5)}")
print()

c1 = t1 + t2
c2 = c1 - t3
c3 = c2 * t4
print("'c' ids:")
print(f"c1: {id(c1)}")
print(f"c2: {id(c2)}")
print(f"c3: {id(c3)}")
print()

print("equations:")
print("c1 = t1 + t2")
print("c2 = c1 - t3")
print("c3 = c2 * t4")
print()

print("calculated chains:")
f = test_tracker(c3, t1)
print([id(i) for i in f])
f = test_tracker(c3, c1)
print([id(i) for i in f])
f = test_tracker(c2, t1)
print([id(i) for i in f])
f = test_tracker(c1, t1)
print([id(i) for i in f])

# nabla method testing to figure out what happened
from gardenpy import tensor, nabla, Initializers

init = Initializers('gaussian').initialize

t1 = init(5, 5)
t2 = init(5, 5)
r1 = t1 * t2
print(nabla(t1, r1))

t3 = init(5, 1)
t4 = init(5, 1)
r2 = t3 * t4
print(nabla(t3, r2))

# issues come from this
# it's an issue with vectors and dimensionality, which isn't super fun to work with, but at least I know now
t5 = tensor(t3.array.T)
t6 = tensor(t4.array.T)
r3 = t5 * t6
print(nabla(t5, r3))

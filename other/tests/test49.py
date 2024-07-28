from gardenpy import nabla, chain, Initializers


init = Initializers(algorithm='xavier')

t1 = init.initialize(5, 5)
t2 = init.initialize(5, 5)
t3 = init.initialize(5, 5)

t4 = t1 * t2 + t3

print(nabla(t4, t3))
print(nabla(t4, t1))

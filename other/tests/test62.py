from gardenpy import Initializers, Activators, nabla, chain

init = Initializers('gaussian').initialize
act = Activators('sigmoid').activate

t1 = init(1, 3)
t2 = init(3, 4)
t3 = init(1, 4)
print('t')
print(t1)
print(t2)
print(t3)
print('\n\n')

r1 = t1 @ t2
r2 = r1 + t3
r3 = act(r2)
print('r')
print(r1)
print(r2)
print(r3)
print('\n\n')

g1 = nabla(r3, r2)
g2 = nabla(r2, t3)
g3 = nabla(r2, r1)
g4 = nabla(r1, t1)
g5 = nabla(r1, t2)
print('g')
print(g1)
print(g2)
print(g3)
print(g4)
print(g5)
print('\n\n')

print('c')
c1 = chain(g1, g2)
c2 = chain(c1, )

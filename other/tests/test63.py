from gardenpy import Initializers, Activators, nabla, chain

init = Initializers('gaussian').initialize
act = Activators('sigmoid').activate

t1 = init(1, 3)
t2 = init(3, 4)
t3 = init(1, 4)

print('\nt')
print(f"t1 (shape(1, 3)): {t1.shape}")
print(f"t2 (shape(3, 4)): {t2.shape}")
print(f"t3 (shape(1, 4)): {t3.shape}\n")

r1 = t1 @ t2
r2 = r1 + t3

print('r')
print(f"r1 (t1 @ t2): {r1.shape}")
print(f"r2 (r1 + t3): {r2.shape}\n")

g1 = nabla(r2, t3)
g2 = nabla(r2, r1)
g3 = nabla(r1, t2)
g4 = nabla(r1, t1)

print('g')
print(f"g4 (nabla(r1, t1)): {g4.shape}")
print(f"g3 (nabla(r1, t2)): {g3.shape}")
print(f"g2 (nabla(r2, r1)): {g2.shape}")
print(f"g1 (nabla(r2, t3)): {g1.shape}\n")

c1 = nabla(r2, t2)
c2 = nabla(r2, t1)

print('c')
print(f"c2 (nabla(r2, t1)): {c2.shape}")
print(f"c1 (nabla(r2, t2)): {c1.shape}\n")

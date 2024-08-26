from gardenpy import Initializers, Losses, Activators, nabla

init = Initializers('gaussian').initialize
act = Activators('sigmoid').activate
loss = Losses('ssr').loss

t1 = init(1, 5)
t2 = init(5, 6)
t3 = init(1, 6)
e1 = init(1, 6)

r1 = t1 @ t2
r2 = r1 + t3
r3 = act(r2)

r4 = loss(r3, e1)

print(nabla(r4, t1))

print(nabla(r3, t1))

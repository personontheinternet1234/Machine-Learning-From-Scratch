from gardenpy import Initializers, Losses, Activators, Optimizers, Tensor, nabla

init = Initializers('gaussian').initialize
act = Activators('sigmoid').activate
loss = Losses('ssr').loss
optim = Optimizers('adam').optimize

x1 = init(1, 5)
w1 = init(5, 6)
b1 = init(1, 6)
w2 = init(6, 4)
b2 = init(1, 4)
e1 = Tensor([[1, 0, 0, 0]])

for i in range(100000):
    a1 = act(x1 @ w1 + b1)
    yh = act(a1 @ w2 + b2)
    cost = loss(yh, e1)

    gb2 = nabla(cost, b2)
    print(nabla(yh, w2).shape)
    gw2 = nabla(cost, w2)
    gb1 = nabla(cost, b1)
    gw1 = nabla(cost, w1)

    # w1 = optim(w1, gw1)
    b1 = optim(b1, gb1)
    # w2 = optim(w2, gw2)
    b2 = optim(b2, gb2)
    print(cost)
    print()
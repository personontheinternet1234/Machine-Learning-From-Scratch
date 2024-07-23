from gardenpy import (
    nabla,
    chain,
    Initializers,
    Losses,
    Activators
)

init = Initializers(algorithm='xavier')
act = Activators(algorithm='softmax')
loss = Losses(algorithm='centropy')

x = init.initialize(1, 5)
y = init.initialize(1, 5)
yhat = act.activate(x)
los = loss.loss(yhat, y)

print(repr(x))
print(repr(y))
print(repr(yhat))
print(repr(los))

print('----------')

dlosyhat = nabla(los, yhat)
dyhatx = nabla(yhat, x)
dlosx = nabla(los, x)
print(dlosyhat)
print(dyhatx)
print(dlosx)
print(chain(dlosyhat, dyhatx))

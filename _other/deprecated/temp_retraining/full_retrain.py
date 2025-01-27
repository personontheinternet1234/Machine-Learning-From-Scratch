import gardenpy as gp
import numpy as np

import pandas as pd

optim = gp.Optimizers('adam', lambda_d=0.01)
lrelu = gp.Activators('lrelu', beta=0.1)
# smax = gp.Activators('softmax')
init = gp.Initializers('xavier')
init_b = gp.Initializers('uniform', value=0.0)
# lss = gp.Losses('centropy')
lss = gp.Losses('ssr')

layout = [784, 256, 128, 64, 32, 16, 10]

optims = [optim] * len(layout)
act = [lrelu] * len(layout)

weights = []
biases = []
for i in range(len(layout) - 1):
    weights.append(init.initialize(layout[i], layout[i + 1]).to_array())
    biases.append(init_b.initialize(1, layout[i + 1]).to_array())

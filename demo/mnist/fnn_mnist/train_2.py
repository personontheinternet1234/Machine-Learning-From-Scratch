import numpy as np

from gardenpy.models import FNN

model = FNN()

model.initialize_model(
    # thetas=np.ones((5, 5)),
    hidden_layers=[16, 16],
    thetas=None
)

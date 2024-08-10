r"""
Dense Neural Network training on checkered or non-checkered pixels.
"""

from gardenpy import DNN

model = DNN(status_bars=True)

# todo: write a code using the fnn from gardenpy to train on the checkered and non-checkered data

# the trained model was trained with the incorrect math, so it might not work well

# run process.py to make the data if it doesn't exist for you (the data is a csv file labeled for reference

# cross-reference the training script from mnist if necessary
# note that the mnist training script has a lot of stuff, especially for the results section
# this includes stuff like violin plots, confusion matrices, data trimming, validation, and loading/saving data/models
# most of this stuff should be ignored initially, but ideally at least try and have a loss_graph done and printed results

# the other stuff could and should eventually be added at some point

# some stuff might be new, such as violin plots and confusion matrices
# if you don't understand what these graphs do, ignore them

# some settings might also be new, such as the l2 term or beta for activation functions
# if you don't know what these settings do, a large portion of the settings have automatic values that work
# if you're unsure if there are automatic values, check the fnn script

# ideally try to have a working script by sunday

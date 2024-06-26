"""
Neural Network application on Keras
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import os

import Garden.Functions.Functional as Fnc
import Garden.Functions.Formatter as Fmr
import Garden.Functions.Metrics as Mtr
from Garden.Models.FeedforwardNeuralNetwork import FeedforwardNeuralNetwork

import Keras_Data_Loading

from Garden.Extra import Credits

# locations
root = os.path.join(os.path.dirname(__file__), 'assets')
saved_location = os.path.join(root, 'saved')
data_location = os.path.join(root, 'data')

# hyperparameters
layer_sizes = [784, 16, 16, 10]
activation = 'leaky relu'
solver = 'mini-batch'
batch_size = 49
learning_rate = 0.001
max_iter = 10000
alpha = 0.001
trim_data = False
trim_frac = 0.01
set_validation = True
val_frac = 0.3
loss_reporting = True
eval_batch_size = 70
eval_interval = max(1, round(max_iter * 0.01))
load_parameters = False
weights_name = 'weights_keras.txt'
biases_name = 'biases_keras.txt'
values_name = 'data_values_keras.csv'
labels_name = 'data_labels_keras.csv'

conf_mat_normal = True

# print Garden credits
Credits.print_credits()

# import assets
if not (os.path.exists(os.path.join(data_location, values_name)) and os.path.exists(os.path.join(data_location, labels_name))):
    Keras_Data_Loading.main()
keras_values = Fmr.format_data(os.path.join(data_location, values_name))
keras_labels = Fmr.format_data(os.path.join(data_location, labels_name))
if load_parameters:
    if os.path.exists(os.path.join(saved_location, weights_name)):
        weights = Fmr.format_parameters(os.path.join(saved_location, weights_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, weights_name)} does not exist')
    if os.path.exists(os.path.join(saved_location, biases_name)):
        biases = Fmr.format_parameters(os.path.join(saved_location, biases_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, biases_name)} does not exist')
else:
    weights = None
    biases = None

# form network
keras_network = FeedforwardNeuralNetwork(weights=weights, biases=biases, layer_sizes=layer_sizes, activation=activation)
# configure loss logging
keras_network.configure_reporting(loss_reporting=loss_reporting, eval_batch_size=eval_batch_size, eval_interval=eval_interval)

# trim data
if trim_data:
    keras_values = Fnc.trim(keras_values, trim_frac=trim_frac)
    keras_labels = Fnc.trim(keras_labels, trim_frac=trim_frac)

# set validation
if set_validation:
    x, y, val_x, val_y = Fnc.test_val(keras_values, keras_labels, val_frac=val_frac)
    keras_network.validation(valid_x=val_x, valid_y=val_y)
else:
    x, y = keras_values, keras_labels

# train model
keras_network.fit(
    x=x,
    y=y,
    solver=solver,
    batch_size=batch_size,
    learning_rate=learning_rate,
    max_iter=max_iter,
    alpha=alpha,
)

# get model results
keras_results = keras_network.get_results(cm_norm=conf_mat_normal)
# Mtr.print_color(keras_results)
Mtr.print_final_results(keras_results)
# graph results
if conf_mat_normal:
    Mtr.prob_visual_cm((keras_results['training confusion matrix']), title='Training Results')
    Mtr.prob_visual_cm((keras_results['validation confusion matrix']), title='Validation Results')
else:
    Mtr.num_visual_cm((keras_results['training confusion matrix']), title='Training Results')
    Mtr.num_visual_cm((keras_results['validation confusion matrix']), title='Validation Results')
Mtr.loss_graph(keras_results['logged losses'])
Mtr.reg_loss_graph(keras_results['logged losses'])
Mtr.prob_violin_plot(keras_results['training outcomes'], title='Training Violin Plot')
if set_validation:
    Mtr.prob_violin_plot(keras_results['validation outcomes'], title='Validation Violin Plot')

# save model
save_parameters = Mtr.input_color("input 's' to save parameters: ")
if save_parameters.lower() == 's':
    Fmr.save_parameters(os.path.join(saved_location, weights_name), keras_results['weights'])
    Fmr.save_parameters(os.path.join(saved_location, biases_name), keras_results['biases'])
else:
    Mtr.print_color('parameters not saved')

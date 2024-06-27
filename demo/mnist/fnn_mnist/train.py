"""
Feedforward Neural Network training for MNIST
"""

import os

from garden.metrics import (
    visualization as vis
)
from garden.utils import (
    data_utils as du,
    helper_functions as hf
)

from garden.models.fnn import (
    FNN
)

import data.process as p

# locations
root = os.path.join(os.path.dirname(__file__))
saved_location = os.path.join(root, 'model')
data_location = os.path.join(root, 'data', 'processed_data')

# hyperparameters
layer_sizes = [784, 16, 16, 10]
activation = 'leaky relu'
solver = 'mini-batch'
batch_size = 49
learning_rate = 0.001
max_iter = 1000
alpha = 0.001
trim_data = False
trim_frac = 0.01
set_validation = True
val_frac = 0.3
loss_reporting = True
eval_batch_size = 70
eval_interval = max(1, round(max_iter * 0.01))
load_parameters = True
weights_name = 'weights.txt'
biases_name = 'biases.txt'
save_data = True
values_name = 'values.csv'
labels_name = 'labels.csv'

conf_mat_normal = True

# print garden credits
hf.print_credits()

# import assets
weights = None
biases = None
if not (os.path.isfile(os.path.join(data_location, values_name)) and os.path.isfile(os.path.join(data_location, labels_name))):
    values_file = os.path.join(data_location, values_name)
    labels_file = os.path.join(data_location, labels_name)
    keras_values, keras_labels = p.process_data(save=save_data, values_file=values_file, labels_file=labels_file)
else:
    keras_values = du.format_data(os.path.join(data_location, values_name))
    keras_labels = du.format_data(os.path.join(data_location, labels_name))
if load_parameters:
    if os.path.exists(os.path.join(saved_location, weights_name)):
        weights = du.format_parameters(os.path.join(saved_location, weights_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, weights_name)} does not exist')
    if os.path.exists(os.path.join(saved_location, biases_name)):
        biases = du.format_parameters(os.path.join(saved_location, biases_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, biases_name)} does not exist')

# form network
keras_network = FNN(weights=weights, biases=biases, layer_sizes=layer_sizes, activation=activation)
# configure loss logging
keras_network.configure_reporting(loss_reporting=loss_reporting, eval_batch_size=eval_batch_size, eval_interval=eval_interval)

# trim processed_data
if trim_data:
    keras_values = du.trim(keras_values, trim_frac=trim_frac)
    keras_labels = du.trim(keras_labels, trim_frac=trim_frac)

# set validation
if set_validation:
    x, y, val_x, val_y = du.test_val(keras_values, keras_labels, val_frac=val_frac)
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
vis.print_results(keras_results)
# graph results
vis.cm_disp((keras_results['training confusion matrix']), title='Training Results', normalized=conf_mat_normal)
vis.cm_disp((keras_results['validation confusion matrix']), title='Validation Results', normalized=conf_mat_normal)
vis.loss_graph(keras_results['logged losses'])
vis.reg_loss_graph(keras_results['logged losses'])
vis.violin_plot(keras_results['training outcomes'], title='Training Violin Plot')
if set_validation:
    vis.violin_plot(keras_results['validation outcomes'], title='Validation Violin Plot')

# save model
save_parameters = hf.input_color("input 's' to save parameters: ")
if save_parameters.lower() == 's':
    du.save_parameters(os.path.join(saved_location, weights_name), keras_results['weights'])
    du.save_parameters(os.path.join(saved_location, biases_name), keras_results['biases'])
else:
    hf.print_color('parameters not saved')

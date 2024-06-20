"""
Neural Network application on Keras
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

from Garden.Extra import Credits
from Garden.Models.NeuralNetwork import NeuralNetwork
import Garden.Functions.Functional as Fnc
import Garden.Functions.Formatter as Fmr
import Garden.Functions.Metrics as Mtr

# hyperparameters
layer_sizes = [784, 16, 16, 10]
activation = 'leaky relu'
solver = 'mini-batch'
batch_size = 42
learning_rate = 0.001
max_iter = 5000
alpha = 0.001
trim_data = False
trim_frac = 0.01
set_validation = True
val_frac = 0.3
loss_reporting = True
eval_batch_size = 60
eval_interval = max(1, round(max_iter * 0.001))
load_parameters = False
weights_location = 'assets/saved/weights_keras.txt'
biases_location = 'assets/saved/biases_keras.txt'
values_location = 'assets/data/data_values_keras.csv'
labels_location = 'assets/data/data_labels_keras.csv'
conf_mat_normal = True

# print Garden credits
Credits.print_credits()

# import assets
keras_values = Fmr.format_data(values_location)
keras_labels = Fmr.format_data(labels_location)
if load_parameters:
    weights = Fmr.format_parameters(weights_location)
    biases = Fmr.format_parameters(biases_location)
else:
    weights = None
    biases = None

# form network
keras_network = NeuralNetwork(weights=weights, biases=biases, layer_sizes=layer_sizes, activation=activation)
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
Mtr.kde_loss_graph(keras_results['logged losses'])
Mtr.prob_violin_plot(keras_results['training outcomes'], title='Training Violin Plot')
if set_validation:
    Mtr.prob_violin_plot(keras_results['validation outcomes'], title='Validation Violin Plot')

# save model
save_parameters = Mtr.input_color("input 's' to save parameters: ")
if save_parameters.lower() == 's':
    Fmr.save_parameters(weights_location, keras_results['weights'])
    Fmr.save_parameters(biases_location, keras_results['biases'])
else:
    Mtr.print_color('parameters not saved')

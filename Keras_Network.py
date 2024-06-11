from Garden import Credits
from Garden.Models.NeuralNetwork import NeuralNetwork
import Garden.Functions.Functional as Fnc
import Garden.Functions.Formatter as Fmt
import Garden.Functions.Metrics as Mtr

# print credits
Credits.print_credits()

# imports
keras_values = Fmt.format_data('assets/data/data_values_keras.csv')
keras_labels = Fmt.format_data('assets/data/data_labels_keras.csv')

# data splitting
x, y, val_x, val_y = Fnc.test_val(keras_values, keras_labels, val_frac=0.3)

# network setup
keras_network = NeuralNetwork(layer_sizes=[784, 16, 16, 10], activation='leaky relu')
keras_network.configure_reporting(loss_reporting=True, eval_interval=100)
keras_network.validation(valid_x=val_x, valid_y=val_y)
# network training
keras_network.fit(x, y, max_iter=1000)
# network results
keras_results = keras_network.get_results()
# graph results
# todo
Mtr.prob_visual_cm((keras_results['training confusion matrix']), title='Training Results')
Mtr.prob_visual_cm((keras_results['validation confusion matrix']), title='Validation Results')
Mtr.loss_graph(keras_results['train losses'], keras_results['logged points'])
Mtr.violin_plot()
Mtr.violin_plot()


"""
Functions for calculating and visualising results
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

GREEN = '\033[32m'


def print_color(text, color_code=GREEN, end='\n'):
    """ print text in color """
    print(f'{color_code}{text}\033[0m', end=end)


def input_color(text, color_code=GREEN):
    """ print input statement in color """
    return input(f'{color_code}{text}\033[0m')


def generate_cm(label, predicted, normalize=True):
    """ generate a confusion matrix based on predictions """
    # find errors
    if len(label) != len(predicted):
        raise ValueError(f"'({len(label)})' is not the same as '{len(predicted)}'")

    # instantiate confusion matrix
    cm = []
    for i in range(len(set(label))):
        cm.append([0] * len(set(predicted)))

    # generate confusion matrix
    for i in range(len(label)):
        cm[label[i] - 1][predicted[i] - 1] += 1

    # normalize confusion matrix
    if normalize:
        for i in range(len(cm)):
            row_sum = np.sum(cm[i])
            if row_sum != 0:
                cm[i] = cm[i] / row_sum
            else:
                cm[i] = cm[i]

    # reformat and return confusion matrix
    cm = np.array(cm)
    return cm


def loss_graph(logged_losses, title='Loss vs. Iteration', x_axis='Iteration', y_axis='Loss', limits='auto'):
    """ loss vs iteration graph from results dataframe """
    # initialize data dictionary
    data = {
        'Label': ['Training' for _ in range(len(logged_losses['logged points']))],
        'Iteration': logged_losses['logged points'].tolist(),
        'Loss': logged_losses['training losses'].tolist()
    }
    # add validation if necessary
    if 'validation losses' in logged_losses.columns:
        for i in range(len(logged_losses['logged points'])):
            data['Label'].append('Validation')
            data['Iteration'].append(logged_losses['logged points'].tolist()[i])
            data['Loss'].append(logged_losses['validation losses'].tolist()[i])
    data = pd.DataFrame(data)
    # make graph
    ax = sns.lineplot(data, x='Iteration', y='Loss', hue='Label', style='Label', alpha=0.75, palette='Greens_d')
    # set graph limits
    if limits == 'auto':
        limits = [[None, 0], [None, None]]
    elif isinstance(limits, list):
        limits = limits
    else:
        raise TypeError(f'{limits} is not properly formatted')
    ax.set_ylim(limits[0][1], limits[1][1])
    ax.set_xlim(limits[0][0], limits[1][0])
    # set axis labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    # plot graph
    plt.show()


def reg_loss_graph(logged_losses, title='Loss vs. Iteration Regression Fit', x_axis='Iteration', y_axis='Loss', frac=0.1, limits='auto'):
    """ lowess regression line of loss vs iteration graph from results dataframe """
    # set lowess predictor
    lowess = sm.nonparametric.lowess
    # initialize data dictionary
    data_train = {
        'Label': ['Training' for _ in range(len(logged_losses['logged points']))],
        'Iteration': logged_losses['logged points'].tolist(),
        'Loss': logged_losses['training losses'].tolist()
    }
    data_valid = None
    fit_valid = None
    # add validation if necessary
    has_valid = 'validation losses' in logged_losses
    if has_valid:
        for i in range(len(logged_losses['logged points'])):
            data_valid = {
                'Iteration': logged_losses['logged points'].tolist(),
                'Loss': logged_losses['validation losses'].tolist()
            }
    # make scatters and lowess predicted
    sns.scatterplot(x=data_train['Iteration'], y=data_train['Loss'], label='Training', marker='x', linewidth=1, alpha=0.5, color='#75b377')
    fit_train = lowess(data_train['Loss'], data_train['Iteration'], frac=frac)
    if has_valid:
        sns.scatterplot(x=data_valid['Iteration'], y=data_valid['Loss'], label='Validation', marker='x', linewidth=1, alpha=0.5, color='#497a4f')
        fit_valid = lowess(data_valid['Loss'], data_train['Iteration'], frac=frac)
    # plot lowess
    plt.plot(fit_train[:, 0], fit_train[:, 1], label='Training', alpha=0.75, color='#75b377')
    if has_valid:
        plt.plot(fit_valid[:, 0], fit_valid[:, 1], label='Validation', linestyle='--', alpha=0.75, color='#497a4f')
    # set graph limits
    if limits == 'auto':
        limits = [[None, 0], [None, None]]
    elif isinstance(limits, list):
        limits = limits
    else:
        raise TypeError(f'{limits} is not the correct datatype (list)')
    plt.ylim(limits[0][1], limits[1][1])
    plt.xlim(limits[0][0], limits[1][0])
    # set axis labels
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend(title='Label')
    plt.title(title)
    # plot graph
    plt.show()


def prob_visual_cm(cm, title=None, x_labels=None, y_labels=None, x_axis='Predicted', y_axis='True', annotation=True):
    """ visual confusion matrix from probabilities """
    if cm is None:
        return 1
    # set labels
    if not x_labels:
        x_labels = [i for i in range(0, len(cm))]
    if not y_labels:
        y_labels = [i for i in range(0, len(cm[0]))]
    # make heatmap
    ax = sns.heatmap(cm, annot=annotation, xticklabels=x_labels, yticklabels=y_labels, linewidth=.5, cmap='Greens', vmin=0, vmax=1)
    # set axis labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    # plot heatmap
    plt.show()


def num_visual_cm(cm, title=None, x_labels=None, y_labels=None, x_axis='Predicted', y_axis='True', annotation=True):
    """ visual confusion matrix from numbers """
    if cm is None:
        return 1
    # set labels
    if not x_labels:
        x_labels = [i for i in range(0, len(cm))]
    if not y_labels:
        y_labels = [i for i in range(0, len(cm[0]))]
    # make heatmap
    ax = sns.heatmap(cm, annot=annotation, xticklabels=x_labels, yticklabels=y_labels, linewidth=.5, cmap='Greens')
    # set axis labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    # plot heatmap
    plt.show()


def prob_violin_plot(df, title=None, x_axis='Label', y_axis='Maximum Probability'):
    """ violin plot from results dataframe """
    # initialize data dictionary
    data = {
        'Label': df['label'].tolist(),
        'Maximum Probability': [],
        'Predicted': df['accurate'].tolist()
    }
    # calculate maximum probabilities from softmax
    softmax = df['probability'].tolist()
    for i in range(len(softmax)):
        data['Maximum Probability'].append(max(softmax[i]))
    data = pd.DataFrame(data)
    # make violin plot
    ax = sns.violinplot(data, x='Label', y='Maximum Probability', hue='Predicted', hue_order=[True, False], split=True, gap=0.1, inner='quart', density_norm='count', palette='Greens_d')
    # set axis labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    # plot violin plot
    plt.show()


def print_final_results(results_dict, round_value=5):
    """ print final results of model from results dataframe """
    # evaluate final results
    results_ser = results_dict['final results']
    train_loss = round(results_ser['mean training loss'], round_value)
    train_accu = round(results_ser['training accuracy'], round_value)
    val_loss = None
    val_accu = None
    elapsed_time = round(results_ser['elapsed training time'], round_value)
    if ('mean validation results' and 'validation accuracy') in results_ser:
        val_loss = round(results_ser['mean validation loss'], round_value)
        val_accu = round(results_ser['validation accuracy'], round_value)
    # print final results
    print_color(f'Results - Train Loss: {train_loss} - Valid Loss: {val_loss} - Train Accuracy: {train_accu} - Valid Accuracy: {val_accu} - Training Time: {elapsed_time}s')

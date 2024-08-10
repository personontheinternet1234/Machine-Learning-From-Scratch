# todo: recode

"""
result visualization functions
"""

import pandas as pd

'''
Works for Derek's computer
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from gardenpy.utils.helper_functions import print_color

from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def loss_graph(logged_losses, title='Loss vs. Iteration', x_axis='Iteration', y_axis='Loss', limits='auto'):
    """ loss vs iteration graph from results dataframe """
    # initialize processed_data dictionary
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
    elif limits == 'none':
        limits = [[None, None], [None, None]]
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
    # initialize processed_data dictionary
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
    elif limits == 'none':
        limits = [[None, None], [None, None]]
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


def cm_disp(cm, title=None, x_labels=None, y_labels=None, x_axis='Predicted', y_axis='True', normalized=True, annotation=True):
    """ visual confusion matrix in a heatmap """
    if cm is None:
        return 1
    # set labels
    if not x_labels:
        x_labels = [i for i in range(0, len(cm))]
    if not y_labels:
        y_labels = [i for i in range(0, len(cm[0]))]
    # make heatmap
    if normalized:
        ax = sns.heatmap(cm, annot=annotation, xticklabels=x_labels, yticklabels=y_labels, linewidth=.5, cmap='Greens', vmin=0, vmax=1)
    else:
        ax = sns.heatmap(cm, annot=annotation, xticklabels=x_labels, yticklabels=y_labels, linewidth=.5, cmap='Greens')
    # set axis labels
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)
    # plot heatmap
    plt.show()


def violin_plot(df, title=None, x_axis='Label', y_axis='Maximum Probability'):
    """ violin plot from results dataframe """
    # initialize processed_data dictionary
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


def print_results(results_dict, sig_value=5):
    # todo: change rounding to use sig figs
    """ print final results of model from results dataframe """
    # evaluate final results
    results_ser = results_dict['final results']
    train_loss = round(results_ser['mean training loss'], sig_value)
    train_accu = round(results_ser['training accuracy'], sig_value)
    val_loss = None
    val_accu = None
    elapsed_time = round(results_ser['elapsed training time'], sig_value)
    if ('mean validation results' and 'validation accuracy') in results_ser:
        val_loss = round(results_ser['mean validation loss'], sig_value)
        val_accu = round(results_ser['validation accuracy'], sig_value)
    # print final results
    print_color(f'Results - Train Loss: {train_loss} - Valid Loss: {val_loss} - Train Accuracy: {train_accu} - Valid Accuracy: {val_accu} - Training Time: {elapsed_time}s')
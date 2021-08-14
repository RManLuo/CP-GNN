#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/8 15:14
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : helper.py
# @Software: PyCharm
# @Describe:
import torch
import os
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt
from .evluator import Evaluator
import json
import numpy as np
import matplotlib


def to_device(tensor_list, device):
    return [tensor.to(device) for tensor in tensor_list]


def get_logits(p_emb, contex_emb, pos_g, neg_g):
    pos_edges = pos_g.edges()
    neg_edges = neg_g.edges()
    src = torch.cat([pos_edges[0], neg_edges[0]])
    dst = torch.cat([pos_edges[1], neg_edges[1]])
    labels = torch.cat([torch.ones_like(pos_edges[0]), torch.zeros_like(neg_edges[1])]).float()
    logits = torch.sum((p_emb[src] * contex_emb[src]) * (p_emb[dst] * contex_emb[dst]), dim=1)
    return logits, labels


def load_latest_model(checkpoint_path, model):
    files = os.listdir(checkpoint_path)
    get_time = lambda file: os.path.getmtime(os.path.join(checkpoint_path, file))
    files.sort(key=get_time, reverse=True)
    try:
        latest_model_path = os.path.join(checkpoint_path, files[0])
    except:
        return model
    if os.path.isdir(latest_model_path):
        latest_model_path = os.path.join(latest_model_path, 'model.pth')
    print("Load model: ", latest_model_path)
    model.load_state_dict(torch.load(latest_model_path))
    return model


def evaluate(p_emb, CF_data, LP_data, method, metric=['CF', 'LP', 'CL'], save_result=False, result_path='./result',
             random_state=123, max_iter=150,
             n_jobs=1):
    evaluator = Evaluator(method, CF_data, LP_data, result_path, random_state, max_iter, n_jobs)
    if 'CF' in metric:
        evaluator.evluate_CF(p_emb)
    if 'LP' in metric:
        evaluator.evluate_LP(p_emb)
    if 'CL' in metric:
        evaluator.evluate_CL(p_emb)
    if save_result:
        return evaluator.dump_result(p_emb, metric)
    return None


def save_attention_matrix(model, path, K):
    attention_matrix_path = os.path.join(path, 'atten.json')
    loss_weight_softmax = torch.softmax(torch.exp(-model.loss_weight), dim=0).detach().cpu().numpy()
    np.savetxt(os.path.join(path, 'raw_length_attention.txt'), model.loss_weight.detach().cpu().numpy(), fmt='%.03f')
    np.savetxt(os.path.join(path, 'softmax_length_attention.txt'), loss_weight_softmax, fmt='%.03f')
    model.eval()
    with torch.no_grad():
        for k in range(1, K + 2):
            model(k)
    model.dump_cgnn_attention_matrix(attention_matrix_path)
    print("Attention matrix saved in {}".format(attention_matrix_path))
    return attention_matrix_path


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=18)
    ax.set_yticklabels(row_labels, fontsize=18)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels())

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, "%.2f" % data[i, j], **kw)
            texts.append(text)

    return texts


def generate_attention_heat_map(node_types, attention_matrix_path):
    with open(attention_matrix_path) as f:
        attention_matrix_dict = json.load(f)
    x_label = node_types
    y_label = node_types
    for hop, hop_attention_matrix in attention_matrix_dict.items():
        data = []
        for x in x_label:
            temp = []
            for y in y_label:
                key = x + y
                value = 0
                for _, head in hop_attention_matrix.items():
                    value = value + head.get(key, 0)  # Sum the value of each head
                temp.append(value)
            data.append(temp)
        fig, ax = plt.subplots()
        Y_label = [i.upper() for i in y_label]
        X_label = [i.upper() for i in x_label]
        data = np.array(data)
        im, _ = heatmap(data, Y_label, X_label, ax=ax, vmin=0,
                        cmap="magma_r",
                        cbarlabel="Relation Attention Matrix of {} hop Context".format(hop.split('_')[-1]))
        annotate_heatmap(im, valfmt="{x:d}", size=15, threshold=20,
                         textcolors=["red", "white"])
        fig.tight_layout()
        figure_path = os.path.join(os.path.dirname(attention_matrix_path), "{}-length.png".format(hop.split('_')[-1]))
        plt.savefig(figure_path)
        # show
        plt.show()


def save_config(config, path):
    config_path = os.path.join(path, "config.py")
    with open(config_path, 'w') as f:
        f.write('data_config =' + str(config.data_config))
        f.write("\n")
        f.write('model_config =' + str(config.model_config))
        f.write("\n")
        f.write('train_config =' + str(config.train_config))
        f.write("\n")
        f.write('evaluate_config =' + str(config.evaluate_config))

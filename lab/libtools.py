#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def downsample_matrix(matrix, factor):
    return matrix[::factor, ::factor]

def plot_gradient(y, colormap, sz=10, edge_width=0.5):
    x = np.arange(len(y))
    y_norm = y + abs(y.min())
    y_norm *= 1/y_norm.max()
    colors = plt.colormaps[colormap](y_norm)
    for i in range(len(x) - 1):
        plt.plot(x[i], y[i], '.', color=colors[i], mec='k', ms=sz, mew=edge_width)
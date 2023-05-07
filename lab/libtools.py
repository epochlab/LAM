#!/usr/bin/env python3

import itertools
import numpy as np
import matplotlib.pyplot as plt

def construct(src, r=5, sigmaX=4.0, sigmaA=1.0):
    tmp = itertools.product(range(src.shape[0]), range(src.shape[1]))
    combi_all = itertools.combinations(tmp, 2)
    combi = [x for x in combi_all if np.sqrt((x[0][0] - x[1][0])**2 + (x[0][1] - x[1][1])**2) < r]

    edgelist_w = []
    for x1, x2 in combi:
        n1 = int(x1[0] * src.shape[1] + x1[1])
        n2 = int(x2[0] * src.shape[1] + x2[1])
        dspace = (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2
        dangle = np.pi - np.fabs(np.pi - np.fabs(src[x1[0],x1[1]] - src[x2[0],x2[1]]))
        corr = np.exp(-(dspace**2/sigmaX + dangle**2/sigmaA))
        edgelist_w.append((n1, n2, corr))

    P = src.shape[0] * src.shape[1]
    W = np.zeros([P, P])
    for x in edgelist_w:
        W[x[0],x[1]] = x[2]
        W[x[1],x[0]] = x[2]
    return W

def GL_eigen(W, norm_mode='asym'):
    if norm_mode == 'sym':
        Dnorm = np.diag(np.sum(W, axis=1)**-0.5)
    elif norm_mode == 'asym':
        Dnorm = np.diag(np.sum(W, axis=1)**-1)
        
    L = np.eye(W.shape[0]) - Dnorm @ W
    e, v = np.linalg.eig(L)
    e = np.real(e)
    v = np.real(v)
    order = np.argsort(e)
    e = e[order]
    v = v[:,order]
    return e, v

def downsample_matrix(matrix, factor):
    return matrix[::factor, ::factor]

def gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def plot_gradient(y, colormap, sz=10, edge_width=0.5):
    x = np.arange(len(y))
    y_norm = y + abs(y.min())
    y_norm *= 1/y_norm.max()
    colors = plt.colormaps[colormap](y_norm)
    for i in range(len(x) - 1):
        plt.plot(x[i], y[i], '.', color=colors[i], mec='k', ms=sz, mew=edge_width)
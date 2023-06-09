#!/usr/bin/env python3

import itertools, math
import numpy as np
import matplotlib.pyplot as plt

def construct_LAM(src, r=5, sigmaX=4.0, sigmaI=0.1):
    tmp = itertools.product(range(src.shape[0]), range(src.shape[1]))
    combi_all = itertools.combinations(tmp, 2)
    combi = [x for x in combi_all if np.sqrt((x[0][0] - x[1][0])**2 + (x[0][1] - x[1][1])**2) < r]

    edgelist_w = []
    for x1, x2 in combi:
        n1 = int(x1[0] * src.shape[1] + x1[1])
        n2 = int(x2[0] * src.shape[1] + x2[1])
        intensity = np.exp(-np.sum((src[x1[0],x1[1]] - src[x2[0],x2[1]])**2) / (sigmaI**2))
        spatial = np.exp(-((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2) / (sigmaX**2))
        corr = intensity * spatial
        edgelist_w.append((n1, n2, corr))

    P = src.shape[0] * src.shape[1]
    W = np.zeros([P, P])
    for x in edgelist_w:
        W[x[0],x[1]] = x[2]
        W[x[1],x[0]] = x[2]
    return W

def construct_SLAM(src, sigmaX=1.0, sigmaA=0.1):
    tmp = itertools.product(range(src.shape[0]), range(src.shape[1]))
    combi = itertools.combinations(tmp, 2)

    x_vals = np.linspace(0, 1, src.shape[1])
    y_vals = np.linspace(0, 1, src.shape[0])

    edgelist_w = []
    for x1, x2 in combi:
        p1 = (x_vals[x1[1]], y_vals[x1[0]])
        p2 = (x_vals[x2[1]], y_vals[x2[0]])

        dspace = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        dangle = np.pi - np.fabs(np.pi - np.fabs(src[x1[0],x1[1]] - src[x2[0],x2[1]]))
        corr = np.exp(-(dspace**2/sigmaX + dangle**2/sigmaA))

        n1 = int(x1[0] * src.shape[1] + x1[1])
        n2 = int(x2[0] * src.shape[1] + x2[1])
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

def downsample(mat, factor):
    return mat[::factor, ::factor]

def gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    # return np.exp(-(dspace**2/sigma))

def gabor_filter(sigma_x, sigma_y, deg, samples=20, k=2, min=-5, max=5):
    gradient = np.linspace(min, max, samples)
    X, Y = np.meshgrid(gradient, gradient)

    rad = np.deg2rad(deg)
    X = X * np.cos(rad) - Y * np.sin(rad)
    Y = X * np.sin(rad) + Y * np.cos(rad)

    C = 1 / (2 * math.pi * sigma_x * sigma_y)
    z = C * np.exp(-(X**2) / (2 * sigma_x**2) - (Y**2) / (2 * sigma_y**2))
    gabor = np.cos(X * k) * z
    return gabor

def gabor_conv(img, src, step=20, k_size=5):
    pad = int(step/2)
    pad_im = np.pad(img, pad, mode='constant', constant_values=0)
    features = np.zeros_like(pad_im)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            atan = src[i][j] - np.pi # Arc tangent | -π and π
            deg = np.rad2deg(atan*0.5) # *0.5 to keep range between -90 and 90
            kernel = gabor_filter(2, 1, deg, samples=step, min=-k_size, max=k_size) # Orientation
            patch = pad_im[i:i+step, j:j+step]
            features[int(i+pad),int(j+pad)] = np.sum(patch * kernel) # Firing Rate / Response

    features = features[pad:pad+src.shape[0],pad:pad+src.shape[1]]
    features[features<=0] = 0 # Rectify

    return features

def activation_prob(x, temp):
    p = 1 / (1.0 + np.exp(-x/temp))
    y = (np.random.rand(*x.shape) < p) * 1.0
    return y

def HSV2RGB(h,s,v):
    i = np.floor(h*6)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    imod = i%6
    if(imod==0): return p,q,v
    elif(imod==1): return t,p,v
    elif(imod==2): return v,p,q
    elif(imod==3): return v,t,p
    elif(imod==4): return q,v,p
    elif(imod==5): return p,v,t

def plot_gradient(y, colormap, sz=10, edge_width=0.5):
    x = np.arange(len(y))
    y_norm = y + abs(y.min())
    y_norm *= 1/y_norm.max()
    colors = plt.colormaps[colormap](y_norm)
    for i in range(len(x) - 1):
        plt.plot(x[i], y[i], '.', color=colors[i], mec='k', ms=sz, mew=edge_width)
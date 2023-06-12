#!/usr/bin/env python3

import libtools

import numpy as np
from tqdm import tqdm

class LAM():
    """
    LAPLACIAN ASSOCIATIVE MEMORY (LAM)
    """
    def __init__(self, N, P, prob, H, gamma, norm_mode):
        self.N = N                      # Neurons (n)
        self.P = P                      # Random memory patterns (n)
        self.prob = prob                # Sparsity (Activation Probability)
        self.H = H                      # Hetero-associative weights
        self.gamma = gamma              # Inhibition ratio (Regularisation)
        self.norm_mode = norm_mode      # Normalisation

        self.V = self.prob * (1-self.prob)
        self.NV = self.N * self.V
        
        # BINARY STATE VECTORS
        self.xi = (np.random.rand(self.N, self.P) < self.prob).astype('float') # Binary dipole (+/-) input with sparsity
        self.xi_mean = np.sum(self.xi, axis=1, keepdims=True) / self.P # Mean activation of each neuron across all inputs
        self.xi_bias = self.xi - self.xi_mean

        # NORMALIZATION
        if self.norm_mode == "sym": # SYMMETRIC WEIGHTS
            Dnorm = np.diag(np.sum(self.H, axis=1)**-0.5)
            self.H = Dnorm @ self.H @ Dnorm
            self.Wauto = (self.xi_bias @ self.xi_bias.T) / self.NV
            self.Whetero = (self.xi_bias @ self.H @ self.xi_bias.T) / self.NV
            self.WG = self.gamma / self.N

        elif self.norm_mode == "asym": # ASYMMETRIC WEIGHTS
            Dnorm = np.diag(np.sum(self.H, axis=1)**-1) # Degree matrix (D**-1*A)
            self.H = Dnorm @ self.H # Hetero-associative weights
            self.Wauto = (self.xi @ self.xi.T) / self.NV # Local inhibition
            self.Whetero = (self.xi @ self.H @ self.xi.T) / self.NV # Excitatory
            self.WG = self.P * self.xi_mean @ self.xi_mean.T / self.NV + self.gamma / self.N # Global inhibition
        else:
            print("Error: Normalization mode 'sym' or 'asym' was not specified.")
            exit()

    def _step(self, z): # Heaviside step function
        return 0.5 * np.sign(z) + 0.5

    def _set_weight(self, a): # Decompose weights
        self.W = a * self.Wauto + self.Whetero - (a+1) * self.WG

    def _set_state(self, features, temp):
        M = np.size(self.xi, axis=1) # Nodes
        I = np.zeros_like(self.xi) # Malloc

        for node in range(M):
            k = features.flatten()[node] # Gabor response
            state = (self.xi[:, node].copy() * 2) - 1 # State of each node and re-map between -1 and 1
            state *= k # Multiply against response
            I[:,node] = state

        Inorm = np.sum(I, axis=1) * 1/M
        init = self._boltzmann_prob(Inorm, temp)
        return init

    def _boltzmann_prob(self, x, temp):
        p = 1 / (1.0 + np.exp(-x/temp))
        y = (np.random.rand(*x.shape) < p) * 1.0
        return y

    def _kronecker_delta(self, i, j):
        return 1 if i==j else 0

    def simulate_single(self, a, eta, simlen, start_node, cond=False, energycheck=True):
        self._set_weight(a) # Set weight based on alpha

         # Init network state
        if cond==False:
            self.x = self.xi[:, start_node] + 0.0
        else:
            self.x = start_node
        
        self.m_log = np.zeros([simlen, self.P])
        self.obj_log = np.zeros([simlen])

        for t in tqdm(range(simlen)):
            self.r = self._step(self.W @ self.x) # Threshold activation (Response) - Input to each neuron, dot product of weight matrix (self.W) and current network state (self.x)
            self.x += eta * (self.r - self.x) # Network update - Simple gradient descent, updating neuron activity as a weighted (eta) average of previous activity (x) to current input (r)
            self.m = (self.xi_bias.T @ self.x) / self.NV # Pattern overlap / magnetisation - A measure of similarity between the state of the neuron and the average state of its neighbours in the network.
            self.m_log[t,:] = self.m # Update log

            if energycheck:
                self.obj_log[t] = -(self.x).T @ self.W @ self.x / self.NV # Compute energy

        return (self.m_log, self.obj_log)
    
    def simulate_allstarts(self, a, eta, simlen):
        self._set_weight(a)
        self.x = self.xi + 0.0
        self.m_log = np.zeros([simlen,self.P,self.P])

        for t in tqdm(range(simlen)):
            self.r = self._step(self.W @ self.x)
            self.x += eta * (self.r - self.x)
            self.m = (self.xi_bias.T @ self.x) / (self.N * self.V)
            self.m_log[t,:,:] = self.m

        self.cor_activity = np.corrcoef(self.x.T) # Correlation between attractors
        return (self.m_log, self.cor_activity)
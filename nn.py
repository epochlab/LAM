#!/usr/bin/env python3

import numpy as np

class hopfield():
    """
    HOPFIELD NETWORK
    """
    def __init__(self, ndim):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))

    def _step(self, x):
        return np.where(x > 0, 1, -1)

    def _compute_energy(self, state):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)

    def train(self, data):
        for sample in data:
            memory = np.array([sample])
            self.weights += memory * memory.T
            self.weights = (self.weights + self.weights.T) / 2
        self.weights /= data.shape[0]
        np.fill_diagonal(self.weights, 0)

    def infer(self, state, units):
        idx = np.random.randint(0, self.ndim, size=units)
        spin = np.dot(self.weights[idx,:], state)
        state[idx] = self.step(spin)
        return state

class LAM():
    """
    LAPLACIAN ASSOCIATIVE MEMORY (LAM)
    """
    def __init__(self, N, P, prob, H, gamma, norm_mode):
        self.N = N
        self.P = P
        self.prob = prob
        self.H = H
        self.gamma = gamma
        self.norm_mode = norm_mode

        self.V = self.prob * (1-self.prob)
        self.NV = self.N * self.V
        
        # GENERATE PATTERNS
        self.xi = (np.random.rand(self.N, self.P) < self.prob).astype('float')
        self.xi_mean = np.sum(self.xi, axis=1, keepdims=True) / self.P
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

    def simulate_single(self, a, eta, epochs, start_node, energycheck=True):
        self._set_weight(a)
        self.x = self.xi[:, start_node] + 0.0
        self.m_log = np.zeros([epochs, self.P])
        self.obj_log = np.zeros([epochs])

        for t in range(epochs):
            self.r = self._step(self.W @ self.x) # Threshold activation
            self.x += eta * (self.r - self.x) 
            self.m = (self.xi_bias.T @ self.x) / self.NV # Pattern overlap
            self.m_log[t,:] = self.m

            if energycheck:
                self.obj_log[t] = -(self.x).T @ self.W @ self.x / self.NV # Compute energy

        return (self.m_log, self.obj_log)
    
    def simulate_allstarts(self, a, eta, simlen):
        self._set_weight(a)
        self.x = self.xi + 0.0
        self.m_log = np.zeros([simlen,self.P,self.P])

        for t in range(simlen):
            self.r = self._step(self.W @ self.x)
            self.x += eta * (self.r - self.x)
            self.m = (self.xi_bias.T @ self.x) / (self.N * self.V)
            self.m_log[t,:,:] = self.m

        self.cor_activity = np.corrcoef(self.x.T) # Correlation between attractors
        return (self.m_log, self.cor_activity)
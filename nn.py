#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))

    def hopfield(self, data, threshold=0.1):
        for _, sample in enumerate(data):
            memory = np.array([np.where(sample > threshold, 1, -1)]) # Binary dipole (+/-)
            delta_weights = memory.T * memory
            self.weights += delta_weights
            self.weights = (self.weights + self.weights.T) / 2 # Enforce symmetrical weights
        self.weights /= data.shape[0] # Normalise against dataset
        np.fill_diagonal(self.weights, 0) # Set the diagonal to zero - ensure nodes to influence themselves

    def infer(self, state, units):
        for _ in range(units):
            rand_idx = np.random.randint(1, self.ndim)
            spin = np.dot(self.weights[rand_idx,:], state) # Activation function

            # state[rand_idx] = self.step(spin)

            prob = self.sigmoid(spin)
            state[rand_idx] = self.bernoulli(prob)

        return state

    def step(self, y): # Step / Threshold activation function
        return 1 if y > 0 else -1

    def sigmoid(self, x): # Contiuous activation function
        return 1.0 / (1.0 + np.exp(-x))

    def bernoulli(self, prob):
        return 1 if np.random.uniform(0, 1) < prob else -1 # Bernoulli sampling

    def compute_energy(self, state): # As per original paper
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)
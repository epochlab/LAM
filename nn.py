#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))

    def train(self, data, threshold=0.1):
        for sample in data:
            memory = np.array([np.where(sample > threshold, 1, -1)]) # Binary dipole (+/-)
            delta_weights = memory.T * memory
            self.weights += delta_weights
            self.weights = (self.weights + self.weights.T) / 2 # Enforce symmetrical weights
        self.weights /= data.shape[0] # Normalise against dataset
        np.fill_diagonal(self.weights, 0) # Set the diagonal to zero - ensure nodes to influence themselves

    def infer(self, state, units, T):
        for _ in range(units):
            rand_idx = np.random.randint(1, self.ndim)
            spin = np.dot(self.weights[rand_idx,:], state) # Activation function

            # state[rand_idx] = self.step(spin)

            prob = self.sigmoid(spin / T)
            state[rand_idx] = self.bernoulli(prob)

        return state

    def step(self, y): # Step / Threshold activation function
        return 1 if y > 0 else -1

    def sigmoid(self, x): # Continuous activation function
        return 1.0 / (1.0 + np.exp(-x))

    def bernoulli(self, prob): # Bernoulli distribution
        return 1 if np.random.uniform(0, 1) < prob else -1

    def compute_energy(self, state): # As per original paper
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)

class elman():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Init weights
        self. w_ih = np.random.randn(input_size, hidden_size)
        self. w_hh = np.random.randn(hidden_size, hidden_size)
        self. w_ho = np.random.randn(hidden_size, output_size)

        # Init bias
        self.b_h = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
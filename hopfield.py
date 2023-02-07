#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))

    def train(self, data, threshold):
        for _, sample in enumerate(data):
            memory = np.array([np.where(sample > threshold, 1, -1)])
            self.weights += (memory.T * memory)
            np.fill_diagonal(self.weights, 0)
        self.weights /= data.shape[0]

    def infer(self, state, units):
        for _ in range(units):
            rand_idx = np.random.randint(1, self.ndim)
            spin = np.dot(self.weights[rand_idx,:], state)

            if spin > 0:
                state[rand_idx] = 1
            else:
                state[rand_idx] = -1

        return state

    def compute_energy(self, state):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)
#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))

    def train(self, sample):
        xbin = np.where(sample > 0.1, 1, -1)
        memory = np.array([xbin])
        self.weights += (memory.T * memory)

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
        return -np.dot(np.dot(self.weights, state), state.T)
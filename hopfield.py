#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim, momentum=0.9):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))
        self.momentum = momentum
        self.v = 0

    # def train(self, sample):
    #     xbin = np.where(sample > 0.1, 1, -1)
    #     memory = np.array([xbin])
    #     self.weights += (memory.T * memory)

    def train(self, data, threshold):
        for _, sample in enumerate(data):
            memory = np.array([np.where(sample > threshold, 1, -1)])
            delta_weights = memory.T * memory
            self.v = self.momentum * self.v + delta_weights
            self.weights += self.v
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

    # def infer(self, state, units):
    #     for _ in range(units):
    #         spin = np.dot(self.weights, state)
    #         state = np.where(spin > 0, 1, -1)
    #     return state

    def compute_energy(self, state):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)

    def L1_loss(self, state, regularization=0.1):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T) + regularization * np.sum(np.abs(self.weights))

    def L2_loss(self, state, regularization=0.1):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T) + (regularization / 2) * np.sum(self.weights ** 2)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
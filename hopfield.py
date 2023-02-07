#!/usr/bin/env python3

import numpy as np

class hopfield():
    def __init__(self, ndim, momentum=0.9):
        self.ndim = ndim
        self.weights = np.zeros((self.ndim, self.ndim))
        self.momentum = momentum
        self.v = 0

    # Train single sample
    # def train(self, sample):
    #     xbin = np.where(sample > 0.1, 1, -1)
    #     memory = np.array([xbin])
    #     self.weights += (memory.T * memory)

    def train(self, data, threshold):
        for _, sample in enumerate(data):
            memory = np.array([np.where(sample > threshold, 1, -1)]) # Binary dipole
            delta_weights = memory.T * memory
            self.v = self.momentum * self.v + delta_weights
            self.weights += self.v
            np.fill_diagonal(self.weights, 0) # Symmetrical weights
        self.weights /= data.shape[0] # Normalise against dataset

    def infer(self, state, units):
        for _ in range(units):
            rand_idx = np.random.randint(1, self.ndim)
            spin = np.dot(self.weights[rand_idx,:], state)

            if spin > 0:
                state[rand_idx] = 1
            else:
                state[rand_idx] = -1

        return state

    # Infer on changed spin states
    # def infer(self, state, units):
    #     for _ in range(units):
    #         spin = np.dot(self.weights, state)
    #         state = np.where(spin > 0, 1, -1)
    #     return state

    def compute_energy(self, state): # Original Paper
        return -0.5 * np.dot(np.dot(self.weights, state), state.T)

    def L1_loss(self, state, regularization=0.1):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T) + regularization * np.sum(np.abs(self.weights))

    def L2_loss(self, state, regularization=0.1):
        return -0.5 * np.dot(np.dot(self.weights, state), state.T) + (regularization / 2) * np.sum(self.weights ** 2)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

# Batch training: Currently, the train method updates the weights for each sample in the data separately. Instead, it can be improved to update the weights based on the average of all samples, which is more efficient.
# Early Stopping: Add a condition to break out of the loop if the state does not change anymore during inference. - Val Loss
# More flexible thresholding: The current thresholding used in the train method is a simple threshold on the input data. It can be improved to a more flexible thresholding function, such as a sigmoid function, to capture more information in the input data. (Sigmoid)
# Initial state: Implement a method to set the initial state to the closest stored pattern in the network during inference.
# Convergence criteria: Implement a convergence criteria to determine when to stop inference.
# More complex training algorithms: Consider using more advanced training algorithms such as gradient descent or stochastic gradient descent instead of simple Hebbian learning.
# Parallel processing: The current implementation runs the inference for each unit sequentially. Parallel processing can be used to speed up the inference process by updating multiple units simultaneously.
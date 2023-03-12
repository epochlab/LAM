# memory = np.array([np.where(sample > threshold, 1, -1)]) # Binary dipole (+/-)

# self.degree = np.diag(1/np.sqrt(np.sum(self.weights, axis=1))) # Inverse square root of the (diagonal) degree matrix   

# def sigmoid(self, x): # Continuous activation function
#     return 1.0 / (1.0 + np.exp(-x))

# def bernoulli(self, prob): # Bernoulli distribution
#     return 1 if np.random.uniform(0, 1) < prob else -1

# prob = self.sigmoid(spin / T) # Activation function
# state[idx] = self.bernoulli(prob) # Non-deterministic

# delta_weights = (memory-alpha) * (memory.T-alpha) # Modify with alpha to store biased patterns

# spin = np.dot(self.weights[idx,:], state) + self.laplacian[idx, idx] * state[idx] # Activation function

# self.degree = np.diag(np.sum(self.weights, axis=1)) # Degree of each vertex
# self.adjacency[np.abs(self.weights) > 0] = 1 # Normalized connectivity of network
# np.fill_diagonal(self.adjacency, 0) # Set diagonal to 0
# self.laplacian = self.degree - self.adjacency # Construct laplacian matrix (simple)

# def step(self, x): # Heaviside step function
#     return np.where(x > 0, 1, -1)

# data = np.random.choice([-1, 1], size=n_dim*n_memory, p=[p, 1-p]) # Binary dipole (+/-) input with sparsity

# w = int((n_dim * (n_dim - 1)) / 2)
# print(n_dim, \"neurons\")
# print(w, \"weighted connections\")
# print(f\"{n_dim * 0.138:.2f} point attactors\")
# print(f\"{0.24 * w:.2f} bit capacity\")
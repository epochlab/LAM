import torch
from tqdm import tqdm

class LAM():
    """
    LAPLACIAN ASSOCIATIVE MEMORY (LAM)
    """
    def __init__(self, N, P, prob, H, gamma, norm_mode, start_node, features=None, temp=None):
        self.N = N                      # Neurons (n)
        self.P = P                      # Random memory patterns (n)
        self.prob = prob                # Sparsity (Activation Probability)
        self.H = H                      # Hetero-associative weights
        self.gamma = gamma              # Inhibition ratio (Regularisation)
        self.norm_mode = norm_mode      # Normalisation

        self.V = self.prob * (1 - self.prob)
        self.NV = self.N * self.V
        
        self.start_node = start_node
        self.features = features
        self.temp = temp

        # Binary state vectors
        self.xi = torch.rand(self.N, self.P) < self.prob
        self.xi = self.xi.float()

        if self.temp!=None:
            state = self._set_state(self.features)
            self.xi[:, self.start_node] = state
            print("Using feature-based initial condition")
            print("Sparsity:", torch.sum(state/torch.numel(state)))

        self.xi_mean = torch.sum(self.xi, dim=1, keepdim=True) / self.P # Mean activation of each neuron across all inputs
        self.xi_bias = self.xi - self.xi_mean

        # NORMALIZATION
        if self.norm_mode == "sym": # SYMMETRIC WEIGHTS
            Dnorm = torch.diag(np.sum(self.H, axis=1)**-0.5).float()
            self.H = Dnorm @ self.H @ Dnorm
            self.Wauto = (self.xi_bias @ self.xi_bias.T) / self.NV
            self.Whetero = (self.xi_bias @ self.H @ self.xi_bias.T) / self.NV
            self.WG = self.gamma / self.N

        elif self.norm_mode == "asym": # ASYMMETRIC WEIGHTS
            Dnorm = torch.diag(torch.sum(self.H, dim=1)**-1).float() # Degree matrix (D**-1*A)
            self.H = Dnorm @ self.H # Hetero-associative weights
            self.Wauto = (self.xi @ self.xi.T) / self.NV # Local inhibition
            self.Whetero = (self.xi @ self.H @ self.xi.T) / self.NV # Excitatory
            self.WG = self.P * self.xi_mean @ self.xi_mean.T / self.NV + self.gamma / self.N # Global inhibition
        else:
            print("Error: Normalization mode 'sym' or 'asym' was not specified.")
            exit()

    def _step(self, z): # Heaviside step function
        return 0.5 * torch.sign(z) + 0.5
    
    def _set_weight(self, a): # Decompose weights
        self.W = a * self.Wauto + self.Whetero - (a+1) * self.WG

    def _set_state(self, features):
        I = torch.zeros_like(self.xi) # Malloc
        for node in range(self.xi.shape[1]):
            state = (self.xi[:, node].clone() * 2) - 1 # State of each node and re-map between -1 and 1
            I[:,node] = state * features.flatten()[node]

        Inorm = torch.sum(I, axis=1) * 1/self.xi.shape[1]
        act = self._boltzmann_prob(Inorm, self.temp)
        return act

    def _boltzmann_prob(self, x, temp):
        p = 1 / (1.0 + torch.exp(-x/temp))
        y = (torch.rand(*x.shape) < p) * 1.0
        return y

    def _kronecker_delta(self, i, j):
        return 1 if i==j else 0

    def simulate_single(self, a, eta, simlen, energycheck=True):
        self._set_weight(a)  # Set weight based on alpha
        self.x = self.xi[:, self.start_node].clone()

        self.m_log = torch.zeros([simlen, self.P])
        self.n_log = torch.zeros([simlen, self.N])
        self.obj_log = torch.zeros([simlen])

        for t in tqdm(range(simlen)):
            self.r = self._step(self.W @ self.x)  # Use torch.matmul for matrix multiplication
            self.x += eta * (self.r - self.x)
            self.m = (self.xi_bias.T @ self.x) / self.NV  # Use torch.matmul for matrix multiplication
        
            self.m_log[t,:] = self.m
            self.n_log[t,:] = self.x

        if energycheck:
            self.obj_log[t] = -(self.x).t() @ self.W @ self.x / self.NV # Compute energy

        return self.m_log, self.n_log, self.obj_log
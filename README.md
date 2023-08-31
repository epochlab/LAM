# Laplacian Associative Memory (LAM)

**Project ID:**  Tl44vh6n

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

### Community structures of associative memory encoding
Abstract: *A novel formalism to integrate correlational structure into a extended hopfield network using graph-based hierarchical computation to generate communities (clusters) of associative links between cortical units, storing node assemblies (state/memory patterns) through auto-association and finds relationships between nodes using hetero-association.*

#### What is the neurological representation?
A neural map is a collection of receptive fields which respond to different values of common stimulus; where cortical units correspond selectively to the occurance of selective events in the world. Adjacent neurons respond preferentially to similar patterns of afferent input, communicating across the cortical sheet.

- Key Features
    - Correlation & clustering
    - Attractor states
    - Binary state vectors
    - Afferent stimulus
    - Boltzmann distribution
    - Pattern overlap
- Theory
    - Neurobiological maps
    - Adjaceny degree matrices
    - Hetero-associative weights
    - Hopfield networks
    - Information efficiency

#### Nomenclature
Parameter | Description
--- | ---
`cortex` | A cortical sheet of neurons
`neuron` | A cell/unit of information
`graph` | A structure of nodes
`node` | Binary state vector
`network` | Laplacian associative hopfield network
`unit` | An adaptive parameter assigned to a node
`correlation` | An adjacency matrix for the input feature space
`weights` | A decomposition of the correlation matrix into excitatory, local & global inhibitory.
`n` | Neuron or pixel
`P` | Total number of neurons (n-by-n)
`W` | Correlation matrix (P-by-P)
`N` | Number of nodes (per neuron)
`start_node` | Initial cortical unit
`alpha` | Ratio between hetero- and auto- association
`prob` | Sparsity or number of active neurons
`temp` | Boltzmann distribution
`H` | Hetero-associative weights (Asymmetric normalisation)
`m` | Pattern overlap
`xi` | Binary state vector (one for each neuron)
`x` | Network response
`I` | Initial condition (Afferent initial state)
`sigmaX` | Spatial component (Euclidean)
`sigmaA` | Orientation preference (Value)
`e` | Energy

#### References
[Multiscale representations of community structures in attractor neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf) (2021)<br>

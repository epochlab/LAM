# Laplacian Associative Memory (LAM)

**Project ID:**  Tl44vh6a

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### Community structures of associative memory encoding
Abstract: *An extended network model for graph-based hierarchical computation, generating communities (clusters) of associative links between memory items, with scale regulated by the heterogenous modulation of inhibitory circuits.*

### What is the neurological representation?
A neural map is a collection of receptive fields which respond to different values of common stimulus; where cortical units correspond selectively to the occurance of selective events in the world. Adjacent neurons respond preferentially to similar patterns of afferent input, communicating across the cortical sheet.

- Key features
    - Clustering
    - Attractor states
    - Binary state vector
    - Afferent stimulus
    - Boltzmann distribution
    - Contour completion???
- Theory
    - Neurobiological maps
    - Hopfield networks
    - Information efficiency

### Glossary
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

### References:
[Neural networks and physical systems with emergent collective computational abilities](https://www.researchgate.net/publication/16246447_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities) (1982)<br>
[Learning internal representations by error propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) (1985)<br>
[Information storage in neural networks with low levels of activity](https://sci-hub.ru/10.1103/PhysRevA.35.2293) (1987)<br>
[The enhanced storage capacity in neural networks with low activity level](https://sci-hub.ru/10.1209/0295-5075/6/2/002) (1988)<br>
[Finding structure in time](http://psych.colorado.edu/~kimlab/Elman1990.pdf) (1990)<br>
[Extended temporal association memory by modulations of inhibitory circuits](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.123.078101) (2019)<br>
[Multiscale representations of community structures in attractor neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf) (2021)<br>
[Laplacian associative memory](https://github.com/TatsuyaHaga/laplacian_associative_memory_codes/tree/v1.0.1) (2021)

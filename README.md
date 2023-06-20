# Laplacian Associative Memory (LAM)

**Project ID:**  Tl44vh6a

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### Community structures of associative memory encoding
Abstract: *An extended network model for graph-based hierarchical computation, generating communities (clusters) of associative links between memory items, with scale regulated by the heterogenous modulation of inhibitory circuits.*

### Concept
- Chunking and clustering of items
    - Increasing the number of items retained in the limited capacity of working memory
- Segmentation of images & words
    - Learning and comprehension of scenes/language
- Temporal abstraction of repeated sequences
    - Accelerate reinforcement learning

### What is the neurological representation?
- Having experienced a seqence of events, the brain, learns the temporal associations between the successive events and captures the structure of a state-transition graph
- Event segmentation performed by humans reflects community structures (clusters)
- Characteristics of place cells and entorhinal grid cells
- Asymmetrical links generate chunked sequential activities in the hippocampus
- How strong is the relationship between hebbian learning, state attractors and sequential segmentation with the hippocampus?

### Blurb (WIP)
A binary recurrent network into which a embedded topological map structure can reveal patchy connectivity. By setting the state of hopfield network using a point of activation (start_node) and presenting stimulation with location and orientation components, the network identifies overlapping patterns and/or community clusters within the source data.

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
`node` | A cell/unit of information
`neuron` | An adaptive parameter assigned to a node
`graph` | A structure of pair-wise nodes
`weights` | A matrix which transforms the relationship between neurons
`n` | Node or pixel
`P` | Total number of nodes (n-by-n)
`W` | Adjacency matrix (P-by-P)
`N` | Number of neurons (per node)
`start_node` | Initial cortical unit
`alpha` | Ratio between hetero- and auto- association
`prob` | Sparsity or number of active neurons
`temp` | Boltzmann distribution
`H` | Hetero-associative weights (Asymmetric normalisation)
`m` | Pattern overlap
`xi` | Binary state vectors (one for each node)
`x` | Network response
`I` | Initial condition (Binary afferent state vector)
`sigmaX` | Spatial component (Euclidean)
`sigmaA` | Orientation preference (Value diff.)
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

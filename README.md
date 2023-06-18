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
Topic | Description
--- | ---
[State-transition graphs](https://en.wikipedia.org/wiki/State_diagram) | A set of states, events or inputs that can trigger transitions between states and describe how the system operates from one state to another in response to events or inputs.
[Community structure](https://en.wikipedia.org/wiki/Community_structure) | Patterns similar to each other are grouped together into the same community, while patterns that are dissimilar from each other are grouped into different communities.
[Auto-association weights](https://en.wikipedia.org/wiki/Autoassociative_memory) |  Weights that connect a neuron to itself; used to complete or rectrieve/recover missing or corrupted parts of the pattern (Remove interference).
[Hetero-association weights](https://en.wikipedia.org/wiki/Autoassociative_memory) | Weights that connect two different neurons; when a new pattern is presented to the network, hetero-associative weights recall the stored pattern most similar to the input pattern (Banana > Monkey).
[Degree matrix](https://en.wikipedia.org/wiki/Degree_matrix) | A diagonal matrix which contains information about the degree of each vertex, the number of edges attached to each vertex.
[Adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) | A square matrix that represents a graph. The rows and columns of the matrix correspond to the vertices of the graph and sample both the spatial component and orientation preference of the data structures.
[Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) | A square matrix constructed from the adjacency matrix; which encodes the relationships between the vertices. The Laplacian matrix is a useful tool to study various properties of graphs, such as connectivity and community structure.
[Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance) | The length of a line segment between the two points.
[Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) | A characteristic vector of a linear transformation, a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it.
[Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) | A form of recurrent artificial neural network serving a as content-addressable ("associative") memory system with binary threshold nodes.
[Afferent stimuli](https://en.wikipedia.org/wiki/Afferent_nerve_fiber) | Afferent neurons bring stimuli to the brain, where the signal is integrated and processed.
[Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) | A probability distribution or probability measure that gives the probability that a system will be in a certain state as a function of that state's energy and the temperature of the system.

--------------------------------------------------------------------

Parameter | Description
--- | ---
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

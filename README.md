# PSY6431

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
A binary recurrent network into which a embedded topological map structure can reveal patchy connectivity. By setting the state of hopfield network using a point of activation (start_node) and presenting stimulation with location and orientation components, settles into a overlappig pattern identifying community clusters within the source data.

### SOM Notes
- Phi disabled?

### Sigma Notes
- Increasing sigmaX value, increases the weight given to the opposing SigmaY component
- LAM using a Laplacian distribution....

### To-Do
- Key features
    - Clustering
    - Attractor states
    - Contour Completion
- Benchmark
    - SSIM
    - Dimensional clustering
- Theory
    - Relationship to Hippocampus
- Proposal
    - Clustering Unsupervised SOM
    - Relationship to place cells and entorhinal grid cells (Start Node)
    - Denoising
    - Pattern Recognition
    - Memory Capacity
    - Effect of Network Size
    - Rebustness to noise

### Q&A
??? | Notes
------- | -------
[State-transition graphs](https://en.wikipedia.org/wiki/State_diagram) | A set of states, events or inputs that can trigger transitions between states and describe how the system operates from one state to another in response to events or inputs.
[Community structure](https://en.wikipedia.org/wiki/Community_structure) | Patterns similar to each other are grouped together into the same community, while patterns that are dissimilar from each other are grouped into different communities.
[Auto-association weights](https://en.wikipedia.org/wiki/Autoassociative_memory) |  Weights that connect a neuron to itself; used to complete or recover the missing or corrupted parts of the pattern.
[Hetero-association weights](https://en.wikipedia.org/wiki/Autoassociative_memory) | Weights that connect two different neurons; when a new pattern is presented to the network, hetero-associative weights retrieve the stored pattern most similar to the input pattern.
[Degree matrix](https://en.wikipedia.org/wiki/Degree_matrix) | A diagonal matrix which contains information about the degree of each vertex, the number of edges attached to each vertex.
[Adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) | A square matrix that represents a graph. The rows and columns of the matrix correspond to the vertices of the graph and sample spatial, orientation and luminance data structures.
[Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) | A square matrix constructed from the adjacency matrix; which encodes the relationships between the vertices. The Laplacian matrix is a useful tool to study various properties of graphs, such as connectivity and community structure.
[Eigenvalues](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) | A characteristic vector of a linear transformation, a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it.
[Negative associated weights](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7116367/) | Negative associated weights are weights in a neural network that represent inhibitory connections between neurons; where neurons compete with each other to become active.
[Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta) | A mathematical function that takes two indices [i] and [j] and returns 1 if they are equal and 0 otherwise.
[Entorhinal grid cells](https://en.wikipedia.org/wiki/Grid_cell) | Spatial navigation and memory; each grid cell has a unique firing pattern, with multiple peaks of activity that form a hexagonal grid pattern.
[Cell assemblies](http://www.scholarpedia.org/article/Cell_assemblies) | A network of neurons being repeatedly activated causing excitatory synaptic connections among its members are being strengthened.
Sub-goal finding in RL | Identifying intermediate goals or sub-goals that can help an agent achieve its ultimate objective more efficiently.
[Kaiming initialization](https://arxiv.org/pdf/1502.01852v1.pdf) | An initialization method accounting for the non-linearity of activation functions, avoid reducing or magnifying the magnitudes of input signals exponentially.
[One-shot learning](https://en.wikipedia.org/wiki/One-shot_learning) | A network which aims to classify objects from one, or only a few, examples
[Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) |  The minimum number of substitutions required to change one string into the other.
[Undirected graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) | A set of objects (called vertices or nodes) that are connected together, where all the edges are bidirectional.
Pattern overlap |
Reverse querying |

### References:
[Neural networks and physical systems with emergent collective computational abilities](https://www.researchgate.net/publication/16246447_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities) (1982)<br>
[Learning internal representations by error propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) (1985)<br>
[Information storage in neural networks with low levels of activity](https://sci-hub.ru/10.1103/PhysRevA.35.2293) (1987)<br>
[The enhanced storage capacity in neural networks with low activity level](https://sci-hub.ru/10.1209/0295-5075/6/2/002) (1988)<br>
[Finding structure in time](http://psych.colorado.edu/~kimlab/Elman1990.pdf) (1990)<br>
[A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003)<br>
[Extended temporal association memory by modulations of inhibitory circuits](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.123.078101) (2019)<br>
[Multiscale representations of community structures in attractor neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf) (2021)<br>
[Laplacian associative memory](https://github.com/TatsuyaHaga/laplacian_associative_memory_codes/tree/v1.0.1) (2021)

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

### Research questions
- How strong is the relationship between hebbian learning, state attractors and sequential segmentation with the hippocampus?

### To-Do
- Datasets
    - Original source (Repeatition problem)
    - Karate club network
    - The structure of compartmentalised rooms
    - Images, Language & RL?
- Networks
    - LAM
    - Boltzmann Machine
    - GRU, LSTM & Transformer
- Key features
    - Clustering
    - Attractor states
    - Pattern completion
- LAM
    - Arbitary symmetrical graphs
    - Negative associated weights
- Benchmark
    - SSIM
    - Dimensional clustering

### Q&A
? | Notes
------- | -------
State-transition graphs | A set of states, a set of events or inputs that can trigger transitions between states, and a set of transitions that describe how the system operates from one state to another in response to events or inputs.
Community structures (Clusters) | Patterns that are similar to each other are grouped together into the same community, while patterns that are dissimilar from each other are grouped into different communities.
Auto-association weights |  Weights that connect a neuron to itself; used to complete or recover the missing or corrupted parts of the pattern.
Hetero-association weights | weights that connect two different neurons; when a new pattern is presented to the network, hetero-associative weights retrieve the stored pattern most similar to the input pattern.
Local and global inhibitory circuits |
Pattern overlap |
Cell assembilies |
Abstract excitatory and inhibitatory activity |
Assembly specific inhibition | Negative associated weights are weights in a neural network that represent inhibitory connections between neurons; where neurons compete with each other to become active.
Entorhinal grid cells | Spatial navigation and memory; each grid cell has a unique firing pattern, with multiple peaks of activity that form a hexagonal grid pattern.
Relationship between LAM and Laplacian |
Adjacency graph matrices | A square matrix that represents a graph. The rows and columns of the matrix correspond to the vertices of the graph, and the entries of the matrix indicate whether there is an edge between two vertices.
Laplacian eigenvectors | A square matrix constructed from the adjacency matrix; which encodes the relationships between the vertices. The Laplacian matrix is a useful tool to study various properties of graphs, such as connectivity and community structure.
Sub-goal finding in RL | Identifying intermediate goals or sub-goals that can help an agent achieve its ultimate objective more efficiently.
Kronecker delta | A mathematical function that takes two indices [i] and [j] and returns 1 if they are equal and 0 otherwise.

### References:

[Neural networks and physical systems with emergent collective computational abilities](https://www.researchgate.net/publication/16246447_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities) (1982)<br>
[Learning internal representations by error propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) (1985)<br>
[Finding structure in time](http://psych.colorado.edu/~kimlab/Elman1990.pdf) (1990)<br>
[A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003)<br>
[Multiscale representations of community structures in attractor neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf) (2021)
# PSY6431

**Project ID:**  Tl44vh6a

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### Community structure of associative memory encoding
Abstract: *An extended network model for graph-based hierarchical computation, generating communities (clusters) of associative links between memory items, with scale regulated by the heterogenous modulation of inhibitory circuits.*

### :cloud: Concept

- Chunking / clustering of items
    - Increasing the number of items retained in the limited capacity of working memory
- Segmentation of images & words
    - Learning and comprehension of scenes/language
- Temporal abstraction of repeated sequences
    - Accelerate reinforcement learning

### :brain: What is the neurological representation?
- Having experienced a seqence of events, the brain, learns the temporal associations between the successive events and captures the structure of a state-transition graph
- Event segmentation performed by humans reflects community structures (clusters)
- Characteristics of place cells and entorhinal grid cells
- Asymmetrical links generate chunked sequential activities in the hippocampus

### Research questions?
- How strong is the relationship between hebbian learning, state attractors and sequential segmentation with the hippocampus?

### Build
- Datasets
    - Original source
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
    - Arbitary symmetrical graphs; generalise the one-dimensional sequential structure of temporal associations of the conventional hopfield model
    - Negative associated weights; assembly specific inhibition
- Benchmark
    - SSIM
    - Dimensional clustering

### Q&A
? | Notes
------- | -------
State-transition graphs |
Community structures (Clusters) |
Association memory weights (Auto & Hetero) |
Local and global inhibitory circuits |
Pattern overlap |
Cell assembilies |
Abstract excitatory and inhibitatory activity |
Assembly specific inhibition |
Entorhinal grid cells |
Relationship between LAM and Laplacian |
Laplacian eigenvectors |
Adjacency graph matrices |
Sub-goal finding in RL |
Kronecker delta |

### Papers:

[Neural networks and physical systems with emergent collective computational abilities](https://www.researchgate.net/publication/16246447_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities) (1982)<br>
[Learning internal representations by error propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) (1985)<br>
[Finding structure in time](http://psych.colorado.edu/~kimlab/Elman1990.pdf) (1990)<br>
[A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003)<br>
[Multiscale representations of community structures in attractor neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf) (2021)
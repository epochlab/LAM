# LAM

**Project ID:**  Tl44vh6a

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### Laplacian Associative Memory (LAM)
Abstract: *An extended attractor network model for graph-based hierarchical computation, generating multiscale representations for communities (clusters) of associative links between memory items, and the scale is regulated by the heterogenous modulation of inhibitory circuits.*

[Hopfield, 1982](https://www.researchgate.net/publication/16246447_Neural_Networks_and_Physical_Systems_with_Emergent_Collective_Computational_Abilities)<br>
[LAM, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf)

--------------------------------------------------------------------

## WORK-IN-PROGRESS

### Concept
Graph-based segmentation and abstraction.

Attractor dynamics of recurrent neural circuits offer a biologically plausiable for hierarchical segmentation.
Relationship between attrator networks and the graph-theoretic processing of knowledge structures.

- Chunking of items
    - Increasing the number of items retained in the limited capacity of working memory
- Segmentation of words
    - Learning and comprehension of language
- Temporal abstraction of repeated sequences
    - Accelerate reinforcement learning

### What is the neurological representation?
- Having experienced a seqence of events, the brain, learns the temporal associations between the successive events and captures the structure of a state-transition graph
- Event segmentation performed by humans reflects community structures (clusters)
- Characteristics of place cells and entorhinal grid cells
- Asymmetrical links generate chunked sequential activities in the hippocampus

### Research questions?
- How strong is the relationship between hebbian learning, state attractors and sequential segmentation with the hippocampus?
- How can this technique be improved?
- Alternate abstraction
- CG application (Segmentation / ...)

### Build
- Datasets
    - Original graph
    - Karate club network
    - The structure of compartmentalised rooms
    - Image-based, RL & robotic? (NEW)
- Hopfield RNN
    - Hebbian learning (spike-timing-dependant plasticity)
    - Pattern completion
    - Attractor states
- Key features
    - Arbitary symmetrical graphs; generalise the one-dimensional sequential structure of temporal associations of the conventional hopfield model
    - Negative associated weights; assembly specific inhibition
- LAM network with hetero-associative weights
    - Adjacency graph matrices 
- Laplacian eigenvectors 
    - Unsupervised learning?

--------------------------------------------------------------------

? | Notes
------- | -------
State-transition graphs | 
Community structures (Clusters) | 
Association memory networks (Auto & Hetero) | 
Sub-goal finding in RL | 
Local and global inhibitory circuits | 
Kronecker delta | 
Relationship between LAM and Laplacian | 
Pattern overlap | 
Cell assembilies | 
Abstract excitatory and inhibitatory activity | 
Entorhinal grid cells |
Assembly specific inhibition |
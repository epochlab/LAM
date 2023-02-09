# LAM

**Project ID:**  Tl44vh6a

<p align="center">
  <img src="https://github.com/epochlab/LAM/blob/main/sample.png">
</p>

--------------------------------------------------------------------

#### Laplacian Associative Memory (LAM)
Abstract: *An extended attractor network model for graph-based hierarchical computation, generating multiscale representations for communities (clusters) of associative links between memory items, and the scale is regulated by the heterogenous modulation of inhibitory circuits.*

[LAM, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8412329/pdf/pcbi.1009296.pdf)
[Hopfield, 1982](https://sci-hub.ru/10.1073/pnas.79.8.2554)

--------------------------------------------------------------------

#### WORK-IN-PROGRESS

++ Concept<br>
Graph-based segmentation and abstraction.

Attractor dynamics of recurrent neural circuits offer a biologically plausiable for hierarchical segmentation.
Relationship between attrator networks and the graph-theoretic processing of knowledge structures.

- Chunking of items; increasing the number of items retained in the limited capacity of working memory
- Segmentation of words; learning and comprehension of language
- Temporal abstraction of repeated sequences; accelerates reinforcement learning

++ What is the neurological representation?
- Having experienced a seqence of events, the brain, learns the temporal associations between the successive events and captures the structure of a state-transition graph
- Event segmentation performed by humans reflects community structures (clusters)
- Characteristics of place cells and entorhinal grid cells
- Asymmetrical links generate chunked sequential activities in the hippocampus

++ Build
- Datasets [3]: Repeat graph, karate club network and the structure compartmentalised rooms
- Hopfield RNN - Hebbian learning (spike-timing-dependant plasticity), pattern completion and attractor states
- LAM network with hetero-associative weights - Adjacency graph matrices 
- Laplacian eigenvectors - unsupervised learning?
- Random binary patern to each node

Feature | Notes
------- | -------
Arbitary symmetrical graphs | Generalise the one-dimensional sequential structure of temporal associations of the conventional model - Hopfield?
Negative associated weights | Assembly specific inhibition

++ Unknown concepts and observations<br>
Community structures
Auto-associative memory networks
Subgoal finding in RL
Local and global inhibitory circuits
Kronecker delta
Relationship between LAM and Laplacian

++ Questions?<br>
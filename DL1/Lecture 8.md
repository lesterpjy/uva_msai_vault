## Graph Neural Networks


Graphs are more common than one might think.
Many structures can be represented as graphs.
Graphs can be visualized as geometrical shapes, with nodes represented as points and edges as lines connecting those points. This visualization highlights the structural relationships within the graph.

### What are graph networks?

![[gnn_basic.png | 400]]

### Graph networks
![[gnn_uses.png | 450]]
**Graph networks process graphs** by transforming the feature vectors of each node over multiple layers. They can be used for various tasks, including:

- **Graph classification**: Assigning a label to an entire graph, similar to image classification. The sources provide an example of predicting whether a molecule is toxic. For graph-level tasks, the output node embeddings are typically combined (e.g., by averaging) and passed through a linear transformation or neural network to produce a fixed-size vector for regression or classification.
- **Node classification**: Predicting a label for each individual node, analogous to image segmentation. The sources provide an example of classifying points in a point cloud as belonging to either the wings or fuselage of an airplane. For node-level tasks, the loss function is defined independently for each node.
- **Link prediction**: Predicting the presence or absence of an edge between two nodes. The sources provide an example of predicting whether two people in a social network know and like each other. Link prediction is often a binary classification task. The probability of an edge existing between two nodes can be computed, for example, by taking the dot product of the node embeddings and passing the result through a sigmoid function.
- **Graph generation**: Creating new graphs, similar to image or text generation.

### Graph classification

![[graph_class.png]]

### Slide: 2) Node classification

![[node_class.png]]

### Slide: 3) Link prediction

![[link_pred.png]]

### Slide: 4) Graph generation

![[graph_gen.png]]

### Generated molecule, Example molecule

This slide shows examples of molecules represented as graphs. **Graphs can be used to represent molecules, with atoms as nodes and chemical bonds as edges**. The sources describe how graph neural networks can be used to classify molecules as toxic or harmless based on their graph representation. The node features in this case are one-hot vectors indicating the presence of specific elements from the periodic table.

### Graphs can be dynamic

**Graphs are often assumed to have fixed structures**, but many graphs change over time. These **changes can be gradual or sudden**.

### Regular structures and graphs

**Regular structures can be considered a subset of graphs**. For example, **images are grid graphs**. The table below highlights the differences between regular structures, such as images, and graphs:

|Regular Structures|Graphs|
|:--|:--|
|Convolution + pooling|Message passing + coarsening|
|Local neighborhood: fixed window|Local neighborhood: 1-hop|
|Constant number of neighbors|Different number of neighbors|
|With fixed ordering|**No ordering of neighbors**|
|Translation equivariance|**Local permutation equivariance**|

**Convolutional neural networks (CNNs)** process images by applying convolutional filters to extract features from local neighborhoods, which have a fixed window size and ordering of neighbors. **Pooling operations** then downsample the feature maps, reducing their spatial resolution. In contrast, **graph neural networks (GNNs)** process graphs by aggregating information from neighboring nodes, which can have varying numbers and no inherent ordering. **Message passing** refers to the process of exchanging information between connected nodes. **Coarsening** operations in GNNs can also be used to reduce the size of the graph by grouping nodes together. CNNs are designed to be **translation equivariant**, meaning that shifting an object in an image does not change its classification. GNNs exhibit **local permutation equivariance**, meaning that changing the order of nodes in a local neighborhood does not affect the output for those nodes.

### Definition of a graph

![[graph_definition.png]]

Directed graphs
In **directed graphs**, edges have a specific direction. A directed graph is defined by a set of vertices $V = {1, ..., n}$ (also called "nodes") and a set of directed edges $E = \{(i, j) : i, j \in V\}\subseteq V \times V$.

Undirected graphs
In **undirected graphs**, edges have no direction. An undirected graph is defined by a set of vertices $V = {1, ..., n}$ and a set of undirected edges $E = \{\{i, j\} : i, j \in V\}\subseteq V \times V$.

Graph neighborhood
The **neighborhood** of a node $i$ in a graph consists of all nodes directly connected to it, denoted as $N(i) = \{j : (i, j) \in E\}$. The **degree** of a node is the number of its neighbors: $d_i = |N(i)|$. The **diagonal matrix $D$ contains all node degrees** along its diagonal.

Attributes
- **Node features**: Each node can have associated features represented as a vector $\mathbf{x}_i \in \mathbb{R}^{d}$, where $d$ is the feature dimension. All node features are collected in a matrix $X = (\mathbf{x}_1, ..., \mathbf{x}_n)$. In the social network example, each individual might be characterized by a vector representing their interests.
- **Edge features**: Each edge $(i, j)$ can have associated features represented as a vector $\mathbf{e}_{ij} \in \mathbb{R}^{d'}$. In the road network example, each edge might be characterized by its length, number of lanes, frequency of accidents, and speed limit.

Adjacency matrix
The **adjacency matrix** $A$ is an $n \times n$ matrix that represents the connections between nodes in a graph with $n$ nodes. The element $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$, and $A_{ij} = 0$ otherwise. The element $(A^z)_{ij}$ represents the number of paths of length $z$from node $i$ to node $j$.
![[adj_matrix.png]]

**The adjacency matrix for undirected graphs is symmetric**, meaning that $A_{ij} = A_{ji}$ for all $i$ and $j$.

Weighted adjacency matrix
If the edges in a graph have weights, then the **adjacency matrix is also weighted**, with the element $A_{ij}$ representing the weight of the edge between nodes $i$ and $j$.

### Graph layer as a convolution layer

One perspective is to view a graph layer as a convolution layer.
![[gnn_as_conv.png | 450]]

When extending convolutions to graphs, certain assumptions from image convolutions no longer hold. These include:
- **The number of neighbors is not fixed** for each node in a graph.
- There is **no inherent ordering between neighbors** in a graph.

Stacking graph convolution layers
![[gnn_conv_update_rule.png | 300]]
1. Information Aggregation:
    - Each graph convolution layer aggregates information from a node’s **direct neighbors**.
    - Stacking multiple layers allows nodes to gather information from **further-away neighbors** (i.e., increasing the receptive field).
2. Adding Non-Linearity:
    - After each layer, a **non-linear activation function** like ReLU is applied. This introduces complexity and allows the model to learn non-linear relationships.
3. Why Stack Layers?:
    - Stacking layers enables the model to capture richer and more global information about the graph structure.
    - For example, a 2-layer graph convolution allows a node to aggregate information from neighbors of its neighbors.

**Graph convolution layer in matrix form**
$$
f(\mathbf{X}, \mathbf{A}) := \sigma(\mathbf{D}^{-1/2}(\mathbf{A}+\mathbf{I})\mathbf{D}^{-1/2}\mathbf{X}\mathbf{W})
$$
Where:
- $A$: Adjacency matrix (defines connections between nodes).
- $I$: Identity matrix (adds self-loops to each node).
- $D$: Degree matrix (diagonal matrix where $D_{ii}$​ is the number of edges connected to node $i$).
- $X$: Input features (matrix where each row is a node's feature vector).
- $W$: Learnable weight matrix (applies transformation to features).
- $\sigma$: Activation function (e.g., ReLU or Sigmoid).

Key Components:
1. Adding Self-Loops $(A+I)$
    - The identity matrix $I$ adds self-loops to the adjacency matrix $A$.
    - This ensures each node considers its own features during the update process.
2. Normalization $\mathbf{D}^{-1/2}(\mathbf{A}+\mathbf{I})\mathbf{D}^{-1/2}$
    - Normalizing the adjacency matrix by the degree matrix $D$ helps stabilize the learning process.
    - The normalization divides each entry in the adjacency matrix by the square root of the degrees of the corresponding nodes.
    - Without normalization, nodes with many connections (high degree) could dominate the aggregation process.
3. Feature Transformation $(XW)$
    - The input feature matrix $X$ is linearly transformed using the learnable weight matrix $W$.
4. Activation $\sigma$
    - A non-linear activation (e.g., ReLU) is applied to the transformed features to introduce learning flexibility.

**Rewriting into 2 steps**
![[gnn_conv_2step.png]]

Left side $\tilde{A}X$
![[gnn_lhs.png | 550]]
![[gnn_lhs_eq.png | 200]]

Right side $XW$
![[gnn_rhs.png]]

Why not simply divide by the node degree?
- **Dividing by Degree $\mathbf{D}^{-1}\mathbf{A}$
    - Each node's contribution is scaled inversely by its degree.
    - Only scales outgoing contributions of source nodes.
    - However, this approach can still introduce instability during training due to imbalances in the propagation of features.
- Symmetric Normalization $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$
    - Balances the contributions more effectively by symmetrically normalizing both source and target nodes.
        - contribution of each node to its neighbors is scaled down by the square root of its degree.
        - features received by each node are also scaled down by the square root of its degree.
    - Ensures that the aggregation process is stable and consistent across the graph.

### Graph attention layers
**Graph attention layers** utilize attention mechanisms to aggregate information from neighboring nodes. The formula for a graph attention layer is:
$$
y_i = h(\sum_{j \in N(i)} a_{ij}z_j)
$$
where:

- $y_i$ is the output embedding for node $i$
- $h(\cdot)$ is an activation function
- $\sum_{j \in N(i)}$ represents the aggregation over neighbors of node $i$
- $a_{ij}$ is the attention weight between nodes $i$ and $j$
- $z_j$ is the transformed embedding of node $j$

**Self-attention** can be used to compute the attention weights:
$$
a_{ij} = \frac{exp(e_{ij})}{\sum_{k \in N(i)} exp(e_{ik})}
$$
where $e_{ij}$ represents the self-attention weights, calculated as:
$$
e_{ij} = LeakyReLU(u^{\top}[\mathbf{W}\mathbf{x}_i||  \mathbf{W}\mathbf{x}_j])
$$
where:
- $\mathbf{x}_i$ and $\mathbf{x}_j$ are the feature vectors of nodes $i$ and $j$, respectively
- $W$ - feature transformation matrix
- $\mathbf{u}$ - learnable weight vector
- || - Concatenation operator
**Advantages**:
- GAT allows the model to learn which neighbors are more relevant for feature aggregation.
- This is especially useful in graphs where neighbors have varying importance

**Four steps of GAT**
![[4steps_gat.png | 400]]
- **Input Feature Transformation**:
    - Transform the input features of each node using a weight matrix $W$.
    - This step is similar to the linear transformation in a traditional Graph Convolutional Network (GCN).
- **Attention Score Calculation**:
    - Compute attention coefficients $e_{ij}$​ for each edge $(i,j)$ using
$$
e_{ij} = LeakyReLU(u^{\top}[\mathbf{W}\mathbf{x}_i||  \mathbf{W}\mathbf{x}_j])
$$
    - attention score measures the importance of node $j$ to node $i$
- **Attention Normalization**:
    - Normalize the attention scores across all neighbors $\mathcal{N}(i)$ of node $i$:
$$
a_{ij} = \frac{exp(e_{ij})}{\sum_{k \in N(i)} exp(e_{ik})}
$$
- **Feature Aggregation**:
    - Aggregate the features of neighboring nodes using the normalized attention scores
$$
h'_{i} = \sigma\left(\sum_{j \in \mathcal{N}(i)} a_{ij} Wh_{j}\right)
$$

Connecting graphs, convolutions, and transformers
1. **Transformers and Complete Graphs**:
    - Transformers treat input tokens as nodes in a fully connected graph, where every token attends to every other token.
    - This is analogous to a graph where every node is connected to every other node.
2. **Attention-Based GNNs**:
    - Using attention mechanisms (as in GAT), GNNs can mimic the behavior of transformers.
    - By incorporating self-attention, GNNs can generalize better and operate on graph structures while leveraging the strengths of attention mechanisms.
3. **Recovering Transformers with Attention GCNs**:
    - When GNNs use attention across all nodes, they effectively become transformers.
    - This shows that transformers can be viewed as a special case of GNNs applied to complete graphs.

### Optimizing graph networks
1. Challenges
    - GNNs can be computationally expensive, especially for large graphs.
    - Over-smoothing: Stacking too many layers causes node features to become indistinguishable.
2. Optimization Techniques
    - Sparse Matrix Operations:
        - Leverage sparsity in graph structures to reduce computation time.
        - Focus only on non-zero entries in adjacency matrices.
    - Layer-Wise Aggregation:
        - Use fewer layers to limit over-smoothing while maintaining good performance.
    - Efficient Sampling:
        - Instead of processing the entire graph, sample a subset of nodes and their neighborhoods (e.g., GraphSAGE, Cluster-GCN).
3. Regularization
    - Apply techniques like dropout or weight decay to prevent overfitting.
    - Normalize intermediate feature representations to improve stability.

Pooling in graph networks
**Pooling** is an optional operation for graph networks, mainly used in graph classification tasks. Pooling groups nodes together, reducing the computational complexity, but requires updating the adjacency matrix accordingly.
### Applications of graph networks
Graph networks have numerous applications in various domains, including:

- Drug discovery
- Modeling the spread of diseases
- Social networks
- Recommendation
- Traffic forecasting
- Scene graph generation of visual data
- Point cloud classification
- Object interactions
- Text classification
- Particle physics


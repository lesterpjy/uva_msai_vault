This lecture discusses **non-Euclidean deep learning**, focusing on **hyperbolic deep learning in computer vision**.

- **Euclidean space is not always the best choice for representing data**:
    - For example, **hierarchical structures** like trees, ontologies, and taxonomies are ubiquitous in knowledge representation.
    - In hierarchies, the number of nodes grows exponentially with depth, but the volume of a ball in Euclidean space only grows polynomially with diameter, leading to **distortion when embedding hierarchies in Euclidean space**.
- **Hyperbolic space offers a better fit for hierarchical data**:
    - The volume of a ball in hyperbolic space grows exponentially with its radius, naturally accommodating the exponential growth of nodes in hierarchies.
    - Hyperbolic space can be considered a **continuous version of trees** with tree-like properties.
    - **Hyperbolic embeddings** have demonstrated superior performance in embedding hierarchies with minimal distortion.
- **Origins of hyperbolic geometry**: The slide deck does not mention the origins of hyperbolic geometry. Hyperbolic geometry is a non-Euclidean geometry that violates Euclid's parallel postulate. It is characterized by negative curvature and offers a distinct set of geometric properties compared to Euclidean geometry. The sources do not delve into the historical origins of hyperbolic geometry but highlight its properties relevant to deep learning.
- **Applications of Hyperbolic Deep Learning**:
    - **Hyperbolic image embeddings** improve few-shot learning by leveraging the natural hierarchical structure of images.
    - **Hyperbolic action embeddings** enhance action recognition in videos due to the inherent hierarchy in video data.
    - **Hyperbolic zero-shot learning** benefits from embedding semantic hierarchies in hyperbolic space.
    - **Hyperbolic embeddings for text** are suitable because of the hierarchical nature of text at multiple levels.
- **Key Concepts in Hyperbolic Deep Learning**:
    - **Poincaré Ball Model**: One of the most commonly used models of hyperbolic space, where points are represented within a unit ball.
        
    - **Klein Model**: Another model of hyperbolic space, where points are represented within a unit disk.
        
    - **Lorentz Model**: A model of hyperbolic space based on the Lorentzian inner product.
        
    - **Distance Functions**: Hyperbolic space has different distance functions compared to Euclidean space, such as the Poincaré distance and the Busemann distance.
        
        - The **Busemann distance** is particularly useful when dealing with points on the boundary of the Poincaré ball, which are at infinite distance from points inside the ball.
        - The Busemann distance for a point $z$ inside the Poincaré ball and a point $p$ on the boundary is given by:
        
        $d_B(z,p) = \log \frac{1 + ||z||}{1 - ||z||} - \log \frac{1 + |<z,p>|}{1 - |<z,p>|}$
        
    - **Exponential and Logarithmic Maps**: These maps are used to transform data between Euclidean space and hyperbolic space.
        
    - **Hyperbolic Neural Networks**: Neural network architectures specifically designed to operate in hyperbolic space. These include:
        
        - **Hyperbolic linear layers**: Generalizations of linear layers to hyperbolic space.
        - **Hyperbolic convolutional layers**: Extensions of convolutional layers to operate on data in hyperbolic space.
        - **Hyperbolic attention mechanisms**: Adaptations of attention mechanisms for hyperbolic representations.
- **Supervised Hyperbolic Learning Strategies**:
    - **Sample-to-Gyroplane Learning**: Classifying samples by learning hyperbolic hyperplanes, known as gyroplanes. This approach involves classifying samples based on their distances to these gyroplanes in hyperbolic space.
    - **Sample-to-Prototype Learning**: Representing classes as prototypes in hyperbolic space. This involves learning prototype representations for each class and classifying samples based on their proximity to these prototypes.
    - **Sample-to-Sample Learning**: Contrasting sample representations in hyperbolic space to learn discriminative embeddings.
- **Unsupervised Hyperbolic Learning Directions**:
    - **Generative Learning**: Using hyperbolic VAEs to learn latent representations and generate data that reflects hierarchical structure.
    - **Clustering**: Grouping data points in hyperbolic space to discover hierarchical relationships.
    - **Self-Supervised Learning**: Utilizing hyperbolic geometry in self-supervised learning tasks, such as contrastive learning and pretext tasks designed to exploit hierarchical relationships.
- **Advantages of Hyperbolic Deep Learning**:
    - **Hierarchical Learning**: Hyperbolic geometry is naturally suited for capturing hierarchical relationships, leading to improved performance in tasks involving hierarchies.
    - **Uncertainty Modeling**: The structure of hyperbolic space allows for representing uncertainty effectively, for instance, in action prediction where ambiguous actions can be represented by higher-level abstractions in the hierarchy.
    - **Low-Dimensional Learning**: Hyperbolic representations can achieve comparable performance to Euclidean counterparts with lower dimensionality, making them suitable for resource-constrained settings.
- Disadvantages: 
	- Computational challenge: numerical stability and speed of computation
	- No learning at scale atm. 
- **Open Research Questions**:
    - **Fully Hyperbolic Learning**: Exploring architectures where all layers operate in hyperbolic space, requiring adaptations of convolutions, attention mechanisms, and normalization techniques.
    - **Choice of Hyperbolic Models**: Determining the optimal hyperbolic model (Poincaré ball, Klein, Lorentz) for different tasks and operations.
    - **Evaluation Metrics**: Developing appropriate evaluation metrics for assessing the performance of hyperbolic models.

**HypLL**: The Hyperbolic Learning Library, provides a starting point for exploring hyperbolic deep learning.
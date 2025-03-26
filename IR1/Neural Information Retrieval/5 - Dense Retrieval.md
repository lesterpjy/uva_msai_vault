
## Introduction to Neural Information Retrieval

Neural IR methods use relevance judgments to learn how to rank search results. Unlike traditional IR systems that rely on exact keyword matching, neural approaches learn improved text representations that can capture semantic relationships between queries and documents.

The basic flow of information retrieval involves:

1. A user query (e.g., "black bear attacks")
2. A collection of text documents
3. A ranking method that scores the relevance of documents to the query
4. Returning a ranked list of documents

Traditional IR methods often rely on hand-crafted features like BM25 scores, spam scores, etc., while neural methods learn these representations from data.

## Types of Encoders in Neural IR

The slides identify two main types of encoders:

### 1. Cross-encoders

Cross-encoders process the query and document together in a single pass through a neural network. This allows the model to directly capture interactions between query and document terms at every layer of the network. While this approach generally <span style="color:rgb(172, 96, 230)">achieves higher accuracy</span>, it comes with a <span style="color:rgb(172, 96, 230)">significant computational cost since each query-document pair must be processed separately</span>.

### 2. Bi-encoders

Bi-encoders use separate encoding processes for the query and document:

- One encoder processes the query to create a query representation: $\eta(q)$
- Another encoder (or the same one) processes the document to create a document representation: $\eta(d)$
- A similarity function $\phi$ then computes a relevance score between these representations: $\phi(\eta(q), \eta(d))$

The key advantage of bi-encoders is that <span style="color:rgb(172, 96, 230)">document representations can be pre-computed and indexed, making retrieval much more efficient at query time.</span>

## Sparse vs. Dense Representations

### Sparse Representations

Traditional IR methods like BM25 use sparse vector representations where:

- Most dimensions are zero (only terms that appear in the document have non-zero values)
- Each dimension corresponds to a unique term in the vocabulary
- The BM25 formula calculates relevance based on term frequency and document frequency:

$$\text{BM25}(q, d) = \sum_{t \in q \cap d} \log \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot (1 - b + b \cdot \frac{l_d}{L})}$$

**Advantages of sparse representations:**

1. <span style="color:rgb(172, 96, 230)">Fast retrieval using inverted indices</span>, as queries are typically short
2. <span style="color:rgb(172, 96, 230)">Fast computation because the intersection of query and document</span> terms is usually small

**Main disadvantage:** <span style="color:rgb(172, 96, 230)">Terms must match exactly</span>, leading to the "lexical mismatch" problem. For example, a query about "fix my air conditioner" might not match a relevant document about "AC repair" because different terms are used to express the same concept.

### Dense Representations

Neural IR methods create dense vector representations where:

- All dimensions potentially contain non-zero values
- Vectors exist in a continuous space $\mathbb{R}^D$
- Semantically similar terms or documents are close to each other in the vector space

In dense retrieval:

1. A query "fix my air conditioner" is encoded into a vector $\eta(q)$
2. A document "... AC repair ..." is encoded into a vector $\eta(d)$
3. A similarity function $\phi$ (like inner product or cosine similarity) computes relevance: $\phi(\eta(q), \eta(d))$

This approach can handle lexical mismatch by learning that "air conditioner" and "AC" refer to the same concept, placing their vectors close together in the embedding space.

### Brute-force search

![[brute-force-search.png]]

<span style="color:rgb(172, 96, 230)">Brute force won't scale</span> as we're often searching for billions of texts.
When using dense representations, finding the most relevant documents for a query requires finding the nearest vectors in the embedding space. With large document collections (billions of documents), a brute-force approach that compares the query vector to every document vector becomes impractical.
### Approximate Nearest Neighbor Search

![[approximate_nearest_neighbor_search.png]]

<span style="color:rgb(172, 96, 230)">Approximate Nearest Neighbor</span> (ANN) search <span style="color:rgb(172, 96, 230)">trades some accuracy for speed</span>:

- Methods like k-means clustering partition the vector space
- At query time, the system first <span style="color:rgb(172, 96, 230)">identifies which partition(s) the query is closest to</span>
- Then, it <span style="color:rgb(172, 96, 230)">only compares the query to documents within those partitions</span>

Popular ANN libraries include Faiss, Annoy, and ScaNN. These methods address IR Challenge 1: Scalability and efficiency.

## Distance-based Approaches & Hard Negatives

Distance-based approaches use simple similarity functions between query and document vectors:

- Inner (dot) product: $\phi(u, v) = \eta(u) \cdot \eta(v)$
- Cosine similarity: measures the angle between vectors

Several important distance-based approaches:
### SentenceBERT

<span style="color:rgb(172, 96, 230)">SentenceBERT</span> uses siamese BERT networks to create sentence embeddings. It can represent text in several ways:

- CLS token representation
- mean of token embeddings
- max of token embeddings

SentenceBERT can be trained using classification (with a <span style="color:rgb(172, 96, 230)">softmax classifier</span>) or regression (using <span style="color:rgb(172, 96, 230)">cosine similarity</span>) <span style="color:rgb(172, 96, 230)">objectives</span>.

### Dense Passage Retrieval (DPR)

<span style="color:rgb(172, 96, 230)">DPR</span> is similar to SentenceBERT but uses:

- <span style="color:rgb(172, 96, 230)">Inner product as the similarity function</span> (rather than cosine similarity)
- <span style="color:rgb(172, 96, 230)">CLS token representation</span> (rather than mean or max pooling)
- <span style="color:rgb(172, 96, 230)">Contrastive learning with InfoNCE loss</span>

The InfoNCE loss formula is:

$$L(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-) = -\log \frac{e^{\text{sim}(q_i, p_i^+)}}{e^{\text{sim}(q_i, p_i^+)} + \sum_{j=1}^n e^{\text{sim}(q_i, p_{i,j}^-)}}$$

where $q_i$ is the query, $p_i^+$ is a positive (relevant) passage, and $p_{i,j}^-$ are negative (non-relevant) passages.
<span style="color:rgb(172, 96, 230)">Selection of negatives</span>:
- random
- non-relevant docs by BM25
- In batch examples
### Hard Negatives

The effectiveness of contrastive learning depends significantly on the <span style="color:rgb(172, 96, 230)">quality of negative examples</span>. "Hard negatives" are <span style="color:rgb(172, 96, 230)">non-relevant documents that are difficult to distinguish from relevant ones</span>.

For example, with a query about "Black Bear Attacks", potential negative examples might include:

1. A document about black bears being poached (contains both "black" and "bear" but not about attacks)
2. A document using "bear" in a different context ("does not bear on the safety...")
3. A document in German about black bear attacks (topically relevant but in a different language)

The best hard negatives are topically related but not relevant to the specific information need.

### ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)

![[ance.png | 400]]

![[ance_training.png | 500]]

<span style="color:rgb(172, 96, 230)">ANCE</span> improves on DPR by using better hard negatives:

- It regularly checkpoints the model during training
- <span style="color:rgb(172, 96, 230)">Uses the model from the previous checkpoint to identify hard negatives</span> for the current training iteration
- Alternatively, a more efficient approach <span style="color:rgb(172, 96, 230)">caches negative representations and slowly updates the cache</span>

This approach significantly outperforms basic DPR in experiments.

### Topic-Aware Sampling

Another approach <span style="color:rgb(172, 96, 230)">composes training batches such that in-batch negatives are hard</span> (from similar topics). This is competitive with ANCE but simpler to implement.

### CLEAR (Complementing LEexical Approaches with Semantic Residual embedding)

<span style="color:rgb(172, 96, 230)">CLEAR</span> is designed to complement traditional lexical matching (like BM25):

- It prepends special tokens to queries and documents (QRY or DOC)
- Uses an exact matching component (<span style="color:rgb(172, 96, 230)">lexical features</span>) alongside the <span style="color:rgb(172, 96, 230)">dense embeddings features</span>
- Focuses on "matching errors" as negatives
- Uses a <span style="color:rgb(172, 96, 230)">dynamic hinge loss margin</span>

## Comparison-based Approaches

Comparison-based approaches use more sophisticated <span style="color:rgb(172, 96, 230)">interaction patterns</span> between query and document terms, <span style="color:rgb(172, 96, 230)">rather than just comparing global representations</span>.

### ColBERT

![[colbert.png | 350]]

<span style="color:rgb(172, 96, 230)">ColBERT</span> uses a "late interaction" mechanism:

- Encodes each token in the query and document separately
- Computes a "MaxSim" operator that <span style="color:rgb(172, 96, 230)">finds the maximum similarity for each query token with any document token</span>
- <span style="color:rgb(172, 96, 230)">Sums these maximum similarities</span> to get the final score:

$$s_{q,d} = \sum_{i \in \eta(q)} \max_{j \in \eta(d)} \eta(q)_i \cdot \eta(d)_j^T$$

For example, with a query "green ocean turtle" and a document containing "reptiles such as sea turtles spend most of...", ColBERT would:

4. Find the maximum similarity of "green" with any document token
5. Find the maximum similarity of "ocean" with any document token
6. Find the maximum similarity of "turtle" with any document token
7. Sum these maximal similarities

![[colbert_score.png | 500]]

ColBERT is:

- 70x faster than BERT-large cross-encoder
- Achieves strong performance (MRR@10 of 0.360)
- However, it's not immediately compatible with standard ANN indices

In practice, <span style="color:rgb(172, 96, 230)">ColBERT uses the reranking pipeline</span>:
1. Identify candidates using ANN with query token embeddings
2. Rerank these candidates using the MaxSim mechanism

## Robustness & Out-of-domain Evaluation

An important challenge for neural IR methods is robustness across different domains and tasks. The <span style="color:rgb(172, 96, 230)">BEIR benchmark evaluates retrieval models on diverse datasets outside their training domain</span>.

Key findings:

- <span style="color:rgb(172, 96, 230)">Neural IR methods are generally less robust than traditional statistical methods</span> when applied to new domains
- Single-vector dense retrieval methods (like DPR) are the least robust
- Methods that use token-level interactions (like ColBERT) or <span style="color:rgb(172, 96, 230)">combine dense and sparse approaches show better cross-domain performance</span>
- Approaches are improving gradually (e.g., newer methods like DRAGON show better robustness)

## Conclusion

Neural IR approaches substantially improve query-document matching, with several key developments:

- Bi-encoders provide efficient first-stage retrieval
- ANN indices enable searching billions of documents
- Distance-based approaches with hard negatives (SentenceBERT, DPR, ANCE) show strong performance
- Comparison-based approaches like ColBERT capture more fine-grained term interactions
- Robustness remains a challenge, especially for single-vector methods

The field continues to advance with new methods that balance efficiency and effectiveness, with more focus on creating models that generalize well across domains.
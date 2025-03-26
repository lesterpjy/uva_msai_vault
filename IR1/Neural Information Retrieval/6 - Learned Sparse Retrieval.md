
## Introduction to Neural Information Retrieval

Information retrieval (IR) systems aim to find the most relevant documents from a collection in response to a user query. Traditional approaches like BM25 use term-based matching, but neural approaches can learn improved text representations for ranking.

In a typical retrieval pipeline:

1. A user submits a query (e.g., "black bear attacks")
2. The system matches this against a collection of documents
3. A ranking method (increasingly learned/neural) orders results by relevance
4. Performance is measured with metrics like MRR (Mean Reciprocal Rank) or Recall

## Dense vs. Sparse Retrieval: Key Differences
### Dense Retrieval (DR)

- Represents texts as continuous vectors in $\mathbb{R}^D$ (typically 768 dimensions)
- Elements are mostly non-zero values (hence "dense")
- Dimensions are latent and not directly interpretable
- Example representation: [0.8, -1.2, ..., 2.4, -0.3]
- Uses <span style="color:rgb(172, 96, 230)">approximate nearest neighbor (ANN)</span> search for retrieval
- Often<span style="color:rgb(172, 96, 230)"> slower and requires larger indices</span>

### Sparse Retrieval

- Represents texts as vectors where most elements are zero
- Dimensions correspond to vocabulary terms (interpretable)
- Example representation: {"amsterdam": 4.2, "is": 0.1, "going": 1.2, "to": 0.1, "rain": 2.3}
- Uses <span style="color:rgb(172, 96, 230)">inverted indices for fast retrieval</span>
- Generally <span style="color:rgb(172, 96, 230)">faster than dense methods</span>

When choosing between approaches:

- BM25 (sparse) offers <span style="color:rgb(172, 96, 230)">interpretability and efficiency</span> for long documents
- Dense models potentially handle vocabulary mismatch better but lose interpretability
- <span style="color:rgb(172, 96, 230)">Learned Sparse Retrieval (LSR) aims to combine the benefits of both</span>

## Document Expansion with doc2query

A key method for improving sparse retrieval is <span style="color:rgb(172, 96, 230)">document expansion</span>:
### doc2query
This approach <span style="color:rgb(172, 96, 230)">generates possible queries</span> that a user might use to find a particular document, then <span style="color:rgb(172, 96, 230)">uses these to expand the original text</span>.

![[doc2query.png | 550]]

How it works:
1. Train a seq2seq transformer model on pairs of <query, relevant document>
2. For each document, generate 5-40 possible queries using the model
3. Concatenate these generated queries to the original document
4. Index the expanded document


Benefits:
- <span style="color:rgb(172, 96, 230)">Addresses vocabulary mismatch</span>
- <span style="color:rgb(172, 96, 230)">Improves term re-weighting (important terms are repeated)</span>
- Significantly improves retrieval metrics (e.g., MRR@10 from 0.184 to 0.277)

## Recap

![[retrieval_pipeline_recap.png]]
## Learned Sparse Retrieval (LSR)

### BM25 as Query & Document Encoder
BM25, traditionally thought of as a ranking function, can be <span style="color:rgb(172, 96, 230)">interpreted as a specialized form of query and document encoding</span>:

$$\text{BM25}(q, d) = \sum_{i=1}^{n} \text{IDF}(q_i) \times \frac{tf(q_i, d) \times (k_1 + 1)}{tf(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

This can be rewritten to separate the query and document components:

$$= \sum_{j=1}^{|V|} 1_q(v_j) \cdot \text{IDF}(v_j) \times 1_d(v_j) \cdot \frac{tf(v_j,d) \times (k_1+1)}{tf(v_j,d) + k_1 \cdot \left(1-b+b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where:
- $1_q(v_j)$ indicates whether term $v_j$ appears in query
- $1_d(v_j)$ indicates whether term $v_j$ appears in document
- IDF(v_j) is the inverse document frequency of term $v_j$
- tf(v_j,d) is the term frequency of $v_j$ in document d
- |d| is the length of document d
- avgdl is the average document length
- k₁ and b are tuning parameters

This can be further simplified to:

$$= \sum_{i=1}^{|V|} f_Q(q)_i \times f_D(d)_i = \langle f_Q(q), f_D(d) \rangle$$

This final form shows BM25 as a dot product between two vectors, meaning BM25 is equivalent to <span style="color:rgb(172, 96, 230)">encoding both query and document into sparse vectors and computing their dot product</span>. This reinterpretation bridges traditional retrieval methods with the vector space perspective.

### Learned Sparse Representations (LSR)

Learned Sparse Representations (LSR) aims to combine the efficiency of sparse vector operations with the semantic understanding of neural models.

#### LSR Concept

LSR approaches predict query term weights and document term weights over the vocabulary V:

- <span style="color:rgb(172, 96, 230)">Query encoder</span>: Given a query, produce term weights over vocab V
- <span style="color:rgb(172, 96, 230)">Document encoder</span>: Given a doc, produce term weights over vocab V

Unlike traditional methods (like TF-IDF) where weights are based on explicit term occurrence statistics, LSR methods learn to predict term importance. Crucially, they can assign weights to terms that don't explicitly appear in the text but are semantically relevant.

#### LSR in Practice

The <span style="color:rgb(172, 96, 230)">process</span> works as follows:

1. <span style="color:rgb(172, 96, 230)">Indexing</span>: Document encoder creates sparse representations that feed into inverted indices
2. <span style="color:rgb(172, 96, 230)">Retrieval</span>: Query encoder creates a sparse query representation used for retrieval

The term weights in LSR are not binary (present/absent) but learned weights indicating term salience. The BM25 formula serves as an inspiration, but rather than using predefined statistics, transformer-based models (like BERT) predict term salience.

![[lsr.png]]

Transformer encoders process the input text (document or query) and output sparse vectors where most entries are zero, but important terms receive non-zero weights. These <span style="color:rgb(172, 96, 230)">sparse vectors are compatible with traditional inverted index structures, allowing for efficient retrieval</span>.

### Advantages
#### Dense Representations
- Capture semantic similarity even when vocabulary doesn't overlap
- Can understand paraphrases, synonyms, and related concepts
- Often provide <span style="color:rgb(172, 96, 230)">better recall for relevant documents</span>
#### Learned Sparse Representations
- Maintain <span style="color:rgb(172, 96, 230)">efficiency of inverted indices</span> for fast retrieval
- Understand semantic importance <span style="color:rgb(172, 96, 230)">beyond simple term frequency</span>
- <span style="color:rgb(172, 96, 230)">Compatible</span> with existing infrastructure
- Can learn to <span style="color:rgb(172, 96, 230)">assign weights to terms not explicitly in the document but semantically related</span>

### LSR Ingredients

LSR methods combine three key components:
#### 1. Sparse Encoder
<span style="color:rgb(172, 96, 230)">Encoders may be separate or shared between the query and document</span>.
Different types of encoders transform text into sparse vectors:

![[sparse_encoders.png]]

- Binary (BoW) encoder: Simple term presence (0/1)
- MLP encoder: Neural network assig ns learned weights to terms
- expMLP encoder: Binary + expansion + learned weights

![[sparse_encoders_2.png]]

- clsMLM encoder: Uses CLS token and MLM head for predicting logits over vocab.
- MLM encoder: Uses Masked Language Model head to predict logits over vocab, max-pooling over inputs to get term weights of size |V|.

#### 2. Sparse Regularization and Pruning

Controls the <span style="color:rgb(172, 96, 230)">trade-off between effectiveness and efficiency by limiting non-zero weights</span>:

- <span style="color:rgb(172, 96, 230)">Top-k pruning</span>: 
	- Select top-k highest weighted terms (<span style="color:rgb(172, 96, 230)">post-processing</span>)
	- drop non-positive terms
- <span style="color:rgb(172, 96, 230)">L1 regularizer</span>:
	- Limit number and magnitude of positive term
	- Given a vector representation r, <span style="color:rgb(172, 96, 230)">penalize the sum of absolute weights</span> during training: $$L1(r) = \sum_{i=1}^{|V|} abs(r_i)$$
- FLOPS regularizer: Similar to L2, limits computational complexity

#### 3. Supervision Approaches

- <span style="color:rgb(172, 96, 230)">Term-level supervision</span>: Train to predict term importance (e.g., DeepCT)
- <span style="color:rgb(172, 96, 230)">Passage-level supervision</span>: Train using relevance judgments (e.g., SPLADE)
- Can incorporate negative examples and distillation techniques
- <span style="color:rgb(172, 96, 230)">Label type: human labels, teachers' labels (distillation), weak-labels (self-labels)</span>

## Specific LSR Approaches

### DeepCT
<span style="color:rgb(172, 96, 230)">Query: MLP Encoder</span>
<span style="color:rgb(172, 96, 230)">Document: MLP Encoder</span>

![[deepct.png]]

DeepCT uses <span style="color:rgb(172, 96, 230)">term-level weak supervision</span> to predict term weights:
- Takes document text as input
- Outputs term importance scores
- Training <span style="color:rgb(172, 96, 230)">target is term recall</span> with respect to relevant documents 
- <span style="color:rgb(172, 96, 230)">term recall</span> = (rel. docs with term) / (num of rel. docs)
- Uses <span style="color:rgb(172, 96, 230)">regression loss</span>: $\text{loss} = \sum_t (\hat{y}_{t,d} - y_{t,d})^2$

![[deepct_predict.png]]

When indexing with DeepCT:
- For each term in a document, predict its weight
- <span style="color:rgb(172, 96, 230)">Use these weights in place of traditional term frequencies</span>
- Results show significant improvement over BM25 (MRR@10 from 0.184 to 0.243)

### DeepImpact
<span style="color:rgb(172, 96, 230)">Query: Binary Encoder</span>
<span style="color:rgb(172, 96, 230)">Document: expMLP Encoder</span>

![[deepimpact.png]]

<span style="color:rgb(172, 96, 230)">DeepImpact combines document expansion with term weighting</span>:
1. Use doc2query to expand documents
2. Use a DeepCT-like term scorer to weight terms
3. Index documents with these expanded terms and weights

Results show further improvement (MRR@10 from 0.243 to 0.326) while maintaining reasonable efficiency.

### SPLADE
<span style="color:rgb(172, 96, 230)">Query: MLM Encoder</span>
<span style="color:rgb(172, 96, 230)">Document: MLM Encoder</span>

<span style="color:rgb(172, 96, 230)">SPLADE (Sparse Lexical and Expansion Model)</span> uses the <span style="color:rgb(172, 96, 230)">MLM head for both weighting and expansion</span>:
- Input: query Q or document D
- Output: sparse vector of dimension |V| (vocabulary size)
- MLM encoder:
    1. For each term in the input, the MLM head predicts scores for all |V| terms in the vocabulary
    2. For each vocabulary term, the maximum score across all input positions becomes that term's weight
    3. These weights represent the input as a |V|-dimensional vector
- Trained with <span style="color:rgb(172, 96, 230)">FLOPS regularizer and relevance supervision</span>
- Achieves MRR@10 of 0.369

## Impact of Encoder and Regularization Choices

Experimental analysis revealed important design choices:
1. Document weighting is crucial for effectiveness
2. Query weighting provides modest but consistent gains
3. Document expansion improves over weighting only but adds latency
4. Query expansion adds latency, but no benefit with doc expansion
5. Top-k pruning provides no benefit over regularization.

## Out-of-domain Evaluation (BEIR)

The BEIR benchmark evaluates how models perform on unseen domains. Results show:
- Sparse methods (SPLADE) outperform single-vector dense methods
- LSR approaches are often more robust to domain shift
- BM25+CE (cross-encoder) remains competitive

## Conclusion and Trade-offs

![[lsr_vs_dr.png]]

<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
Learned Sparse Retrieval offers several advantages:

- Interpretability and transparency (term-based)
- Efficient inference with existing infra on fast/small index
- Reduces vocabulary mismatch through neural expansion and weighting
- Ability to boost terms/ edit representations
- Competitive performance with dense methods

Key limitations compared to dense retrieval:

- Only supports positive weights (difficult to encode negative information)
- Soft matching requires explicit expansion (vs. expressive representation)


Overall, LSR represents a promising approach that combines the efficiency of traditional sparse retrieval with the effectiveness of neural methods, particularly as a first-stage retrieval component in multi-stage systems.
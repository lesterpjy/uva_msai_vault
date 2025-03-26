
## Introduction to Neural Information Retrieval

<span style="color:rgb(172, 96, 230)">Neural Information Retrieval</span> (Neural IR) refers to methods that use machine learning, specifically neural networks, to improve information retrieval effectiveness. The key insight is <span style="color:rgb(172, 96, 230)">moving from traditional retrieval based on exact term matching to methods that understand semantic relationships</span> between query and documents.

The basic IR problem involves:
- A user submitting a query (e.g., "black bear attacks")
- A system retrieving relevant documents from a collection
- Documents being scored and ranked based on their relevance to the query

Neural IR approaches this by learning improved text representations for ranking, rather than relying solely on handcrafted features.

### Traditional vs. Neural Approaches

**Traditional IR Methods**:
- Use hand-crafted scoring functions like BM25
- Focus on <span style="color:rgb(172, 96, 230)">exact term matching</span>
- Score documents based on formula:
- $score(D, Q) = \sum_{i=1}^{|Q|} IDF(q_i) * TF(q_i, D)$

**Neural IR Methods**:
- <span style="color:rgb(172, 96, 230)">Learn representations</span> directly from data
- Perform <span style="color:rgb(172, 96, 230)">soft matching of terms</span> rather than exact matching
- <span style="color:rgb(172, 96, 230)">Capture semantic relationships</span> between terms

### Advantages of Neural IR Methods

1. **Less Handcrafting**:
    - Traditional IR requires manually defining elements like IDF calculation (e.g., $\log \frac{N}{n_i}$, $\log (1 + \frac{N}{n_i})$)
    - Neural models learn these patterns automatically from training data
2. **Soft Matching of Terms**:
    - Neural methods can <span style="color:rgb(172, 96, 230)">identify semantically related terms</span> even when they don't match exactly
    - Example: A query with "document ranking" can match documents containing "text retrieval" or "web search" because the neural model understands these are semantically related
    - This addresses the fundamental<span style="color:rgb(172, 96, 230)"> lexical mismatch problem</span> in IR

## Test Collections for Neural IR Research

Several important test collections are used to evaluate neural IR methods:
1. **TREC Robust04**:
    - News articles
    - Contains detailed query descriptions (title, description, narrative)
    - About 250 queries, ~528,000 documents
    - Average document length: ~549 words
2. **MS MARCO**:
    - Web passages and documents
    - Contains short questions derived from real search queries
    - Large scale: ~8.8M passages, ~3.2M documents
    - Passages are quite short (average ~56 words)
    - Documents are longer (average ~1,131 words)
    - Sparse judgments (typically just 1 relevant document per query)
3. **TREC Deep Learning**:
    - Uses same passages/documents as MS MARCO
    - But has denser, more comprehensive relevance judgments

These collections allow researchers to compare different retrieval methods on standardized datasets.

## Neural Ranking with Static Embeddings

<span style="color:rgb(172, 96, 230)">IR Challenge: Lexical mismatch, queries and documents use different words to express the same meaning.</span>

### Word Embeddings Basics

Early neural IR approaches used static word embeddings like Word2Vec, GloVe, or FastText that represent words as dense vectors (typically 100-300 dimensions).

- <span style="color:rgb(172, 96, 230)">soft matching</span> in traditional methods possible but less effective (ex: stemming)
- <span style="color:rgb(172, 96, 230)">query expansion</span>: add related terms to query
- vector embedding allows comparison of different terms
- ranking with embedding: place document terms in similarity buckets, then computing relevance based on size of buckets:

![[basic_embedding_ranking.png | 400]]
### Deep Relevance Matching Model (DRMM)

The DRMM model was one of the pioneering neural IR models using embeddings:
- It creates a similarity matrix between query and document terms using cosine similarity of their embeddings
- For each query term, it computes a histogram of similarity scores with all document terms
- These histograms are processed through a feed-forward network to get a relevance score for each query term
- The final relevance score is a weighted sum of per-term scores

The process works as follows:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. Load document D from forward index and represent D as list of embeddings
2. Compute histograms $h(t)$ for each query term $t$, capturing similarity between query and document terms.
3. Compute term score $z(h(t))$ using feedforward network $z$.
4. Compute weight $g(t)$ for each query term (IDF or embeddings)
5. Relevance score: sum of $z(h(t)) * g(t)$ over all query terms.

DRMM is <span style="color:rgb(172, 96, 230)">trained using binary cross-entropy loss</span>:
$$
L = - \sum_{j \in J_{pos}} \log(s_j) - \sum_{j \in J_{neg}} \log(1 - s_j)
$$

Where $J_{pos}$ are relevant documents and $J_{neg}$ are non-relevant documents.

## Transformers & Contextualized Embeddings

### From Static to Contextualized Embeddings
A key limitation of models like Word2Vec is that they assign the same embedding to a word regardless of context. For example, "bank" always has the same representation whether it means financial institution or river bank.
<span style="color:rgb(172, 96, 230)">Contextualized embeddings </span>solve this by generating different representations based on context.
- one static embedding per word $\leftrightarrow$ embedding based on co-occurrences

### BERT Overview
BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP and IR:
- Pre-trained on massive text corpora using self-supervised learning
- Uses transformer architecture with multi-head attention
- Creates contextualized word representations
- Input format: `[CLS] query [SEP] document [SEP]`
- Trained on masked language modeling (predicting masked words) and next sentence prediction

BERT's input is the sum of:
- Token embeddings (the actual words)
- Segment embeddings (which sequence the token belongs to)
- Position embeddings (location in the sequence)

### Limitations of BERT for IR

<span style="color:rgb(172, 96, 230)">IR Challenge: models that effectively bridge the gap are non-explainable.</span>
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. **Computational Expense**:
    - Even a small BERT model has ~110 million parameters
    - Processing is slow, especially for many document-query pairs
2. **Length Restrictions**:
    - Limited to 512 tokens maximum input length
    - Most documents are longer than this limit

- Solution 1: multi-stage ranking pipeline (reranking)
	- Identify candidate documents
	- re-rank with transformer
- Solution 2: Pre-compute doc. representation
	- Dense retrieval
	- Learned sparse retrieval

#### Approaches
##### Cross Encoder

![[cross-encoder.png | 400]]

- Input Q-D pair
- Model outputs score
- slow but robust
- DS: forward index
##### Dense Retrieval

![[dense-retrieval.png]]

- Input Q or D
- Outputs vector
- Score by comparing Q-D vectors
- Faster, less effective or robust
- DS: ANN index
##### Learned Sparse Retrieval

![[learned-sparse-retrieval.png | 400]]

- Input Q or D
- Outputs term weights replacing TF-IDF
- Faster than cross-encoder
- Newest/Least studied
- DS: inverted index

## Transformer Approaches for Reranking

### Multi-stage Retrieval Pipeline

![[multi-stage-retrieval.png]]

To address the computational limitations of transformers, a multi-stage pipeline is typically used: <span style="color:rgb(172, 96, 230)">Initial retrieval -> Neural reranking</span>
1. **Initial Retrieval** (e.g., BM25):
    - Fast retrieval from millions of documents
    - Retrieves candidate set (~1000 documents)
2. **Neural Reranking** (e.g., BERT-based reranker):
    - More expensive but more effective ranking
    - Applied only to the candidate set
    - Can further be divided into:
        - <span style="color:rgb(172, 96, 230)">Point-wise reranking</span> (individual document scoring)
        - <span style="color:rgb(172, 96, 230)">Pair-wise reranking</span> (comparing document pairs)

This approach <span style="color:rgb(172, 96, 230)">balances efficiency and effectiveness:</span>
- Initial retrieval: Good scalability, fast, but less effective
- Reranking: Less scalable, slower, but much more effective

### Cross-encoder Reranking Approaches

#### 1. MonoBERT

![[monobert.png | 450]]

A simple but effective approach:
- Takes a <span style="color:rgb(172, 96, 230)">query-document pair as input</span>: `[CLS] query [SEP] document [SEP]`
- Adds a <span style="color:rgb(172, 96, 230)">binary classification layer on top of BERT</span>
- Outputs a probability score that the document is relevant
- Score is used to <span style="color:rgb(172, 96, 230)">rerank candidate documents</span>

MonoBERT shows substantial improvements over BM25 baselines in benchmarks (e.g., improving nDCG@10 from ~0.51 to ~0.74 on TREC Deep Learning).

#### 2. CEDR (Contextualized Embeddings for Document Ranking)

![[cedr.png | 450]]

CEDR combines contextualized embeddings with traditional IR models:
- Uses contextualized token embeddings from BERT
- Feeds these into pre-BERT neural ranking architectures (like DRMM)
- <span style="color:rgb(172, 96, 230)">Preserves the interaction-focused architecture of earlier models</span>
- But <span style="color:rgb(172, 96, 230)">benefits from the improved representations of BERT</span>

CEDR models consistently outperform both vanilla BERT and traditional models like DRMM with static embeddings.

#### 3. MonoT5

![[monot5.png | 250]]

An evolution beyond BERT-based rerankers:

- Uses T5 (Text-to-Text Transfer Transformer) instead of BERT
- <span style="color:rgb(172, 96, 230)">Recasts reranking as a text generation task</span>
- Prompt format: "Query: {query} Document: {document} Relevant:"
- Model generates "true" or "false"
- Score is derived from the probability of generating "true", no classification needed.

#### 4. DuoT5

![[duot5.png]]

A pairwise ranking approach:
- Takes a query and two documents
- <span style="color:rgb(172, 96, 230)">Predicts which document is more relevant</span>
- <span style="color:rgb(172, 96, 230)">More computationally expensive</span> than pointwise ranking
- But often more effective, especially for top positions
- Usually applied only to a small set of top candidates (~50 documents)

## Handling Long Documents

Since transformers like BERT have input length limitations (typically 512 tokens), specialized techniques are needed for long documents:

### 1. Score Aggregation

- Split document into chunks
- <span style="color:rgb(172, 96, 230)">Score each chunk individually</span> using a transformer model
- Aggregate chunk scores to get a document score
- Aggregation methods:
    - <span style="color:rgb(172, 96, 230)">MaxP</span>: Take the maximum score of any chunk
    - <span style="color:rgb(172, 96, 230)">SumP</span>: Sum the scores of all chunks
    - <span style="color:rgb(172, 96, 230)">FirstP</span>: Use only the first chunk's score

Research shows MaxP typically performs best for news articles.

### 2. Representation Aggregation (PARADE)

PARADE (PAssage Representation Aggregation for Document rE-ranking):
- Split document into chunks/passages
- Process each passage with BERT to get [CLS] token embeddings
- <span style="color:rgb(172, 96, 230)">Aggregate these embeddings to create a document representation</span>
- <span style="color:rgb(172, 96, 230)">Aggregation methods</span>:
    - Simple averaging
    - Max pooling
    - Attention-weighted averaging
    - Using transformer layers to process passage representations

PARADE with transformer aggregation achieves better results than simple score aggregation methods.

### 3. Extended Context Models (Longformer, QDS)

Models specifically designed for longer inputs:
- Longformer uses <span style="color:rgb(172, 96, 230)">sparse attention patterns to handle longer sequences</span>
- QDS (Query-Directed Sparse) Transformer specializes attention patterns for IR
- These models can process more tokens while maintaining reasonable computational efficiency

## Open-domain Question Answering

Question answering (QA) is a major application of neural IR
### Traditional vs. Open-domain QA
- **Machine Reading Comprehension** (e.g., SQuAD):
    - Given a document, answer questions about it
    - Assumes the relevant document is already provided
- **Open-domain QA**:
    - Must first retrieve relevant documents
    - Then answer based on retrieved information
    - More challenging but more practical
<span style="color:rgb(172, 96, 230)">Harder in practice: we don't assume relevant docs are available.</span>
### Two Main Approaches

1. **Non-parametric (Retrieval-Based)**:
    - <span style="color:rgb(172, 96, 230)">Retrieve relevant documents </span>from an external knowledge source
    - Extract or <span style="color:rgb(172, 96, 230)">generate answer based on retrieved content</span>
    - Examples: DPR (Dense Passage Retrieval) + reader models
    - Best suited for queries requiring factual knowledge, especially for rare topics
2. **Parametric (Memory-Based)**:
    - <span style="color:rgb(172, 96, 230)">Train large language models to memorize knowledge during pre-training</span>
    - Generate answers directly from model parameters
    - Examples: T5, GPT models
    - Effective for common knowledge, but may hallucinate for rare topics

### The Role of Retrieval in QA

Retrieval is especially important for:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Open-domain questions (no provided context)
- Non-factoid questions (requiring longer explanations)
- Long-tail queries (rare topics not well-represented in training data)
- Time-dependent information (that may change after model training)

## Conclusion

Neural IR approaches offer substantial improvements in search quality:
- Move beyond exact term matching to semantic understanding
- Address the vocabulary mismatch problem
- Show dramatic improvements on benchmarks (8-19 point improvements)

Different neural architectures are suited to different IR scenarios:
- Cross-encoders for high-precision reranking
- Dense retrievers for efficient first-stage retrieval
- Specialized methods for long documents and question answering

This field continues to evolve rapidly, with promising directions in:
- Dense retrieval
- Learned sparse retrieval
- Retrieval-augmented generation (RAG)

As these technologies mature, they will likely become standard components in search engines and knowledge-intensive applications.
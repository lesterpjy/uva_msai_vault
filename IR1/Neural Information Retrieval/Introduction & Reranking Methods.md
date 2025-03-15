
## Introduction to Neural Information Retrieval

Neural Information Retrieval (Neural IR) refers to methods that use machine learning, specifically neural networks, to improve information retrieval effectiveness. The key insight is moving from traditional retrieval based on exact term matching to methods that understand semantic relationships between query and documents.

The basic IR problem involves:
- A user submitting a query (e.g., "black bear attacks")
- A system retrieving relevant documents from a collection
- Documents being scored and ranked based on their relevance to the query

Neural IR approaches this by learning improved text representations for ranking, rather than relying solely on handcrafted features.

### Traditional vs. Neural Approaches

**Traditional IR Methods**:
- Use hand-crafted scoring functions like BM25
- Focus on exact term matching
- Score documents based on formula:
- $score(D, Q) = \sum_{i=1}^{|Q|} IDF(q_i) * TF(q_i, D)$
- where:
    - $TF(q_i, D)$ is the term frequency (how many times term $q_i$ appears in document $D$)
    - $IDF(q_i)$ is inverse document frequency (measure of how rare the term is across all documents)

**Neural IR Methods**:
- Learn representations directly from data
- Perform soft matching of terms rather than exact matching
- Capture semantic relationships between terms

### Advantages of Neural IR Methods

1. **Less Handcrafting**:
    - Traditional IR requires manually defining elements like IDF calculation (e.g., $\log \frac{N}{n_i}$, $\log (1 + \frac{N}{n_i})$)
    - Neural models learn these patterns automatically from training data
2. **Soft Matching of Terms**:
    - Neural methods can identify semantically related terms even when they don't match exactly
    - Example: A query with "document ranking" can match documents containing "text retrieval" or "web search" because the neural model understands these are semantically related
    - This addresses the fundamental lexical mismatch problem in IR

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

### Word Embeddings Basics

Early neural IR approaches used static word embeddings like Word2Vec, GloVe, or FastText that represent words as dense vectors (typically 100-300 dimensions).

These embeddings capture semantic relationships between words, so that similar words have similar vector representations. For example, "cat" and "kitten" would have embeddings with high cosine similarity.

### Deep Relevance Matching Model (DRMM)

The DRMM model was one of the pioneering neural IR models using embeddings:
- It creates a similarity matrix between query and document terms using cosine similarity of their embeddings
- For each query term, it computes a histogram of similarity scores with all document terms
- These histograms are processed through a feed-forward network to get a relevance score for each query term
- The final relevance score is a weighted sum of per-term scores

The process works as follows:
1. Load document and embeddings
2. Compute histograms capturing similarity between query and document terms
3. Process histograms through a neural network
4. Weight query terms (often by IDF)
5. Combine scores for the final relevance judgment

DRMM is trained using binary cross-entropy loss:
$$
L = - \sum_{j \in J_{pos}} \log(s_j) - \sum_{j \in J_{neg}} \log(1 - s_j)
$$

Where $J_{pos}$ are relevant documents and $J_{neg}$ are non-relevant documents.

## Transformers & Contextualized Embeddings

### From Static to Contextualized Embeddings
A key limitation of models like Word2Vec is that they assign the same embedding to a word regardless of context. For example, "bank" always has the same representation whether it means financial institution or river bank.
Contextualized embeddings solve this by generating different representations based on context.

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
## Transformer Approaches for Reranking

### Multi-stage Retrieval Pipeline

![[multi-stage-retrieval.png]]

To address the computational limitations of transformers, a multi-stage pipeline is typically used:
1. **Initial Retrieval** (e.g., BM25):
    - Fast retrieval from millions of documents
    - Retrieves candidate set (~1000 documents)
2. **Neural Reranking** (e.g., BERT-based reranker):
    - More expensive but more effective ranking
    - Applied only to the candidate set
    - Can further be divided into:
        - Point-wise reranking (individual document scoring)
        - Pair-wise reranking (comparing document pairs)

This approach balances efficiency and effectiveness:
- Initial retrieval: Good scalability, fast, but less effective
- Reranking: Less scalable, slower, but much more effective

### Cross-encoder Reranking Approaches

#### 1. MonoBERT

![[monobert.png | 450]]

A simple but effective approach:
- Takes a query-document pair as input: `[CLS] query [SEP] document [SEP]`
- Adds a binary classification layer on top of BERT
- Outputs a probability score that the document is relevant
- Score is used to rerank candidate documents

MonoBERT shows substantial improvements over BM25 baselines in benchmarks (e.g., improving nDCG@10 from ~0.51 to ~0.74 on TREC Deep Learning).

#### 2. CEDR (Contextualized Embeddings for Document Ranking)

![[cedr.png | 450]]

CEDR combines contextualized embeddings with traditional IR models:
- Uses contextualized token embeddings from BERT
- Feeds these into pre-BERT neural ranking architectures (like DRMM)
- Preserves the interaction-focused architecture of earlier models
- But benefits from the improved representations of BERT

CEDR models consistently outperform both vanilla BERT and traditional models like DRMM with static embeddings.

#### 3. MonoT5

![[monot5.png | 250]]

An evolution beyond BERT-based rerankers:

- Uses T5 (Text-to-Text Transfer Transformer) instead of BERT
- Recasts reranking as a text generation task
- Prompt format: "Query: {query} Document: {document} Relevant:"
- Model generates "true" or "false"
- Score is derived from the probability of generating "true", no classification needed.

#### 4. DuoT5

![[duot5.png]]

A pairwise ranking approach:
- Takes a query and two documents
- Predicts which document is more relevant
- More computationally expensive than pointwise ranking
- But often more effective, especially for top positions
- Usually applied only to a small set of top candidates (~50 documents)

## Handling Long Documents

Since transformers like BERT have input length limitations (typically 512 tokens), specialized techniques are needed for long documents:

### 1. Score Aggregation

- Split document into chunks
- Score each chunk individually using a transformer model
- Aggregate chunk scores to get a document score
- Aggregation methods:
    - MaxP: Take the maximum score of any chunk
    - SumP: Sum the scores of all chunks
    - FirstP: Use only the first chunk's score

Research shows MaxP typically performs best for news articles.

### 2. Representation Aggregation (PARADE)

PARADE (PAssage Representation Aggregation for Document rE-ranking):
- Split document into chunks/passages
- Process each passage with BERT to get [CLS] token embeddings
- Aggregate these embeddings to create a document representation
- Aggregation methods:
    - Simple averaging
    - Max pooling
    - Attention-weighted averaging
    - Using transformer layers to process passage representations

PARADE with transformer aggregation achieves better results than simple score aggregation methods.

### 3. Extended Context Models (Longformer, QDS)

Models specifically designed for longer inputs:
- Longformer uses sparse attention patterns to handle longer sequences
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

### Two Main Approaches
1. **Non-parametric (Retrieval-Based)**:
    - Retrieve relevant documents from an external knowledge source
    - Extract or generate answer based on retrieved content
    - Examples: DPR (Dense Passage Retrieval) + reader models
    - Best suited for queries requiring factual knowledge, especially for rare topics
2. **Parametric (Memory-Based)**:
    - Train large language models to memorize knowledge during pre-training
    - Generate answers directly from model parameters
    - Examples: T5, GPT models
    - Effective for common knowledge, but may hallucinate for rare topics

### The Role of Retrieval in QA

Retrieval is especially important for:
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
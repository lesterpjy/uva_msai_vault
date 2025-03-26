## Introduction and Course Overview

The Information Retrieval 1 course covers the fundamental concepts and techniques used in modern information retrieval systems. Based on the slides, the course explores several key areas:

1. **IR foundations**: Basic concepts, indexing methods, and text analysis
2. **Retrieval models**: From traditional term-based approaches to neural networks
3. **Evaluation methodologies**: How to assess IR system performance
4. **Advanced techniques**: Neural IR, generative systems, and specialized scenarios

## Text Analysis and Preprocessing

Before any retrieval can happen, documents need to be processed. Text analysis in IR involves several key steps:

1. **Tokenization**: Breaking text into individual words or tokens
    
    - Example: "Information Retrieval is fascinating!" → ["Information", "Retrieval", "is", "fascinating", "!"]
2. **Case normalization**: Converting terms to lowercase to ensure consistency
    
    - Example: "Information" → "information"
3. **Stop-word removal**: Filtering common words that add little meaning (the, is, a, etc.)
    
    - These words typically occur in almost all documents and don't help distinguish content
4. **Stemming**: Reducing words to their root/stem form
    
    - Example: "retrieving", "retrieved", "retrieval" → "retriev"
    - Common algorithms include Porter stemmer and Snowball stemmer
5. **Phrase handling**: Identifying and preserving meaningful multi-word expressions
    
    - Example: Treating "New York" as a single unit rather than separate words

These preprocessing steps dramatically impact retrieval performance by standardizing document representation and reducing the vocabulary size.

## Indexing Structures

The slides present three main indexing structures:

### 1. Forward Index

A forward index maps documents to their terms:

- Doc1 → ["Apple", "is", "a", "company"]
- Doc2 → ["Apple", "is", "a", "fruit"]

This structure is useful for:

- Document summarization
- Quickly accessing all terms in a document
- Supporting certain retrieval models that need full document content

### 2. Inverted Index

An inverted index maps terms to the documents containing them:

- "apple" → {Doc1: [0], Doc2: [0]}
- "is" → {Doc1: [1], Doc2: [1]}
- "company" → {Doc1: [3]}
- "fruit" → {Doc2: [3]}

The numbers in brackets represent the positions of terms in documents, enabling phrase queries and proximity searches. This is the most common index structure in IR systems because it allows efficient retrieval of documents containing specific query terms.

### 3. Vector-based Index

Vector indices store documents as vectors in a high-dimensional space. These are especially important for:

- Dense retrieval models using neural embeddings
- Approximate nearest neighbor search
- Similarity-based retrieval

## Unsupervised Retrieval Models

The course covers several classic retrieval models:

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

A statistical measure that evaluates how important a word is to a document in a collection:

- Term Frequency (TF): How often a term appears in a document
- Inverse Document Frequency (IDF): Measures how important a term is across the collection

### 2. BM25 (Best Matching 25)

An advanced ranking function used by search engines like Elasticsearch:

$$\text{BM25} = \sum_{\text{unique } t \in q} \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1((1 - b) + b \cdot (l_d/l_{\text{avg}})) + \text{tf}_{d,t}} \cdot \frac{(k_3 + 1)\text{tf}_{q,t}}{k_3 + \text{tf}_{q,t}} \cdot \text{idf}(\text{df}_t)$$

Where:

- $\text{tf}_{d,t}$ is the term frequency of term $t$ in document $d$
- $\text{tf}_{q,t}$ is the term frequency in the query
- $l_d$ is document length and $l_{\text{avg}}$ is average document length
- $k_1$, $k_3$, and $b$ are parameters (typically $k_1 = 1.2$, $k_3 = 8$, $b = 0.75$)

BM25 improves upon simpler models by accounting for document length and diminishing returns from term frequency.

### 3. Query Likelihood Models (QLM)

These models are based on probabilistic language modeling:

#### Maximum Likelihood Estimation

The basic model that suffers from zero-probability problems when query terms don't appear in documents.

#### Jelinek-Mercer Smoothing

$$\hat{P}_\lambda(w|d) = \lambda\frac{\text{tf}(w; d)}{|d|} + (1 - \lambda)\frac{\text{tf}(w; C)}{|C|}$$

This interpolates between document and collection probabilities using parameter $\lambda$.

#### Dirichlet Prior Smoothing

$$p_\mu(w|\hat{\theta}_d) = \frac{|d|}{|d| + \mu}\frac{\text{tf}(w; d)}{|d|} + \frac{\mu}{\mu + |d|}p(w|C)$$

Uses a Bayesian approach with parameter $\mu$ controlling smoothing strength, often providing better results for varied document lengths.

### 4. Positional Language Models

These extend basic language models by considering term positions, enabling better handling of word order and proximity.

## Evaluation Methodologies

Information retrieval systems are evaluated using:

### 1. Test Collections

Standard benchmarks consisting of:

- Document collections
- Query sets
- Relevance judgments (ground truth)

Examples include TREC, CLEF, and MS MARCO collections.

### 2. Depth-Pooling

A technique to create relevance judgments by:

6. Running multiple IR systems on the same queries
7. Taking the top-k results from each system
8. Manually judging only these pooled results
9. Assuming non-judged documents are non-relevant

This makes large-scale evaluation feasible without judging all document-query pairs.

### 3. Evaluation Metrics

#### Precision

The fraction of retrieved documents that are relevant: $$\text{Precision} = \frac{\text{relevant documents} \cap \text{retrieved documents}}{\text{retrieved documents}}$$

#### Recall

The fraction of relevant documents that are retrieved: $$\text{Recall} = \frac{\text{relevant documents} \cap \text{retrieved documents}}{\text{relevant documents}}$$

#### Average Precision (AP)

The average of precision values calculated at each relevant document in the ranked list: $$\text{AP} = \frac{\sum_{r=1}^{n} (P(r) \times \text{rel}(r))}{\text{number of relevant documents}}$$ Where $P(r)$ is precision at rank $r$ and $\text{rel}(r)$ is 1 if document at rank $r$ is relevant, 0 otherwise.

#### Normalized Discounted Cumulative Gain (nDCG)

Measures the usefulness of a ranking based on graded relevance: $$\text{nDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$$

Where DCG is: $$\text{DCG}_p = \sum_{i=1}^{p} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$$

And IDCG is the DCG of the ideal ranking.

## Neural IR

Neural IR approaches use deep learning to improve retrieval performance:

### 1. Reranking with Cross-Encoders

- **Input**: Both query and document together
- **Output**: Relevance score
- **Data Structure**: Forward index
- **Example**: monoBERT

Cross-encoders like BERT take both the query and document as a single input and produce a relevance score. They achieve high accuracy but are computationally expensive as they need to process query-document pairs individually.

### 2. Dense Retrieval with Bi-Encoders

- **Input**: Query or document separately
- **Output**: Dense vector representations
- **Data Structure**: Vector index for approximate nearest neighbor search
- **Learning Method**: Contrastive learning
- **Examples**: SentenceBERT, ANCE, ColBERT

Bi-encoders encode queries and documents independently into dense vectors, enabling efficient first-stage retrieval through vector similarity search. A key challenge is selecting effective negative examples during training.

### 3. Learned Sparse Retrieval

- **Input**: Query or document
- **Output**: Sparse vector (mostly zeros)
- **Data Structure**: Inverted index
- **Encoder Types**: MLP (weighting only), MLM (weighting and expansion)

These models combine the efficiency of traditional sparse retrieval with the learning capabilities of neural networks. The sparse vectors can be very high-dimensional (vocabulary-sized) but contain mostly zeros.

## Generative Retrieval

### 1. Retrieval-Augmented Generation (RAG)

RAG combines retrieval models with generative models to produce more accurate, factual responses:

- **Core Challenges**:
    - Retrieve-then-read / Retrieve-then-generate (FiD/RAG): Using retrieved documents to inform generation
    - Adapting Retrieval to Language Models (ATLAS): Optimizing retrieval for LM consumption
    - Adapting Language Models to Retrieval (selfRAG): Training LMs to better utilize retrieved content

### 2. Generative IR

A more radical approach where:

- A single model replaces traditional retrieval components
- Takes a query as input and responds with a list of document IDs
- Major issue: How to design effective document ID schemes that are learnable

## Learning-to-Rank

Learning-to-Rank (LTR) optimizes document rankings using machine learning:

### 1. Point-wise Approaches

Treats ranking as a classification or regression problem for individual documents.

### 2. Pair-wise Approaches

Trains on document pairs, optimizing relative ordering:

- Uses metrics like cross-entropy between desired and predicted probabilities of document pairs

### 3. List-wise Approaches

Optimizes entire rankings:

- Acknowledges that not every document pair is equally important
- Uses metric-based or rank-based approximations of evaluation measures

### Dealing with Interaction Data

User interactions provide valuable training signals but are biased:

- **Click models** attempt to distinguish between position bias and actual relevance
- **Inverse Propensity Scoring (IPS)** debiases logged data for learning
- Bias can be estimated from online interactions or jointly with relevance

## Specialized Scenarios

### 1. Adversarial Retrieval

Dealing with malicious attempts to manipulate retrieval systems:

- **Threat models**: Understanding potential attack vectors
- **Gradient-guided attacks**: Using model gradients to craft adversarial queries/documents
- **Defenses**: Adversarial training, perplexity/norm thresholding

### 2. Conversational Retrieval

Retrieving information in multi-turn conversations:

- **Query rewriting**: Transforming context-dependent queries into standalone ones
- **Representation learning**: Specialized models like ConvDR/DISCO for conversational context
- **Mixed initiatives**: Systems that can both respond and ask clarifying questions

## Research Directions

The course concludes by highlighting active research areas:

10. Neural Information Retrieval advancements
11. Adaptive Retrieval Augmented Generation
12. Multi-modal and cross-lingual retrieval
13. Robust retrieval against adversarial attacks and bias
14. Interpretability and manipulations of retrieval systems
15. Agentic information access with user and knowledge agents

## Exam Information

From the slides, the exam will be:

- In-person on a provided computer
- You may bring a cheat sheet (requirements specified on Canvas)
- A simple calculator is permitted
- Location and time available on DataNose

Is there any specific area from this overview that you'd like me to explain in more detail for your exam preparation?
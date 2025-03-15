
## Introduction to Information Retrieval Systems

Information Retrieval (IR) has traditionally been built on a well-established pipeline architecture consisting of three major components:

1. **Indexing**: Building an index for each document in the corpus
2. **Retrieval**: Finding an initial set of candidate documents for a query
3. **Re-ranking**: Determining the relevance degree of each candidate

This pipelined approach has withstood the test of time and has been enhanced with advanced machine learning and deep learning techniques. However, this architecture has several inherent limitations.

### Limitations of Traditional IR Systems

1. **Effectiveness challenges**: The heterogeneous ranking components are difficult to optimize in an end-to-end manner toward a global objective. Each component is often optimized separately, leading to potential misalignment in the overall system goals.
2. **Efficiency problems**: Traditional systems require large document indices to search over the corpus, leading to significant memory consumption and computational overhead.

##### What if we replaced the pipelined architecture with a single consolidated model that efficiently and effectively encodes all of the information contained in the corpus?

## The Generative Retrieval Paradigm

![[gr_paradigm.png]]

- Directly generate the identifiers of information resources (e.g., docids) that are relevant to an information need (e.g., an input query) in an autoregressive fashion.
- Usually exploits a Seq2Seq encoder-decoder architecture to generate a ranked list of docids for an input query

### Two Families of Generative Retrieval
1. **Closed-book Generative Retrieval**: The language model is the only source of knowledge leveraged during generation.
    - Capturing document ids in the language models
    - Language models as retrieval agents via prompting
2. **Open-book Generative Retrieval**: The language model can draw on external memory prior to, during, and after generation.
    - Retrieval augmented generation of answers
    - Tool-augmented generation of answers

**Closed-book GR**: formulates IR as a sequence-to-sequence generation problem:
- **Input**: A sequence of query words
- **Output**: A sequence of document identifiers (docids)

### Advantages of Generative Retrieval
1. **Effectiveness**:
    - Knowledge of all documents in the corpus is encoded into model parameters, which can be optimized directly in an end-to-end manner toward a global objective.
    - versus Heterogeneous objectives of different components
2. **Efficiency**:
    - Main memory computation of GR is the storage of document identifiers and model parameters. 
    - Heavy retrieval process is replaced with a light generative process over the vocabulary of identifiers.
    - ![[gr_efficiency.png]]

### Basic operations in GR
- Indexing:
	- Memorize information about each document
	- docid $\leftrightarrow$ content of document
- Retrieval:
	- Input query $\rightarrow$ GR Model $\rightarrow$ ranked list of candidate docids (autoregressively generate docid strings)

## Model Training for Generative Retrieval

Training involves two basic operations:
1. **Indexing**: Memorizing information about each document by learning to associate content with corresponding docid.
    - original document $d \in D$ $\rightarrow$ docid $id \in I_{D}$ 
	- Seq2Seq
	- $$\mathcal{L}_{\text{indexing}}(D,I_{D};\theta)=-\sum_{d\in D}\log P(id | d;\theta)$$
	- $P(id | d;\theta)$: likelihood of each docid given document

2. **Retrieval**: Generating a ranked list of relevant docids for a query.
    - query $q \in Q$ $\rightarrow$ docids $id^q \in I_{Q}$
    - $$\mathcal{L}_{Retrieval}(Q, I_Q; \theta) = -\sum_{q \in Q} \sum_{id^q \in I_Q} \log P(id^q | q; \theta)$$
    - $P(id^q | q; \theta)$ likelihood of each relevant docid $id^q$ given the query $q$

These operations can be combined into a global objective for end-to-end optimization:
$$\mathcal{L}_{Global}(Q, D, I_D, I_Q;\theta) = \mathcal{L}_{Indexing}(D, I_D;\theta) + \mathcal{L}_{Retrieval}(Q, I_Q;\theta)$$
#### Example

![[gr_example.png]]

### Training Challenges
1. **Corpus memorization**: How to effectively and efficiently memorize the whole corpus
    - Rich information in documents
    - Limited labeled data
2. **Heterogeneous task learning**: How to learn different tasks well within a single model
    - Different data distributions
    - Different optimization objectives
3. **Dynamic document collection**: How to handle evolving document collections
    - Internal index in model parameters
    - High retraining costs for updates

## Model Inference for Generative Retrieval

Once a GR model is trained, it can generate candidate docids for a test query in an autoregressive fashion: 
$$w_t = GR_\theta(q_t, w_0, w_1, \ldots, w_{t-1})$$

The generation stops when decoding a special EOS token. 
The docids generated with the top-k highest likelihood (joint probability of generated tokens within a docid) form a ranking list.

### Inference Challenges
1. **Valid docid generation**: Ensuring the model generates valid identifiers
    - Limited docids vs. free generation
2. **Large-scale corpus organization**: Managing docids for millions of documents
    - Data structure considerations
3. **Ranked list generation**: Methods to produce ranked results
    - One-by-one generation using likelihood probabilities
    - One-time generation directly decoding a sequence of docids

## Document ID Design in Generative Retrieval

Document ID design is a critical component of generative retrieval systems. The choice of docid affects model performance, efficiency, and scalability.

- Pre-defined static docids vs. Learnable docids
#### Roadmap of pre-defined static docids

![[predefined_docid_roadmap.png]]
### Number-based DocIDs
1. **Unstructured atomic integers**:
    - An arbitrary (and possibly random) unique integer identifier
    - Simple output layer similar to standard language models
    - Limitations
        - need to learn embeddings for each individual docid
        - large softmax output space
2. **Naively structured strings**:
    - Treat arbitrary unique integers as tokenizable strings
    - Decoding formulation: Generating a docid string in a token-by-token manner
    - ![[naively_strictured_str.png | 400]]
    - Frees the limitation for corpus size that comes with unstructured atomic docid
    - Limitation
        - Lacks semantic structure
        - identifiers assigned arbitrarily
3. **Semantically structured strings**:
    - Properties:
        - Docids capture information about the semantics of the associated document
        - Structured to reduce search space after each decoding step
    - $\rightarrow$ Semantically similar documents share docid prefixes
    - Often implemented using a hierarchical clustering algorithm over document embeddings to induce a decimal tree ![[semantically_structured_str.png]]
    - Pros:
        - doc sementics incorporated in decoding
        - not limited by size of corpus

**Num-based docids Summary**
- Pros
	- easy to build
	- can maintain uniqueness
- Cons
	- unreadable numbers
	- challanging to interpret
### Word-based DocIDs
Query is keyword based: mapping to words should be intuitive

1. **Titles**: Using document titles as identifiers
    - Natural language representation
    - Challenge: Require exact generation (false pruning problem)
2. **URLs**: Using document URLs as identifiers
    - Similar challenges to titles
    - Also exact generation
3. **Important terms**:
    - A set of document terms with high importance scores
    - Any permutation of the term set docid will lead to the retrieval of the corresponding document. ![[tsgen_permutation.png]]
    - Mitigates the false pruning problem
    - Importance scores based on relevance to queries
    - ![[tsgen.png]] Check slides for full formulas.
    - For moderate-scale corpus like Natural Questions, specifying 12 terms is sufficient for uniqueness    

**Word-based docids Summary**
- Pros
	- semantically related to content of document
	- good interpretability
- Cons
	- rely on metadata or labeled data
	- may lead to duplication

### Single vs. Multiple DocIDs
- **Single docid**: One identifier per document
- **Multiple docids**: Multiple identifiers for each document
    - Provides more comprehensive representation
    - Assists model in gaining multifaceted understanding
    - Similar docids across documents can reflect similarity between documents
    - Shows better performance in experiments (e.g., on Natural Questions 320K dataset)

## Take Aways
1. **Paradigm Shift**: Moving from index-retrieve-rank to direct generation of document identifiers.
2. **Key Advantages**: End-to-end optimization, better efficiency, reduced dependence on external indices.
3. **Critical Challenges**: Document ID design, training approaches, and inference strategies.
4. **DocID Design Impact**: The choice of docids significantly affects retrieval accuracy, model efficiency, and scalability.
5. **Open Research Area**: Generative Retrieval remains an active and evolving area with many open questions.
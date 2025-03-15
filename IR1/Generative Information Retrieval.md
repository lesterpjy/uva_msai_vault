
## Introduction to Information Retrieval Systems

Information Retrieval (IR) has traditionally been built on a well-established pipeline architecture consisting of three major components:

1. **Indexing**: Building an index for each document in the corpus
2. **Retrieval**: Finding an initial set of candidate documents for a query
3. **Re-ranking**: Determining the relevance degree of each candidate

This pipelined approach has withstood the test of time and has been enhanced with advanced machine learning and deep learning techniques. However, this architecture has several inherent limitations.

### Limitations of Traditional IR Systems

4. **Effectiveness challenges**: The heterogeneous ranking components are difficult to optimize in an end-to-end manner toward a global objective. Each component is often optimized separately, leading to potential misalignment in the overall system goals.
    
5. **Efficiency problems**: Traditional systems require large document indices to search over the corpus, leading to significant memory consumption and computational overhead. For example, Dense Retrieval systems like GTR on the MS MARCO 300K dataset can consume 1430MB of memory and have online latency of 1.97 seconds.
    

## The Generative Retrieval Paradigm

Generative retrieval (GR) represents a paradigm shift by replacing the pipelined architecture with a single consolidated model that directly generates document identifiers. This approach formulates IR as a sequence-to-sequence generation problem:

- **Input**: A sequence of query words
- **Output**: A sequence of document identifiers (docids)

### Two Families of Generative Retrieval

6. **Closed-book Generative Retrieval**: The language model is the only source of knowledge leveraged during generation.
    
    - Capturing document ids in the language models
    - Language models as retrieval agents via prompting
7. **Open-book Generative Retrieval**: The language model can draw on external memory prior to, during, and after generation.
    
    - Retrieval augmented generation of answers
    - Tool-augmented generation of answers

### Advantages of Generative Retrieval

8. **Effectiveness**: Knowledge of all documents in the corpus is encoded into model parameters, which can be optimized directly in an end-to-end manner toward a global objective.
    
9. **Efficiency**: Main memory computation of GR is the storage of document identifiers and model parameters. Heavy retrieval process is replaced with a light generative process over the vocabulary of identifiers. For example, GenRet uses only 860MB of memory compared to GTR's 1430MB and reduces online latency from 1.97s to 0.16s.
    

## Neural IR Models: Discriminative vs. Generative

### Discriminative Approach

In discriminative models, we calculate: $$p(R = 1|q, d) \approx ... \approx \arg\max s(q⃗,d⃗)$$ (based on the probabilistic ranking principle)

### Generative Approach

In generative models, we calculate: $$p(q|d) \approx p(docID|q) = \arg\max p((I_1,...,I_n)|q)$$ (based on query likelihood)

## Document ID Design in Generative Retrieval

Document ID design is a critical component of generative retrieval systems. The choice of docid affects model performance, efficiency, and scalability.

### Number-based DocIDs

10. **Unstructured atomic integers**:
    
    - An arbitrary (and possibly random) unique integer identifier
    - Simple output layer similar to standard language models
    - Limitations include the need to learn embeddings for each individual docid and the large softmax output space
11. **Naively structured strings**:
    
    - Treat arbitrary unique integers as tokenizable strings
    - Decoding formulation: Generating a docid string in a token-by-token manner
    - Frees the limitation for corpus size that comes with unstructured atomic docid
    - Still lacks semantic structure
12. **Semantically structured strings**:
    
    - Docids capture information about the semantics of the associated document
    - Structured to reduce search space after each decoding step
    - Semantically similar documents share docid prefixes
    - Often implemented using a hierarchical clustering algorithm over document embeddings to induce a decimal tree
13. **Product quantization strings**:
    
    - Another approach to create structured numeric identifiers

### Word-based DocIDs

14. **Titles**: Using document titles as identifiers
    
    - Natural language representation
    - Challenge: Require exact generation (false pruning problem)
15. **URLs**: Using document URLs as identifiers
    
    - Similar challenges to titles
16. **Important terms**:
    
    - A set of document terms with high importance scores
    - Any permutation of the term set is a valid identification
    - Mitigates the false pruning problem
    - Importance scores based on relevance to queries
    - For moderate-scale corpus like Natural Questions, specifying 12 terms is sufficient for uniqueness
17. **Pseudo queries**: Generated queries that would retrieve the document
    

### Single vs. Multiple DocIDs

- **Single docid**: One identifier per document
- **Multiple docids**: Multiple identifiers for each document
    - Provides more comprehensive representation
    - Assists model in gaining multifaceted understanding
    - Similar docids across documents can reflect similarity between documents
    - Shows better performance in experiments (e.g., on Natural Questions 320K dataset)

## Model Training for Generative Retrieval

Training involves two basic operations:

18. **Indexing**: Memorizing information about each document by learning to associate content with corresponding docid.
    
    - Formulated as: $\mathcal{L}_{Indexing}(D, I_D; \theta) = -\sum_{d \in D} \log P(id | d; \theta)$
19. **Retrieval**: Generating a ranked list of relevant docids for a query.
    
    - Formulated as: $\mathcal{L}_{Retrieval}(Q, I_Q; \theta) = -\sum_{q \in Q} \sum_{id^q \in I_Q} \log P(id^q | q; \theta)$

These operations can be combined into a global objective: $\mathcal{L}_{Global}(Q, D, I_D, I_Q;\theta) = \mathcal{L}_{Indexing}(D, I_D;\theta) + \mathcal{L}_{Retrieval}(Q, I_Q;\theta)$

### Training Challenges

20. **Corpus memorization**: How to effectively and efficiently memorize the whole corpus
    
    - Rich information in documents
    - Limited labeled data
21. **Heterogeneous task learning**: How to learn different tasks well within a single model
    
    - Different data distributions
    - Different optimization objectives
22. **Dynamic document collection**: How to handle evolving document collections
    
    - Internal index in model parameters
    - High retraining costs for updates

## Model Inference for Generative Retrieval

Once a GR model is trained, it can generate candidate docids for a test query in an autoregressive fashion: $w_t = GR_\theta(q_t, w_0, w_1, \ldots, w_{t-1})$

The generation stops when decoding a special EOS token. The docids generated with the highest likelihood form a ranking list.

### Inference Challenges

23. **Valid docid generation**: Ensuring the model generates valid identifiers
    
    - Limited docids vs. free generation
24. **Large-scale corpus organization**: Managing docids for millions of documents
    
    - Data structure considerations
25. **Ranked list generation**: Methods to produce ranked results
    
    - One-by-one generation using likelihood probabilities
    - One-time generation directly decoding a sequence of docids

## Performance Comparisons

Research shows significant performance differences between docid design approaches:

26. **Number-based vs. Word-based**:
    
    - Semantically structured strings (27.4% Hits@1) significantly outperform unstructured atomic integers (20.7%) and naively structured strings (6.7%) on Natural Questions 320K
27. **Important terms vs. other identifiers**:
    
    - Important terms (TSGen, 69% Recall@1) outperform titles (59.1%) and URLs+titles (65.4%)
28. **Single vs. Multiple docids**:
    
    - Multiple docids (86.3-86.7% Recall@100) outperform single docids (65.5%)

## Take Aways

29. **Paradigm Shift**: Moving from index-retrieve-rank to direct generation of document identifiers.
    
30. **Key Advantages**: End-to-end optimization, better efficiency, reduced dependence on external indices.
    
31. **Critical Challenges**: Document ID design, training approaches, and inference strategies.
    
32. **DocID Design Impact**: The choice of docids significantly affects retrieval accuracy, model efficiency, and scalability.
    
33. **Open Research Area**: Generative Retrieval remains an active and evolving area with many open questions.
    

In conclusion, generative information retrieval offers a promising alternative to traditional approaches by consolidating the pipeline into a single model. The design of document identifiers plays a crucial role in the performance of these systems, with word-based and multiple docids showing particular promise.
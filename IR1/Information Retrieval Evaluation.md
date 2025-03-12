
## Introduction to IR Evaluation

Information Retrieval systems are designed to match information seekers with the information they need. When developing or improving search systems, we need rigorous methods to evaluate whether our changes actually improve user satisfaction. As the slides note, when we "change the search algorithm, how can we know whether we made the users happier?"

This fundamental question drives the field of IR evaluation. While user happiness is the ultimate goal, it's challenging to measure directly at scale. Instead, we use proxies like relevance of search results to estimate system effectiveness.

## The Cranfield Paradigm

The cornerstone of modern IR evaluation is the Cranfield Paradigm, developed by Cyril W. Cleverdon in the 1960s. This paradigm established a scientific approach to IR evaluation through controlled experiments.

The Cranfield approach uses experimental collections consisting of three key components:

1. **Documents (corpora)**: The collection of items to be searched
2. **Topics (queries)**: Test questions or information needs
3. **Relevance judgments**: Human assessments of whether documents are relevant to queries (also called qrels)

This approach ensures **comparability and repeatability** of experiments, allowing researchers to make scientific claims about the relative effectiveness of different IR systems. It follows the scientific method where we form hypotheses about improvements, test them experimentally, and evaluate the outcomes.

## Test Collections

A good test collection requires careful attention to each component:

### Test Documents

The document collection should be representative of the application domain in terms of number, size, and type. Different application areas (web search, legal search, etc.) require different types of documents.

### Test Queries

Test queries can come from:

- Example queries from potential users
- Query logs from existing systems

More queries are better for statistical validity. These should represent the types of information needs your system will encounter.

### Relevance Judgments

Relevance judgments can be obtained from:

- Users themselves (ideal but expensive)
- Independent judges
- Crowdsourcing platforms

More judgments are better, and for some applications, using graded relevance (beyond binary relevant/not-relevant) is important. Multiple judges can help ensure quality through inter-assessor agreement measures.

## The Depth-k Pooling Technique

Since it's impossible to judge the relevance of all documents for all queries (most collections are too large), the field developed the depth-k pooling method:

4. Take multiple search systems (by different research teams/participants)
5. Consider the top-k results from each system
6. Remove duplicates
7. Present this combined pool of documents to judges in random order

This produces a significant number of judgments efficiently, though it's still incomplete (some relevant documents may be missed). The quality of a pool depends on:

- Using a large enough k-value
- Having diverse enough systems contributing to the pool
- Using enough queries

This pooling approach is a key motivation for organizing large evaluation initiatives like TREC, CLEF, NTCIR, and FIRE, where multiple research teams participate with their systems.

## Evaluating System Effectiveness

### Precision and Recall

The most fundamental evaluation metrics in IR are precision and recall:

- **Precision**: The fraction of retrieved documents that are relevant
    
    - Precision = TP / (TP + FP)
    - Answers: "Of the results I got, how many were relevant?"
- **Recall**: The fraction of relevant documents that are retrieved
    
    - Recall = TP / (TP + FN)
    - Answers: "Of all possible relevant documents, how many did I get?"

Where:

- TP (True Positives): Relevant documents that were retrieved
- FP (False Positives): Non-relevant documents that were retrieved
- FN (False Negatives): Relevant documents that were not retrieved
- TN (True Negatives): Non-relevant documents that were not retrieved

There's typically a trade-off between precision and recall:

- Returning more documents increases recall (eventually to 100%)
- Being more selective often (but not always) increases precision

### Precision and Recall at Cutoffs

For ranked results, we can measure precision and recall at specific cutoff points:

- **P@k (Precision at k)**: The fraction of the top-k retrieved documents that are relevant
- **R@k (Recall at k)**: The fraction of all relevant documents found in the top-k results

### Precision-Recall Curves

We can generalize these metrics to create precision-recall curves:

8. For each rank position, compute precision and recall
9. Plot these as points on a graph (recall on x-axis)
10. Connect the points to create a curve

An "interpolated" precision-recall curve maintains the maximum precision value seen at each recall level, creating a step function that never increases as recall increases.

### Average Precision (AP)

Average Precision is a single-number metric that summarizes the entire precision-recall curve:

$$AP = \frac{\sum_{r=1}^{N} P(r) \times rel(r)}{|relevant\ documents|}$$

Where:

- P(r) is the precision at rank r
- rel(r) is 1 if the document at rank r is relevant, 0 otherwise

It can be interpreted as the average of the precision values at the positions of relevant documents.

### Mean Average Precision (MAP)

MAP is the mean of the average precision scores across a set of queries and is one of the most common overall evaluation metrics.

### Reciprocal Rank (RR)

For tasks where finding just one relevant document is sufficient (known-item search, navigational queries, factoid questions), Reciprocal Rank is useful:

$$RR = \frac{1}{rank\ of\ first\ relevant\ document}$$

This measures how quickly a system can return at least one relevant result.

## Beyond Binary Relevance

### Discounted Cumulative Gain (DCG)

DCG addresses limitations of binary relevance by:

11. Using graded relevance judgments (e.g., 0-4 scale)
12. Giving greater weight to highly relevant documents
13. Discounting the value of documents that appear lower in the ranking

The formula for DCG at position k is:

$$DCG@k = \sum_{r=1}^{k} \frac{2^{rel_r} - 1}{\log_2(1 + r)}$$

Where:

- $rel_r$ is the graded relevance of the document at position r
- The numerator (2^rel - 1) creates a non-linear gain for higher relevance
- The denominator (log2(1+r)) creates a discount for lower ranks

### Normalized DCG (nDCG)

To allow comparisons across queries with different numbers of relevant documents, we normalize DCG against the "ideal" DCG (IDCG) that would be achieved by the perfect ranking:

$$nDCG@k = \frac{DCG@k}{IDCG@k}$$

This gives a value between 0 and 1, where 1 represents a perfect ranking.

## User Models in IR Evaluation

User models move beyond simple metrics by trying to model how users actually interact with search results. They typically include:

14. **A browsing model** - describes how a user interacts with results
15. **A model of document utility** - describes how a user derives value from relevant documents
16. **A utility accumulation model** - describes how utility accumulates during browsing

### Position-based Models

These models assume the probability of examining a document depends primarily on its position in the ranked list. Users start at the top and may abandon their search with some probability at each position.

**Rank-Biased Precision (RBP)** is an example, calculated as:

$$RBP = \sum_{r=1}^{\infty} rel_r \theta^{r-1} (1 - \theta)$$

Where:

- $rel_r$ is the relevance of the document at rank r (typically binary)
- θ is the persistence parameter (probability of continuing to the next result)
- (1-θ) is the probability of stopping at the current result

### Cascade-based Models

These models assume users examine documents sequentially until they find a satisfactory one. The probability of satisfaction depends on the document's relevance level.

**Expected Reciprocal Rank (ERR)** is based on this model:

$$ERR = \sum_{r=1}^{n} \left(\prod_{i=1}^{r-1}(1-\vartheta_i)\vartheta_r\right)\frac{1}{r}$$

Where:

- $\vartheta_i$ is the probability of being satisfied with document i
- The product term represents the probability of not being satisfied with any earlier document
- 1/r represents the reciprocal rank (effort required)

## Emerging Evaluation Methods

### LLM-as-a-judge

Recent research has explored using Large Language Models (LLMs) to assess relevance instead of human judges. Studies show LLMs can:

- Achieve correlation with human judgments comparable to inter-annotator agreement
- Scale more efficiently than human annotation
- Maintain consistency across evaluations

### Nugget-based Evaluation

For more complex retrieval tasks like question answering and RAG (Retrieval-Augmented Generation):

17. Create a list of "information nuggets" that represent atomic facts about a topic
18. Identify which nuggets are vital for a good answer
19. Assess system outputs based on which nuggets they contain

Scoring approaches include:

- **All score**: Average of scores for all nuggets in an answer
- Various weighting schemes for vital vs. non-vital nuggets

These systems can be automated using LLMs to create nuggets and assess their presence in system outputs.

## Practical Considerations in IR Evaluation

### Inter-assessor Agreement

When multiple judges assess relevance, we measure agreement using Cohen's kappa coefficient:

$$\kappa = \frac{P(A) - P(E)}{1 - P(E)}$$

Where:

- P(A) is the observed agreement
- P(E) is the expected chance agreement

Values above 0.8 indicate high agreement, 0.67-0.8 acceptable agreement, and below 0.67 low agreement.

### Large-scale Evaluation Initiatives

Several organizations run ongoing IR evaluation campaigns:

- TREC (Text REtrieval Conference) - USA, since 1992
- NTCIR (NII Testbeds and Community for Information Access Research) - Japan, since 1999
- CLEF (Conference and Labs of the Evaluation Forum) - Europe, since 2000
- FIRE (Forum for Information Retrieval Evaluation) - India, since 2008

These initiatives create valuable test collections and benchmark tasks for the research community.

## Summary of Key Concepts

20. **Purpose of IR Evaluation**: Measure how effectively IR systems match users with relevant information.
21. **Cranfield Paradigm**: Scientific framework using test collections for comparable and repeatable experiments.
22. **Evaluation Metrics**: Precision, Recall, MAP, nDCG, RR - each capturing different aspects of retrieval quality.
23. **User Models**: Move beyond simple metrics to model how users interact with search results.
24. **Advanced Techniques**: Graded relevance, LLM-as-judge, and nugget-based evaluation address complex information needs.

Understanding these evaluation approaches is essential for systematically improving IR systems and making scientific claims about their effectiveness.

## Introduction to IR Evaluation

Information Retrieval systems are designed to match information seekers with the information they need. When developing or improving search systems, we need rigorous methods to evaluate whether our changes actually improve user satisfaction. As the slides note, when we "change the search algorithm, how can we know whether we made the users happier?"

This fundamental question drives the field of IR evaluation. While user happiness is the ultimate goal, it's challenging to measure directly at scale. Instead, we use proxies like relevance of search results to estimate system effectiveness.

### Efficiency of an IR System
- Indexing time
- Indexing space
- Index storage
- Throughput
- Query latency
- Carbon Footprint

## The Cranfield Paradigm

The cornerstone of modern IR evaluation is the Cranfield Paradigm, developed by Cyril W. Cleverdon in the 1960s. This paradigm established a scientific approach to IR evaluation through controlled experiments.

The Cranfield approach uses experimental collections consisting of three key components:

1. **Documents (corpora)**: The collection of items to be searched
2. **Topics (queries)**: Test questions or information needs
3. **Relevance judgments**: Human assessments of whether documents are relevant to queries (also called qrels)

This approach ensures **comparability and repeatability** of experiments, allowing researchers to make scientific claims about the relative effectiveness of different IR systems. It follows the scientific method where we form hypotheses about improvements, test them experimentally, and evaluate the outcomes.

## Test Collections

![[test_collections.png | 450]]

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

### Issue with manual evaluation
- not scalable
- not re-usable

## The Depth-k Pooling Technique

Since it's impossible to judge the relevance of all documents for all queries (most collections are too large), the field developed the depth-k pooling method:
1. Take multiple search systems (by different research teams/participants)
2. Consider the top-k results from each system
3. Remove duplicates
4. Present this combined pool of documents to judges in random order

This produces a significant number of judgments efficiently, though it's still incomplete (some relevant documents may be missed). The quality of a pool depends on:
- Using a large enough k-value
- Having diverse enough systems contributing to the pool
- Using enough queries

The objective is not to allow for computing the “exact” value of
an evaluation measure but rather to comparatively assess
systems and detect significant differences in a robust way

This pooling approach is a key motivation for organizing large evaluation initiatives like TREC, CLEF, NTCIR, and FIRE, where multiple research teams participate with their systems.

## Test Collection Pooling Quality

### Leave-One-Out Tests for Pool Reusability

Leave-one-out tests are a critical methodology for evaluating how robust and reusable a document pool is. This is essential in IR research because we need to know whether a test collection can fairly evaluate new systems that weren't part of the original pooling process.

The methodology works as follows:
1. One system (or group of systems) is removed from the contributing systems that created the original document pool.
2. All systems (including the removed one) are evaluated using both the original complete pool and the newly created reduced pool that lacks the contributions of the removed system.
3. The results from both evaluations are compared using two key metrics:
    - Kendall's τ (tau) correlation coefficient between system rankings
    - The maximum drop in ranking observed for any system

Kendall's τ measures the similarity between two rankings by counting concordant and discordant pairs. Values closer to 1 indicate high similarity, while values closer to 0 indicate dissimilarity. In IR evaluation, a τ value above 0.9 is typically considered to indicate that the rankings are effectively equivalent.

The maximum drop in ranking shows the worst-case impact on any individual system. A small maximum drop suggests that the pool is robust to the removal of a contributing system.

This methodology helps researchers understand whether a test collection is biased toward the systems that contributed to its creation. If removing a system's contributions significantly alters rankings, it suggests the pool may not be fair for evaluating new systems that didn't contribute to the pool.

## Inter-Assessor Agreement
Metrics for measuring agreement between multiple human assessors, which is crucial for establishing the reliability of relevance judgments.

### Cohen's Kappa Coefficient
Cohen's kappa (κ) is a statistical measure that accounts for agreement occurring by chance:

κ = [P(A) - P(E)] / [1 - P(E)]

Where:
- P(A) is the observed agreement proportion
- P(E) is the expected agreement proportion by chance

The interpretation guidelines for κ values are:

- > 0.8: High agreement
- 0.67-0.8: Acceptable agreement
- <0.67: Low agreement

When dealing with more than two assessors, we typically calculate the average of all pair-wise kappa coefficients to get an overall measure of agreement.

### Example Calculation

![[cohens_kappa.png]]
1. **Step 1**: Calculate observed agreement P(A)
    - Documents where both raters agree: 40 (both say relevant) + 30 (both say not relevant) = 70
    - Total documents: 100
    - P(A) = 70/100 = 0.7
2. **Step 2**: Calculate expected agreement P(E) using marginal probabilities
    - Rater 1: P(Relevant) = 50/100 = 0.5, P(Not Relevant) = 50/100 = 0.5
    - Rater 2: P(Relevant) = 60/100 = 0.6, P(Not Relevant) = 40/100 = 0.4
    - Expected agreement: P(E) = (0.5 × 0.6) + (0.5 × 0.4) = 0.3 + 0.2 = 0.5
3. **Step 3**: Calculate kappa
    
    - κ = (0.7 - 0.5) / (1 - 0.5) = 0.2 / 0.5 = 0.4

The result κ = 0.4 indicates "fair agreement" between the raters, which is actually below the threshold for what would be considered acceptable agreement (0.67) in rigorous IR evaluation settings.

## Importance of These Concepts in IR Evaluation

These concepts are fundamental to understanding the reliability and validity of IR evaluation:

1. **Pool quality** affects whether we can fairly evaluate new systems using existing test collections. If pools aren't reusable, we may incorrectly penalize new systems for finding relevant documents that weren't in the original pool.
2. **Assessor agreement** affects the ground truth we use for evaluation. Low agreement suggests high subjectivity in relevance judgments, which means system performance metrics may not be as reliable as we'd hope.

Both concepts address a fundamental challenge in IR evaluation: the creation of reliable, reusable test collections that can fairly evaluate systems, even those that weren't part of the original evaluation.

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
1. For each rank position, compute precision and recall
2. Plot these as points on a graph (recall on x-axis)
3. Connect the points to create a curve

An "interpolated" precision-recall curve maintains the maximum precision value seen at each recall level, creating a step function that never increases as recall increases.

### Average Precision (AP)

Average Precision is a single-number metric that summarizes the entire precision-recall curve:
$$AP = \frac{\sum_{r=1}^{N} P(r) \times rel(r)}{|relevant\ documents|}$$

Where:
- P(r) is the precision at rank r
- rel(r) is 1 if the document at rank r is relevant, 0 otherwise
It can be interpreted as the average of the precision values at the positions of relevant documents.

![[average_precision.png | 350]]

### Mean Average Precision (MAP)
MAP is the mean of the average precision scores across a set of queries and is one of the most common overall evaluation metrics.

### Reciprocal Rank (RR)

For tasks where finding just one relevant document is sufficient (known-item search, navigational queries, factoid questions), Reciprocal Rank is useful:

$$RR = \frac{1}{rank\ of\ first\ relevant\ document}$$

This measures how quickly a system can return at least one relevant result.

## Beyond Binary Relevance

### Discounted Cumulative Gain (DCG)

DCG addresses limitations of binary relevance by:
1. Using graded relevance judgments (e.g., 0-4 scale)
2. Giving greater weight to highly relevant documents
3. Discounting the value of documents that appear lower in the ranking

The formula for DCG at position k is:
$$DCG@k = \sum_{r=1}^{k} \frac{2^{rel_r} - 1}{\log_2(1 + r)}$$
Where:
- $rel_r$ is the graded relevance of the document at position r
- The numerator ($2^{rel_{r}} - 1$) creates a non-linear gain for higher relevance
- The denominator ($log_{2}(1+r)$) creates a discount for lower ranks

### Normalized DCG (nDCG)
To allow comparisons across queries with different numbers of relevant documents, we normalize DCG against the "ideal" DCG (IDCG) that would be achieved by the perfect ranking:

$$nDCG@k = \frac{DCG@k}{IDCG@k}$$

This gives a value between 0 and 1, where 1 represents a perfect ranking.

## User Models in IR Evaluation

User models move beyond simple metrics by trying to model how users actually interact with search results. They typically include:
1. **A browsing model** - describes how a user interacts with results
2. **A model of document utility** - describes how a user derives value from relevant documents
3. **A utility accumulation model** - describes how utility accumulates during browsing

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
1. Create a list of "information nuggets" that represent atomic facts about a topic
2. Identify which nuggets are vital for a good answer
3. Assess system outputs based on which nuggets they contain

Scoring approaches include:
- **All score**: Average of scores for all nuggets in an answer
- Various weighting schemes for vital vs. non-vital nuggets

These systems can be automated using LLMs to create nuggets and assess their presence in system outputs.

## Summary of Key Concepts

4. **Purpose of IR Evaluation**: Measure how effectively IR systems match users with relevant information.
5. **Cranfield Paradigm**: Scientific framework using test collections for comparable and repeatable experiments.
6. **Evaluation Metrics**: Precision, Recall, MAP, nDCG, RR - each capturing different aspects of retrieval quality.
7. **User Models**: Move beyond simple metrics to model how users interact with search results.
8. **Advanced Techniques**: Graded relevance, LLM-as-judge, and nugget-based evaluation address complex information needs.
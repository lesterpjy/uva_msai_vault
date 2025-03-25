## Introduction to Learning to Rank

Learning to Rank (LTR) is a crucial area in information retrieval that focuses on developing machine learning algorithms to automatically construct optimal rankings of items. The lecture by Harrie Oosterhuis from Radboud University provides a thorough exploration of this field.

### Why Learning to Rank Matters

Ranking systems serve as the backbone of both search engines and recommendation systems. They enable users to navigate and retrieve information from large collections of items (documents, products, videos, etc.). The quality of rankings directly impacts:

- User satisfaction
- User engagement
- Overall system performance

The primary goal of ranking is typically to place the most relevant items at the top of the results, while also considering other objectives such as diversity, novelty, and fairness.

### The Ranking Pipeline

The typical ranking pipeline consists of four main stages:

1. **Documents** → **Features**: Feature extraction or embedding lookup
2. **Features** → **Scores**: Scoring function
3. **Scores** → **Ranking**: Sorting
4. **Ranking** → **Search Results Page**: Presentation

This process transforms raw documents into features, which are then scored, sorted, and finally presented to the user.

## Ranking Models and Features

### Traditional Ranking Models

Several traditional ranking models exist in information retrieval:

1. Vector Space Model / TF-IDF: Represents documents and queries as vectors in a high-dimensional space
2. BM25: A probabilistic ranking function that extends TF-IDF with additional parameters
3. Language models: Statistical models that capture how language is used in documents

### Features for Ranking

Real-world ranking systems utilize a wide array of features that can be categorized into:

#### Document-Query Features:

- Vector space model scores
- TF-IDF scores
- BM25 scores
- Language model scores
- Word overlap
- Neural language model scores
- Language match
- Entity match

#### Document Features:

- Webpage popularity
- Spam detection
- Page length
- Overall quality
- PageRank
- URL type

#### Query Features:

- Query type (Navigational/Informational)
- Query length
- Entities in query
- Aggregated IDF of query terms
- User information

Modern search engines use hundreds of features:

- Bing: 136+ features
- Yahoo: 700+ features
- Istella: 220+ features

With so many features, hand-crafted ranking models become impractical, making machine learning essential.

## The Core Challenge of Learning to Rank

The fundamental challenge in learning to rank is that the sorting function (which determines the rank of a document) is difficult to optimize directly. Ranking metrics as functions of document scores have problematic mathematical properties:

- Non-smooth
- Mostly flat
- Discontinuous

These properties make standard optimization techniques ineffective. Each family of LTR methods has developed its own approach to overcome this challenge.

## Learning to Rank Approaches

### 1. Pointwise Learning to Rank

Pointwise approaches treat LTR as a classification or regression problem, where each document-query pair is considered independently.

The typical loss function is Mean Squared Error (MSE):

$$\mathcal{L}_{Squared} = \sum_{q,d} ||y_{q,d} - f(\vec{x}_{q,d})||^2$$

Where:

- $y_{q,d}$ is the relevance label
- $\vec{x}_{q,d}$ are the features of the document-query pair
- $f$ is the scoring function to optimize

**Fundamental Limitation**: Pointwise losses do not directly optimize ranking quality. A lower loss does not necessarily mean a better ranking. For example, if a model scores a relevant document at 0.1 and irrelevant documents at 0.2, the ranking would be poor (irrelevant documents ranked higher) despite potentially having a decent MSE.

### 2. Pairwise Learning to Rank

Pairwise approaches focus on correctly ordering pairs of documents. They aim to minimize the number of pair-inversions in rankings (pairs where a less relevant document is ranked higher than a more relevant one).

The typical pairwise loss has the form:

$$\mathcal{L}_{pairwise} = \sum_{y_i > y_j} \phi(s_i - s_j)$$

Where:

- $s_i = f(\vec{x}_i)$ is the predicted score for document i
- $\phi$ is a function that penalizes incorrect orderings

**RankNet** is a prominent pairwise method that uses cross-entropy loss:

$$\mathcal{L}_{RankNet} = \sum_{y_i > y_j} \log(1 + e^{-\gamma(s_i-s_j)})$$

This can be derived from the predicted probability that document i should be ranked higher than document j:

$$P_{ij} = P(d_i \succ d_j) = \frac{e^{\gamma \cdot s_i}}{e^{\gamma \cdot s_i} + e^{\gamma \cdot s_j}} = \frac{1}{1 + e^{-\gamma(s_i - s_j)}}$$

**Limitation**: While better than pointwise approaches, pairwise methods treat all document pairs as equally important. In reality, correctly ordering documents at the top of the ranking is much more important than ordering documents further down.

### 3. Listwise Learning to Rank

Listwise methods consider the entire ranking in their loss function. They evaluate and optimize the quality of the complete ranking rather than individual scores or pairs.

**ListNet** assumes there is an ideal ranking and maximizes its probability:

$$R^* = [s_1, s_2, s_3, \cdots]$$

$$\mathcal{L}_{Listwise}(R^*) = -\sum_{i=1}^{K} \log\left(\frac{e^{s_i}}{\sum_{j=i}^{|R^*|} e^{s_j}}\right)$$

The first listwise methods optimized a Soft-Max Plackett-Luce distribution, where ranking is done by sampling without replacement:

$$P(R = [d_2, d_1, d_4, d_3]) = \frac{e^{s_2}}{e^{s_1} + e^{s_2} + e^{s_3} + e^{s_4}} \cdot \frac{e^{s_1}}{e^{s_1} + e^{s_3} + e^{s_4}} \cdot \frac{e^{s_4}}{e^{s_3} + e^{s_4}} \cdot \frac{e^{s_3}}{e^{s_3}}$$

**Limitation**: Many listwise methods do not directly optimize actual ranking metrics like NDCG or MAP. Instead, they optimize probabilistic models that may not perfectly align with the desired ranking quality metrics.

### 4. Metric-Based Learning to Rank

Metric-based approaches aim to directly optimize specific ranking metrics (e.g., NDCG, MAP). Since nearly all ranking metrics are listwise, metric-based approaches are a subset of listwise approaches.

**LambdaRank** is the most influential metric-based method:

$$\mathcal{L}_{LambdaRank}(R) = \sum_{d_i \succ d_j} \log(1 + e^{-\gamma(s_i-s_j)}) \cdot |\Delta \text{DCG}_{R,i,j}|$$

LambdaRank takes the pairwise loss of RankNet and weights each pair according to how much swapping it affects the metric (e.g., the change in DCG). The gradients treat these weights as constants.

Another approach, **ApproxDCG**, approximates the rank function through a sigmoid upper bound:

$$\text{rank}(d_i) = 1 + \sum_{j:j\neq i} \mathbb{1}[s_j > s_i] \approx 1 + \sum_{j:i \neq j} \frac{1}{1 + \exp(s_j - s_i)} = \overline{\text{rank}}(d_i)$$

This approximated rank is then plugged into the ranking metric, such as DCG:

$$L = -\sum_d \frac{\text{label}(d)}{\log_2(1 + \overline{\text{rank}}(d))}$$

This approach assumes the metric rank weights decrease monotonically and smoothly, which isn't true for all metrics, especially fairness metrics.

### 5. Stochastic Learning to Rank

Inspired by reinforcement learning, stochastic approaches optimize the expected utility of a probabilistic ranking model:

$$\mathbb{E}_{R|\pi}[\text{DCG}(R)] = \sum_R P(R | q, \pi)\text{DCG}(R) \approx \frac{1}{N}\sum_{i=1}^N \text{DCG}(R_i)$$

Since the number of possible rankings is enormous, this expectation is approximated through sampling. The REINFORCE log-derivative trick is used to approximate the gradient from samples:

$$\nabla_\pi \mathbb{E}_{R|\pi}[\text{DCG}(R)] \approx \frac{1}{N}\sum_{R,i=1}^N \text{DCG}(R_i) \nabla_\pi \log P(R_i | q, \pi)$$

More efficient algorithms like PL-Rank have been introduced to make this approach more practical.

## Perspectives on Categorizing LTR Methods

While LTR methods are traditionally categorized into pointwise, pairwise, and listwise approaches, this categorization has limitations:

- There is no clear consistently-used definition of each category
- Some methods are categorized differently by different researchers
- These categories are not mutually exclusive

A more useful perspective is to think in terms of properties rather than strict categories:

4. **Pointwise properties**:
    
    - Applies standard classification or regression
    - Not truly optimizing for ranking
5. **Pairwise properties**:
    
    - Focuses on document pair inversions
    - Ignores the complete ranking
6. **Listwise properties**:
    
    - Considers the entire ranking
    - May not directly optimize ranking metrics
7. **Metric-based properties**:
    
    - Explicitly optimizes ranking metrics
    - Often uses differentiable approximations
8. **Stochastic properties**:
    
    - Optimizes probabilistic ranking models
    - Related to reinforcement learning

## Conclusion and Future Directions

Learning to rank combines large numbers of features to create optimal rankings. The core challenge is that ranking metrics are not directly optimizable using standard gradient-based methods. Each LTR approach provides a different heuristic to approximate the gradient in a usable manner.

There is no single "best" LTR method - different approaches have different strengths and are suitable for different scenarios. It's more productive to think about the specific properties needed for a particular application rather than rigidly adhering to categories.

Future directions in this field include learning from user interactions rather than explicit relevance labels, which would be covered in a future lecture. This direction is particularly promising as it aligns with how real users interact with ranking systems and can provide more natural, implicit feedback.

Learning to rank remains a vital area of research and application in information retrieval, search engines, recommendation systems, and is even being applied in areas like reinforcement learning from human feedback (RLHF) for large language models.
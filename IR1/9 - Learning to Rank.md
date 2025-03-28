## Introduction to Learning to Rank

Learning to Rank (LTR) is a crucial area in information retrieval that focuses on developing machine learning algorithms to automatically construct optimal rankings of items.

### Why Learning to Rank Matters

Ranking systems serve as the backbone of both search engines and recommendation systems. They enable users to navigate and retrieve information from large collections of items (documents, products, videos, etc.). The quality of rankings directly impacts:

- <span style="color:rgb(172, 96, 230)">User satisfaction</span>
- <span style="color:rgb(172, 96, 230)">User engagement</span>
- <span style="color:rgb(172, 96, 230)">Overall system performance</span>

The primary goal of ranking is typically to <span style="color:rgb(172, 96, 230)">place the most relevant items at the top</span> of the results, while also considering <span style="color:rgb(172, 96, 230)">other objectives such as diversity, novelty, and fairness</span>.

### The Ranking Pipeline

The typical ranking pipeline consists of four main stages:

1. **Documents** → **Features**: Feature extraction or embedding lookup
2. **Features** → **Scores**: Scoring function
3. **Scores** → **Ranking**: Sorting
4. **Ranking** → **Search Results Page**: Presentation

This process transforms raw documents into features, which are then scored, sorted, and finally presented to the user.

## Ranking Models and Features

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

### Setup and Goal

- Setup
	- IR collection (gathered with Cranfield paradigm) $\rightarrow$ features over this dataset
	- Feature representation of doc-query pairs
	- Labels indicating relevance of doc-query pairs
- Goal
	- Function that scores document
	- Gives best ranking when sorted according to scores

## The Core Challenge of Learning to Rank

The fundamental challenge in learning to rank is that the <span style="color:rgb(172, 96, 230)">sorting function (which determines the rank of a document) is difficult to optimize directly</span>. Ranking metrics as functions of document scores have problematic mathematical properties:

- <span style="color:rgb(172, 96, 230)">Non-smooth</span>
- <span style="color:rgb(172, 96, 230)">Mostly flat</span>
- <span style="color:rgb(172, 96, 230)">Discontinuous</span>

These properties make standard optimization techniques ineffective. Each family of LTR methods has developed its own approach to overcome this challenge.

## Learning to Rank Approaches

### 1. Pointwise Learning to Rank

Pointwise <span style="color:rgb(172, 96, 230)">approaches treat LTR as a classification or regression problem</span>, where each document-query pair is considered independently.

The typical loss function is Mean Squared Error (MSE):
$$\mathcal{L}_{Squared} = \sum_{q,d} ||y_{q,d} - f(\vec{x}_{q,d})||^2$$
where:

- $y_{q,d}$ is the relevance label
- $\vec{x}_{q,d}$ are the features of the document-query pair
- $f$ <span style="color:rgb(172, 96, 230)">is the scoring function to optimize</span>

**Fundamental Limitation**: 
- <span style="color:rgb(172, 96, 230)">Pointwise losses do not directly optimize ranking quality. A lower loss does not necessarily mean a better ranking</span>. 
- For example, if a model scores a relevant document at 0.1 and irrelevant documents at 0.2, the ranking would be poor (irrelevant documents ranked higher) despite potentially having a decent MSE.
- Ranking is not a regression or classification problem.

### 2. Pairwise Learning to Rank

Pairwise approaches <span style="color:rgb(172, 96, 230)">focus on correctly ordering pairs of documents</span>. They aim to <span style="color:rgb(172, 96, 230)">minimize the number of pair-inversions in rankings</span> (pairs where a less relevant document is ranked higher than a more relevant one).

The typical pairwise loss has the form:
$$\mathcal{L}_{pairwise} = \sum_{y_i > y_j} \phi(s_i - s_j)$$
where:

- $s_i = f(\vec{x}_i)$ is the predicted score for document i
- $\phi$ is a function that penalizes incorrect orderings

<span style="color:rgb(172, 96, 230)">RankNet is a prominent pairwise method that uses cross-entropy loss</span>:

$$\mathcal{L}_{RankNet} = \sum_{y_i > y_j} \log(1 + e^{-\gamma(s_i-s_j)})$$

This can be derived from the predicted probability that <span style="color:rgb(172, 96, 230)">document i should be ranked higher than document j</span>:

$$P_{ij} = P(d_i \succ d_j) = \frac{e^{\gamma \cdot s_i}}{e^{\gamma \cdot s_i} + e^{\gamma \cdot s_j}} = \frac{1}{1 + e^{-\gamma(s_i - s_j)}}$$
where the desired probabilities are $\bar{P}_{ij}=1$ and $\bar{P}_{ji} = 0$

**Limitation**: While better than pointwise approaches, pairwise methods <span style="color:rgb(172, 96, 230)">treat all document pairs as equally important</span>. In reality, correctly ordering documents at the top of the ranking is much more important than ordering documents further down. Ex. <span style="color:rgb(172, 96, 230)">order of top-5 more important than after position 10.</span>

Remain very popular because:
- <span style="color:rgb(172, 96, 230)">Low cost to compute gradient update</span>, only 2 examples have to be in memory
- For some problems, <span style="color:rgb(172, 96, 230)">relative labels easier</span> to obtain than absolute labels
$\rightarrow$ pairwise LTR still <span style="color:rgb(172, 96, 230)">important in neural IR and RLHF</span>

### 3. Listwise Learning to Rank

Listwise methods consider the entire ranking in their loss function. They <span style="color:rgb(172, 96, 230)">evaluate and optimize the quality of the complete ranking</span> rather than individual scores or pairs.

**ListNet** assumes there is an ideal ranking and maximizes its probability:

$$R^* = [s_1, s_2, s_3, \cdots]$$

$$\mathcal{L}_{Listwise}(R^*) = -\sum_{i=1}^{K} \log\left(\frac{e^{s_i}}{\sum_{j=i}^{|R^*|} e^{s_j}}\right)$$

The first listwise methods optimized a <span style="color:rgb(172, 96, 230)">Soft-Max Plackett-Luce distribution, where ranking is done by sampling without replacement</span>:

$$P(R = [d_2, d_1, d_4, d_3]) = \frac{e^{s_2}}{e^{s_1} + e^{s_2} + e^{s_3} + e^{s_4}} \cdot \frac{e^{s_1}}{e^{s_1} + e^{s_3} + e^{s_4}} \cdot \frac{e^{s_4}}{e^{s_3} + e^{s_4}} \cdot \frac{e^{s_3}}{e^{s_3}}$$

**Limitation**:
- No actual ranking metric
- optimizing a probabilistic model instead of a deterministic model

### 4. Metric-Based Learning to Rank

Metric-based approaches aim to <span style="color:rgb(172, 96, 230)">directly optimize specific ranking metrics</span> (e.g., NDCG, MAP). Since nearly all ranking metrics are listwise, <span style="color:rgb(172, 96, 230)">metric-based approaches are a subset of listwise approaches</span>.

<span style="color:rgb(172, 96, 230)">LambdaRank</span> is the most influential metric-based method:

$$\mathcal{L}_{LambdaRank}(R) = \sum_{d_i \succ d_j} \log(1 + e^{-\gamma(s_i-s_j)}) \cdot |\Delta \text{DCG}_{R,i,j}|$$

LambdaRank takes the <span style="color:rgb(172, 96, 230)">pairwise loss of RankNet and weights each pair according to how much swapping it affects the metric</span> (e.g., the change in DCG). The <span style="color:rgb(172, 96, 230)">gradients treat these weights as constants</span>.

Another approach, <span style="color:rgb(172, 96, 230)">ApproxDCG, approximates the rank function through a sigmoid upper bound</span>:

$$\text{rank}(d_i) = 1 + \sum_{j:j\neq i} \mathbb{1}[s_j > s_i] \approx 1 + \sum_{j:i \neq j} \frac{1}{1 + \exp(s_j - s_i)} = \overline{\text{rank}}(d_i)$$

This <span style="color:rgb(172, 96, 230)">approximated rank is then plugged into the ranking metric</span>, such as DCG:

$$L = -\sum_d \frac{\text{label}(d)}{\log_2(1 + \overline{\text{rank}}(d))}$$


- Assumes the metric rank weights decrease monotonically and smoothly
- Not true for all metrics, especially fairness metrics.

### 5. Stochastic Learning to Rank

Inspired by reinforcement learning, stochastic approaches <span style="color:rgb(172, 96, 230)">optimize the expected utility of a probabilistic ranking model</span>:

$$\mathbb{E}_{R|\pi}[\text{DCG}(R)] = \sum_R P(R | q, \pi)\text{DCG}(R) \approx \frac{1}{N}\sum_{i=1}^N \text{DCG}(R_i)$$

Since the number of possible rankings is enormous, this expectation is approximated through sampling. The REINFORCE <span style="color:rgb(172, 96, 230)">log-derivative trick is used to approximate the gradient from samples</span>:

$$\nabla_\pi \mathbb{E}_{R|\pi}[\text{DCG}(R)] \approx \frac{1}{N}\sum_{R,i=1}^N \text{DCG}(R_i) \nabla_\pi \log P(R_i | q, \pi)$$

More efficient algorithms like <span style="color:rgb(172, 96, 230)">PL-Rank have been introduced to make this approach more practical</span>.

## Perspectives on Categorizing LTR Methods

While LTR methods are traditionally categorized into pointwise, pairwise, and listwise approaches, this categorization has limitations:

- There is no clear consistently-used definition of each category
- Some methods are categorized differently by different researchers
- These categories are not mutually exclusive

A more useful perspective is to <span style="color:rgb(172, 96, 230)">think in terms of properties rather than strict categories</span>:

1. **Pointwise properties**:
    
    - Applies standard classification or regression
    - Not truly optimizing for ranking
2. **Pairwise properties**:
    
    - Focuses on document pair inversions
    - Ignores the complete ranking
3. **Listwise properties**:
    
    - Considers the entire ranking
    - May not directly optimize ranking metrics
4. **Metric-based properties**:
    
    - Explicitly optimizes ranking metrics
    - Often uses differentiable approximations
5. **Stochastic properties**:
    
    - Optimizes probabilistic ranking models
    - Related to reinforcement learning

## Conclusion
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Learning to rank combines large numbers of features to create rankings.
- The derivative of ranking metrics are not usable for optimization.
- Every LTR method heuristically approximates the gradient in a usable manner.
- No one LTR method to rule them all.
- Think about LTR properties instead of categories


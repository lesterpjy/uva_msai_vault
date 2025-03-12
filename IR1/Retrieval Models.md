
## Introduction to Retrieval Models

Retrieval models provide the mathematical foundation for defining how queries match with documents in information retrieval systems. They offer:

1. A formal framework for query-document matching
2. Clear explanation of underlying assumptions
3. The basis for many ranking algorithms

The progression in retrieval models has directly led to improvements in ranking effectiveness over time. Understanding these models is crucial for developing effective search systems.

## Categories of Retrieval Models

Retrieval models can be broadly classified into:

1. Vector space model
    - TF-IDF
    - distributed representations
    - neural models
2. Probabilistic models
    - Probabilistic models (BM25)
    - Language models
    - Topic models
3. Learning to rank models

## Vector Space Model: Documents and Queries as Vectors

In the vector space model, both documents and queries are represented as vectors in a high-dimensional space, where each dimension corresponds to a term in the vocabulary.

### Term Frequency Representation

![[doc_queries_as_vec.png]]

Let's consider the example above. It shows a table where:
- Rows represent terms (Anthony, Brutus, Caesar, etc.)
- Columns represent documents (various Shakespeare plays)
- Values represent term frequencies (occurrences of each term in each document)

For example, "Brutus" appears once in "Anthony and Cleopatra," once in "Julius Caesar," zero times in "The Tempest," and so on.

The query "Brutus Caesar" is also represented as a vector with 1s in the positions corresponding to these terms and 0s elsewhere.

### Cosine Similarity

To determine how well a document matches a query, we calculate the cosine similarity between their vectors:

$$\text{score}(d, q) = \cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||} = \frac{\sum q_i \cdot d_i}{\sqrt{\sum q_i^2} \sqrt{\sum d_i^2}}$$

This measures the cosine of the angle between two vectors, ranging from 0 (orthogonal/unrelated) to 1 (perfectly aligned/identical). The intuition is that documents pointing in a similar direction as the query are more relevant.

## Retrieval Axioms and Term Frequency Constraints

Effective retrieval models should satisfy certain axioms:

1. **TFC1**: Documents with more occurrences of a query term should receive higher scores.
2. **TFC2**: The increase in score due to additional occurrences should diminish for higher frequencies (sublinear).
3. **TFC3**: When total term occurrences are equal, documents covering more distinct query terms should score higher.

These axioms reflect our intuition about what makes documents relevant to queries.

## Term Frequencies and Their Limitations

![[term_freq.png]]

Raw term frequencies (shown above) have limitations when used directly for document scoring:

- Documents can have very different lengths
- Some terms appear frequently in general language
- Some documents might use a term repeatedly without being more relevant

For example, the table shows a collection with terms like "nuclear," "weapon," etc., with their raw frequencies in different documents.

### Improving Term Frequency

Research shows that relevance does not increase proportionally with term frequency. A document containing a term 10 times isn't 10 times more relevant than a document containing it once.

To address this, we apply monotonic, sub-linear transformations like:
$$w(t, d) = 1 + \log \text{tf}(t, d)$$
![[sublinear_trans.png | 300]]

This logarithmic transformation means:
- 0 occurrences → 0
- 1 occurrence → 1
- 2 occurrences → 1.3
- 10 occurrences → 2
- 1000 occurrences → 4

This better aligns with human perceptions of relevance.

## Document Frequencies and Inverse Document Frequency (IDF)

Not all query terms are equally informative. The Term Discrimination Constraint (TDC) states that we should penalize terms popular across the collection.

Looking at word frequency distributions in language (Zipf's Law), a small set of words ("the", "of", "to") appear extremely frequently, while most words are rare. This is why IDF is introduced.

### Document Frequency

Document frequency is the number of documents in the collection containing a specific term:

$$\text{df}(t) := \#\{d : \text{tf}(t, d) > 0\}$$

Terms with lower document frequency are considered more discriminative and informative.

### Inverse Document Frequency (IDF)

IDF is calculated as:

$$\text{idf}(t) = \log \frac{n}{\text{df}(t)} = \log n - \log \text{df}(t)$$

Where n is the total number of documents in the collection. The logarithm "dampens" the effect of document frequency.

## TF-IDF Weighting Scheme

TF-IDF combines term frequency and inverse document frequency:

$$\text{tf-idf}(t, d) = \log(1 + \text{tf}(t, d)) \cdot \left[ \log \frac{n}{\text{df}(t)} \right]$$

For scoring documents, we can use either:

$$\text{score}(d, q) = \cos(\vec{q}, \vec{d})$$

Or more simply:

$$\text{score}(d, q) = \sum_{t:t \in q \wedge t \in d} \text{tf}(t, q) \cdot \text{idf}(t)$$

This weighting scheme provides a good balance of local (document-specific) and global (collection-wide) information about terms.

## BM25: A More Sophisticated Weighting Scheme

BM25 (Best Matching 25) is an advanced ranking function that has become the standard in information retrieval. It incorporates:

1. **Term frequency saturation**: Using a non-linear function to model diminishing returns of term frequency

$$\text{term weight} = \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1 + \text{tf}_{d,t}} \cdot \text{idf}(t)$$
where $k_1$ is a parameter that controls how quickly the function saturates. For high values of $k_1$, the weight continues increasing significantly with term frequency; for low values, it tapers off quickly.

2. **Document length normalization**: Accounting for the fact that longer documents might have higher term frequencies by chance

$$w_t = \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1 \cdot ((1 - b) + b \cdot (l_d/l_{avg})) + \text{tf}_{d,t}} \cdot \text{idf}(t)$$
where:
- $l_d$ is the document length
- $l_{avg}$ is the average document length
- $b$ is a parameter controlling the normalization impact (usually around 0.75)

3. **Query term weighting**: For longer queries (though rarely needed)

The complete BM25 formula is:

$$\text{BM25} = \sum_{\text{unique } t \in q} \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1((1 - b) + b \cdot (l_d/l_{avg})) + \text{tf}_{d,t}} \cdot \frac{(k_3 + 1)\text{tf}_{q,t}}{k_3 \cdot \text{tf}_{q,t}} \cdot \text{idf}(t)$$
- k1, b, and k3 need to be tuned (k3 only for very long queries)
- k1 ≈ 1.5 and b ≈ 0.75 common default

BM25 is highly effective and remains one of the most widely used weighting schemes in information retrieval.

## Language Models for Information Retrieval

A statistical language model is a probability distribution over word sequences. It assigns probabilities to sequences of words:

- P("Today is Wednesday") = 0.001
- P("Today Wednesday is") = 0.0000000000001
- P("The eigenvalue is positive") = 0.00001

Provides a principled way to quantify the uncertainties associated with the use of natural language.
### Query Likelihood Model

The query likelihood approach asks: "Which document model would most likely have generated this query?"

![[rank_by_query_likelihood.png | 400]]
For each document, we:
1. Build a statistical language model $\theta_d$
2. Compute the probability of the query being generated from this model: $P(q|\theta_d)$
3. Rank documents by this probability

Using Bayes' rule:
$$P(d|q) = \frac{P(q|d)P(d)}{P(q)}$$
Since $P(q)$ is constant for all documents, we can simplify:
$$\text{score}(d, q) = \log[p(q|\theta_d) \cdot p(d)]$$
Where $p(d)$ is a prior probability of the document (which can incorporate factors like document authority, recency, etc.).
1. How to define $\theta_d$ ?
2. How to estimate $\theta_d$ based on document $d$ ?
3. How to compute the prior $p(d)$ ?
### Event Models for $\theta_{d}$

Given a document d, what is the generative process for generating q = q1…qm?

Several ways exist to model how queries are generated:
1. **Multiple Bernoulli**: Models presence/absence of terms  in the document.$$p(q|\theta_d) = \prod_{w_i \in q} p(w_i|\theta_d) \prod_{w_i \notin q} (1 - p(w_i|\theta_d))$$
2. **Multinomial**: Models relative frequency of term in the document $$p(q|\theta_d) = \prod_{w_i \in q} p(w_i|\theta_d)$$
### Maximum Likelihood Estimation

To estimate the document model $\theta_d$, we typically use maximum likelihood estimation:

$$p_{ml}(w|\theta_d) = \frac{\text{tf}(w; d)}{|d|}$$

Where tf(w; d) is the frequency of word w in document d, and |d| is the document length.

### The Zero-Probability Problem and Smoothing

A major challenge with language models is that if a query term never appears in a document, the entire query gets a probability of zero (log(0) = -∞). This is known as the zero-probability problem.

Smoothing techniques address this by allocating some probability mass to unseen words. Several smoothing methods exist:

1. **Additive (Laplace) Smoothing**: Add a small count to every term $$p_{\epsilon}(q_i|\theta_d) = \frac{\text{tf}(q_i; d) + \epsilon}{|d| + \epsilon|V|}$$
    where |V| is the vocabulary size and ε is a small constant.
2. **Jelinek-Mercer Smoothing**: Linear interpolation with a background model $$\hat{P}_{\lambda}(w|d) = \lambda\frac{\text{tf}(w; d)}{|d|} + (1 - \lambda)\frac{\text{tf}(w; C)}{|C|}$$
    
3. **Dirichlet Prior Smoothing**: Bayesian smoothing using a Dirichlet prior $$p(w|\hat{\theta}_d) = \frac{\text{tf}(w; d) + \mu p(w|C)}{|d| + \mu}$$
    
    This can be rewritten as:
    
    $$p_{\mu}(w|\hat{\theta}_d) = \frac{|d|}{|d| + \mu}\frac{\text{tf}(w; d)}{|d|} + \frac{\mu}{\mu + |d|}p(w|C)$$
    
    Where μ is a parameter controlling the amount of smoothing.
    

Dirichlet prior smoothing typically performs best because it dynamically adjusts the amount of smoothing based on document length (shorter documents get more smoothing).

## Positional Language Models

Positional language models introduce the concept of term position and proximity:

4. For each word position in a document, define a language model that captures the document content at that position.
    
5. Each position defines a "fuzzy passage" centered at that position but potentially covering all words in the document with decreasing weights as distance increases.
    
6. Terms propagate their occurrence to nearby positions based on proximity kernels:
    
    - Gaussian kernel
    - Triangle kernel
    - Cosine kernel
    - Circle kernel

This produces a position-specific term frequency:

$$\text{tf}'(w, j; d) = \sum_{i=1}^{|d|} \text{tf}(w, i; d) \cdot k(j, i)$$

Where k(j, i) is the kernel function determining how much influence position i has on position j.

The language model for each position is then:

$$p(w|D, i) = \frac{\text{tf}'(w, i; d)}{\sum_{w' \in V} \text{tf}'(w', i; d)}$$

For ranking documents, two strategies can be used:

7. **Best position strategy**: Score document by the position with the maximum score
8. **Average top-k position strategy**: Use the average of the top k positions

The intution is that documents where query terms appear close together should be ranked higher, as proximity often indicates stronger topical relevance.

## Summary and Take-aways

In information retrieval, we've covered several important retrieval models:

9. **TF-IDF**: A vector space model that balances term frequency with inverse document frequency to determine term importance.
    
10. **BM25**: A probabilistic model that extends TF-IDF with better term frequency saturation and document length normalization.
    
11. **Language Models**: Probabilistic models that score documents based on the likelihood of generating the query, with various smoothing techniques to handle unseen terms.
    
12. **Positional Language Models**: Extensions that incorporate term proximity to improve retrieval quality.
    

BM25 and Language Models remain the dominant traditional retrieval models used in search engines and IR systems today, although they are increasingly being enhanced or replaced by neural approaches in some applications.

Each model addresses different aspects of the retrieval problem, and understanding their mathematical foundations provides insight into how to design more effective information retrieval systems.
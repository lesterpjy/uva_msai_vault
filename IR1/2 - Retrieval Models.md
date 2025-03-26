
## Introduction to Retrieval Models

<span style="color:rgb(172, 96, 230)">Retrieval models</span> provide the mathematical foundation for defining how queries match with documents in information retrieval systems. They offer:

1. A <span style="color:rgb(172, 96, 230)">mathematical framework for query-document matching</span>
2. Clear explanation of underlying assumptions
3. The basis for many ranking algorithms

The progression in retrieval models has directly led to <span style="color:rgb(172, 96, 230)">improvements in ranking effectiveness</span> over time. Understanding these models is crucial for developing effective search systems.

## Categories of Retrieval Models

Retrieval models can be broadly classified into:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
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

In the <span style="color:rgb(172, 96, 230)">vector space model</span>, both documents and queries are represented as vectors in a high-dimensional space, where each dimension corresponds to a term in the vocabulary.

### Term Frequency Representation

![[doc_queries_as_vec.png]]

Let's consider the example above. It shows a table where:
- Rows represent terms (Anthony, Brutus, Caesar, etc.)
- Columns represent documents (various Shakespeare plays)
- Values represent term frequencies (occurrences of each term in each document)

For example, "Brutus" appears once in "Anthony and Cleopatra," once in "Julius Caesar," zero times in "The Tempest," and so on.

The query "Brutus Caesar" is also represented as a vector with 1s in the positions corresponding to these terms and 0s elsewhere.

### Cosine Similarity

To <span style="color:rgb(172, 96, 230)">determine how well a document matches a query, we calculate the cosine similarity between their vectors</span>:

$$\text{score}(d, q) = \cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||} = \frac{\sum q_i \cdot d_i}{\sqrt{\sum q_i^2} \sqrt{\sum d_i^2}}$$

This measures the cosine of the angle between two vectors, ranging from 0 (orthogonal/unrelated) to 1 (perfectly aligned/identical). The intuition is that documents pointing in a similar direction as the query are more relevant.

## Term Frequencies and Their Limitations

![[term_freq.png]]

Raw term frequencies (shown above) have limitations when used directly for document scoring:

- Documents can have very different lengths
- Some terms appear frequently in general language
- Some documents might use a term repeatedly without being more relevant

For example, the table shows a collection with terms like "nuclear," "weapon," etc., with their raw frequencies in different documents.

### Improving Term Frequency

Research shows that <span style="color:rgb(172, 96, 230)">relevance does not increase proportionally with term frequency</span>. A document containing a term 10 times isn't 10 times more relevant than a document containing it once.

<span style="color:rgb(172, 96, 230)">Instead, apply monotonic, sub-linear transformations</span>:
$$w(t, d) = 1 + \log \text{tf}(t, d)$$
![[sublinear_trans.png | 300]]

This logarithmic transformation means:
- 0 occurrences → 0
- 1 occurrence → 1
- 2 occurrences → 1.3
- 10 occurrences → 2
- 1000 occurrences → 4

This <span style="color:rgb(172, 96, 230)">better aligns with human perceptions of relevance</span>.

## Retrieval Axioms and Term Frequency Constraints

Effective retrieval models should satisfy certain axioms:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. **TFC1**: Documents with more occurrences of a query term should receive higher scores.
2. **TFC2**: The increase in score due to additional occurrences should diminish for higher frequencies (sublinear).
3. **TFC3**: When total term occurrences are equal, documents covering more distinct query terms should score higher.
4. **Term discrimination constraint** (TDC): we should penalize the terms popular in the collection.

## Document Frequencies and Inverse Document Frequency (IDF)

Not all query terms are equally informative. The Term Discrimination Constraint (TDC) states that we should penalize terms popular across the collection.

Looking at word frequency distributions in language (Zipf's Law), a small set of words ("the", "of", "to") appear extremely frequently, while most words are rare. This is why IDF is introduced.

<span style="color:rgb(172, 96, 230)">Zipf's law: small set of common words appear much more frequently</span>
<span style="color:rgb(172, 96, 230)">Luhn's hypothesis: the significance of a document's words is proportional to their frequency of occurrence within that document. Words with intermediate frequency are more likely to be important for understanding the document's context. </span>

### Document Frequency

Document frequency is the number of documents in the collection containing a specific term:
$$\text{df}(t) := \#\{d : \text{tf}(t, d) > 0\}$$

Terms with lower document frequency are considered more discriminative and informative. This <span style="color:rgb(172, 96, 230)">depends on collection and need (somewhat) heavy computation</span>.

### Inverse Document Frequency (IDF)

<span style="color:rgb(172, 96, 230)">IDF, logarithm of the inverse of the document frequency</span>:

$$\text{idf}(t) = \log \frac{n}{\text{df}(t)} = \log n - \log \text{df}(t)$$

Where n is the total number of documents in the collection. The logarithm "dampens" the effect of document frequency.

## TF-IDF Weighting Scheme

TF-IDF combines term frequency and inverse document frequency:

$$\text{tf-idf}(t, d) = \log(1 + \text{tf}(t, d)) \cdot \left[ \log \frac{n}{\text{df}(t)} \right]$$

For scoring documents, we can use either:

$$\text{score}(d, q) = \cos(\vec{q}, \vec{d})$$

Or sum of tf-idf weights of terms contained in the query:

$$\text{score}(d, q) = \sum_{t:t \in q} \text{tf}(t, q) \cdot \text{idf}(t)$$

This weighting scheme provides a good balance of local (document-specific) and global (collection-wide) information about terms.

## BM25: A More Sophisticated Weighting Scheme

BM25 (Best Matching 25) is an advanced ranking function that has become the standard in information retrieval. It incorporates:

1. **Term frequency saturation**: Using a <span style="color:rgb(172, 96, 230)">non-linear function to model diminishing returns of term frequency</span>
$$\text{term weight} = \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1 + \text{tf}_{d,t}} \cdot \text{idf}(t)$$
where $k_1$ is a parameter that controls how quickly the function saturates. For <span style="color:rgb(172, 96, 230)">high values</span> of $k_1$, the <span style="color:rgb(172, 96, 230)">weight continues increasing</span> significantly with term frequency; for <span style="color:rgb(172, 96, 230)">low values</span>, it <span style="color:rgb(172, 96, 230)">tapers off quickly</span>.

1. **Document length normalization**: Accounting for the fact that <span style="color:rgb(172, 96, 230)">longer documents might have higher term frequencies by chance</span>
$$w_t = \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1 \cdot (l_d/l_{avg}) + \text{tf}_{d,t}} \cdot \text{idf}(t)$$
where:
- $l_d$ is the document length
- $l_{avg}$ is the average document length
<span style="color:rgb(172, 96, 230)">Long document often still convey more information</span>, so to control the level of normalization: $b$ for controlling the normalization impact (usually around 0.75)

$$w_t = \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1 \cdot ((1 - b) + b \cdot (l_d/l_{avg})) + \text{tf}_{d,t}} \cdot \text{idf}(t)$$


1. **Query term weighting**: <span style="color:rgb(172, 96, 230)">k3 for longer queries </span>(though rarely needed)

The complete<span style="color:rgb(172, 96, 230)"> BM25 formula </span>is:

$$\text{BM25} = \sum_{\text{unique } t \in q} \frac{(k_1 + 1)\text{tf}_{d,t}}{k_1((1 - b) + b \cdot (l_d/l_{avg})) + \text{tf}_{d,t}} \cdot \frac{(k_3 + 1)\text{tf}_{q,t}}{k_3 \cdot \text{tf}_{q,t}} \cdot \text{idf}(t)$$
- k1, b, and k3 need to be tuned (k3 only for very long queries)
- k1 ≈ 1.5 and b ≈ 0.75 common default

BM25 is highly effective and remains one of the most widely used weighting schemes in information retrieval.

## Language Models for Information Retrieval

A statistical <span style="color:rgb(172, 96, 230)">language model</span> is a probability distribution over word sequences. It assigns probabilities to sequences of words:

- P("Today is Wednesday") = 0.001
- P("Today Wednesday is") = 0.0000000000001
- P("The eigenvalue is positive") = 0.00001

Provides<span style="color:rgb(172, 96, 230)"> a principled way to quantify the uncertainties associated with the use of natural language</span>.
### Query Likelihood Model

The query likelihood approach asks: "Which document model would most likely have generated this query?"

![[rank_by_query_likelihood.png | 400]]

For each document:

1. Build a statistical language model $\theta_d$
2. Compute the <span style="color:rgb(172, 96, 230)">probability of the query being generated from this model</span>: $P(q|\theta_d)$
3. Rank documents by this probability

Using Bayes' rule:
$$P(d|q) = \frac{P(q|d)P(d)}{P(q)}$$
Since $P(q)$ is constant for all documents, we can <span style="color:rgb(172, 96, 230)">simplify</span>:
$$\text{score}(d, q) = \log[p(q|\theta_d) \cdot p(d)]$$
Where $p(d)$ is a prior probability of the document (which can incorporate factors like document authority, recency, etc.).
1. How to define $\theta_d$ ?
2. How to estimate $\theta_d$ based on document $d$ ?
3. How to compute the prior $p(d)$ ?
### Event Models for $\theta_{d}$

Given a document d, what is the generative process for generating q = q1…qm?

Several ways exist to model how queries are generated:
1. **Multiple Bernoulli**: <span style="color:rgb(172, 96, 230)">Models presence/absence of terms  in the document</span>.$$p(q|\theta_d) = \prod_{w_i \in q} p(w_i|\theta_d) \prod_{w_i \notin q} (1 - p(w_i|\theta_d))$$
2. **Multinomial**: <span style="color:rgb(172, 96, 230)">Models relative frequency of term in the document </span>$$p(q|\theta_d) = \prod_{w_i \in q} p(w_i|\theta_d)$$
3. Multiple Poisson, Hypergeometric
### Maximum Likelihood Estimation

To <span style="color:rgb(172, 96, 230)">estimate the document model</span> $\theta_d$, we typically use maximum likelihood estimation:

$$p_{ml}(w|\theta_d) = \frac{\text{tf}(w; d)}{|d|}$$

where tf(w; d) is the frequency of word w in document d, and |d| is the document length.
Derivation:
![[estimation_theta_d.png | 350]]

### The Zero-Probability Problem and Smoothing

A major challenge with language models is that if <span style="color:rgb(172, 96, 230)">a query term never appears in a document</span>, the entire query gets a probability of zero (log(0) = -∞). This is known as the zero-probability problem. <span style="color:rgb(172, 96, 230)">Incorporating a prior belief of a distribution characterizing the topic of a document.</span>

Smoothing techniques address this by allocating some probability mass to unseen words. Several smoothing methods exist:
<span style="color:rgb(172, 96, 230)">Additive (Laplace) Smoothing</span>: Add a small count to every term $$p_{\epsilon}(q_i|\theta_d) = \frac{\text{tf}(q_i; d) + \epsilon}{|d| + \epsilon|V|}$$
where |V| is the vocabulary size and ε is a small constant.
#### Limitations of Basic Additive Smoothing
The basic additive (Laplace) smoothing assigns the same small probability to all unseen words, which is not ideal because:
1. Not all unseen words should have equal probability of appearing
2. The <span style="color:rgb(172, 96, 230)">probability of an unseen word should be proportional to the probability of a word given a background model</span>
3. Some words are inherently more likely to occur than others, even if unseen in a particular document
#### Background Probability Estimation
A key improvement is incorporating a background language model. This involves estimating $p(w|C)$ where $C$ <span style="color:rgb(172, 96, 230)">represents the entire collection</span>. There are several approaches to estimating this background probability:
##### Words Contributing Equally
This approach treats all word occurrences in the collection equally:
- Each word instance contributes the same amount to the probability
- $p(w|C) = tf(w,C) / |C|$ (where $|C|$ is the total number of words in the collection)
##### Documents Contributing Equally
This approach normalizes by document first, then combines document probabilities:
$$
p(w|C) = \frac{1}{|C|}\sum_{d\in C} \frac{tf(w;d)}{|d|}
$$
- Each document contributes equally regardless of length
- Calculates $p(w|d)$ for each document $d$, then averages these probabilities
##### Document Frequency Approach
This focuses on how many documents contain the word:
$$
p(w|C) = \frac{df(w)}{\sum_{w'\in V}df(w')}
$$
- $p(w|C)$ proportional to the number of documents containing w
- Similar to the IDF concept in traditional IR models
- Gives higher probability to words that appear in many documents
### Jelinek-Mercer Smoothing
This approach directly uses linear interpolation between the document model and the background model:
$$\hat{P}_\lambda(w|d) = \lambda \frac{\text{tf}(w;d)}{|d|} + (1-\lambda)p(w|C)$$
Where:
- $λ$ is the interpolation parameter (typically between 0 and 1)
- $tf(w;d)$ is the term frequency of word $w$ in document $d$
- $|d|$ is the document length
- $p(w|C)$ is the background language model probability

The <span style="color:rgb(172, 96, 230)">key benefit of Jelinek-Mercer smoothing</span> is that it explicitly balances:
- The maximum likelihood estimate from the document $tf(w;d)/|d|$
- The collection statistics represented by $p(w|C)$

The parameter $λ$ controls this balance and can be tuned empirically (typically around 0.1-0.7 depending on queries and collection).

### Dirichlet Prior Smoothing
Dirichlet prior smoothing represents a more theoretically grounded approach based on Bayesian statistics. It uses <span style="color:rgb(172, 96, 230)">Maximum A Posteriori (MAP) estimation rather than maximum likelihood</span>.
$$
\hat{\theta_{d}} = \text{argmax}_{\theta_{d}} p(\theta_{d}|d) =  \text{argmax}_{\theta_{d}} p(d|\theta_{d})p(\theta_{d})
$$
#### Mathematical Framework
1. **Distribution of Evidence (Likelihood)**: The document generation process is modeled as a multinomial distribution:

$$p(d|\theta_d) = \frac{\Gamma\left( \sum_{w\in V} \text{tf}(w;d) + 1 \right)}{\prod_{w\in V} \Gamma(\text{tf}(w;d) + 1)}\prod_{w\in V}p(w|\theta_{d})^{tf(w;d)}$$

Where $Γ$ is the gamma function, an extension of the factorial function.

1. **Prior Distribution**: The parameters follow a <span style="color:rgb(172, 96, 230)">Dirichlet prior distribution</span>:

$$p(\theta_d) = \frac{\Gamma(\sum_{w\in V} \alpha_w)}{\prod_{w\in V} \Gamma(\alpha_w)} \prod_{w\in V} p(w|\theta_d)^{\alpha_w-1}$$

Where:
- $α_w$ <span style="color:rgb(172, 96, 230)">represents the prior count for word</span> $w$
- The expected value $E[p(w|θ_d)] = α_w / Σ_w' α_w'$

2. **Posterior Distribution**: Using Bayes' rule:

$$p(\theta_d|d) \propto p(d|\theta_d)p(\theta_d) \propto \prod_{w\in V} p(w|\theta_d)^{\text{tf}(w;d)} \prod_{w\in V} p(w|\theta_d)^{\alpha_w-1}$$
Which simplifies to:

$$p(\theta_d|d) = \prod_{w\in V} p(w|\theta_d)^{\text{tf}(w;d)+\alpha_w-1}$$
#### MAP Estimation with Lagrange Multipliers
To find the maximum of this posterior distribution subject to the constraint that probabilities sum to 1, we apply Lagrange multipliers, which gives us:

$$p(w|\hat{\theta}_d) = \frac{\text{tf}(w;d) + \alpha_w-1}{|d| + \sum_{w\in V} \alpha_w-|V|}$$
For computational simplicity, we can set $α_w = μp(w|C) + 1$, which yields:

$$p(w|\hat{\theta}_d) = \frac{\text{tf}(w;d) + \mu p(w|C)}{|d| + \mu}$$
#### Special Cases of Parameter Settings
The choice of $α_w$ values creates different smoothing behaviors:
1. If $α_w = 1$ for all $w$: This becomes equivalent to Maximum Likelihood Estimation
2. If $α_w = 2$ for all $w$: This becomes equivalent to Laplace (add-one) smoothing
3. If $α_w = μp(w|C) + 1$ : This creates a background-model-informed smoothing where:
    - μ <span style="color:rgb(172, 96, 230)">controls the overall amount of smoothing</span>
    - The distribution of probability mass to unseen words is proportional to their background probabilities

#### Dirichlet Prior as Interpolation
The Dirichlet prior smoothing can be rewritten as:

$$p_\mu(w|\hat{\theta}_d) = \frac{|d|}{|d|+\mu}\frac{\text{tf}(w;d)}{|d|} + \frac{\mu}{|d|+\mu}p(w|C)$$

This reveals that it's also a form of <span style="color:rgb(172, 96, 230)">interpolation between the document model and background model</span>, but with a document-length-dependent interpolation weight. Longer documents need less smoothing (rely more on document statistics), while shorter documents need more smoothing (rely more on background statistics).

### Variants of Basic Language Models in Information Retrieval

Beyond the core smoothing techniques, language modeling approaches in IR have been extended in several directions:
#### Different Event Models
4. **Multinomial**: The standard approach where each word position is a sampling event
5. **Multiple Bernoulli**: Models presence/absence of words rather than counts
6. **Poisson**: Models word occurrences as Poisson processes
#### Diverse Smoothing Strategies
7. **Hidden Markov Models**: Using state transitions to capture context
8. **IDF-like Reference Models**: Incorporating inverse document frequency intuitions
9. **Chen & Goodman Techniques**: Including Katz backoff, Witten-Bell smoothing, etc.
#### Different Priors
10. **Link Information**: Using hyperlink structure as prior belief
11. **Temporal Information**: Incorporating document recency as a prior
12. **PageRank**: Using web graph authority as a prior for document importance
#### Capturing Term Dependencies
13. **N-grams**: Using bigrams/trigrams to capture local word patterns
14. **Grammatical Dependencies**: Modeling syntactic relationships between terms
15. **Positional Language Models**: Incorporating word proximity information
16. **Markov Random Fields**: Modeling arbitrary dependencies between query terms

### Practical Considerations
When implementing language model approaches for IR:
17. **Parameter Tuning**: The smoothing parameters (λ in Jelinek-Mercer, μ in Dirichlet) significantly affect performance and should be tuned for specific collections and query types.
18. **Document Length Normalization**: Dirichlet smoothing inherently addresses document length issues, making it often preferred for heterogeneous collections.
19. **Query Formulation**: Language models can treat queries differently (as samples from relevant documents or as generative specifications).
20. **Computational Efficiency**: Many smoothing techniques increase computational requirements, requiring efficient implementations for large-scale retrieval.

## Positional Language Models
Positional language models introduce the concept of term position and proximity:
1. For each word position in a document, <span style="color:rgb(172, 96, 230)">define a language model that captures the document content at that position</span>.
2. Each position defines a "fuzzy passage" centered at that position but potentially covering all words in the document with decreasing weights as distance increases.
3. Terms propagate their occurrence to nearby positions based on proximity kernels:
    - Gaussian kernel
    - Triangle kernel
    - Cosine kernel
    - Circle kernel

This produces a position-specific term frequency:
$$\text{tf}'(w, j; d) = \sum_{i=1}^{|d|} \text{tf}(w, i; d) \cdot k(j, i)$$

Where $k(j, i)$ is the kernel function determining how much influence position i has on position j. The language model for each position is then:

$$p(w|D, i) = \frac{\text{tf}'(w, i; d)}{\sum_{w' \in V} \text{tf}'(w', i; d)}$$

For ranking documents, two strategies can be used:
1. **Best position strategy**: Score document by the position with the maximum score
2. **Average top-k position strategy**: Use the average of the top k positions

The intuition is that <span style="color:rgb(172, 96, 230)">documents where query terms appear close together should be ranked higher, as proximity often indicates stronger topical relevance</span>.

## Summary and Take-aways

In information retrieval, we've covered several important retrieval models:
3. **TF-IDF**: A vector space model that balances term frequency with inverse document frequency to determine term importance.
4. **BM25**: A probabilistic model that extends TF-IDF with better term frequency saturation and document length normalization.
5. **Language Models**: Probabilistic models that score documents based on the likelihood of generating the query, with various smoothing techniques to handle unseen terms.
6. **Positional Language Models**: Extensions that incorporate term proximity to improve retrieval quality.

BM25 and Language Models remain the dominant traditional retrieval models used in search engines and IR systems today, although they are increasingly being enhanced or replaced by neural approaches in some applications.

Each model addresses different aspects of the retrieval problem, and understanding their mathematical foundations provides insight into how to design more effective information retrieval systems.
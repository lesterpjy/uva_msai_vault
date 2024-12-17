## Lexical semantics and word embeddings
### Semantics

Compositional semantics
- meanings of phrases are constructed out of meaning of individual words
- meaning of each whole phrase derivable from meaning of its parts
- sentence structure convey meaning by syntactic representation
Lexical semantics
- None of the semantic representations give us a complete account of lexical meaning

**Formal semantics**: a set-theoretic approach to lexical meaning
Example: cat': the set of all cats
Highly problematic: exceptions and enumeration

**Prototype theory**
Notion of graded semantic categories
- No requirement that a property be shared by all members
- Certain member more _prototypical_: chair is more prototypical than stool in furnitures
Categories form around prototypes, members added on resemblance to prototype

**Semantic Relations**
Hyponymy: IS-A
- dog is a hyponym of animal
- animal is a hypernym of dog

Meronomy: PART-OF
- arm is a meronym of body
- wheel a meronym of car

Synonymy: aubergine - eggplant
Antonymy: big - little
Near-synonymy/similarity: exciting - thrilling

**Polysemy**
- Homonymy: unrelated word senses, bank (river bank) vs bank (finance)
- Regular polysemy
	- metaphorical senses: swallow food, swallow anger (substitution with comparison)
	- metonymy: he drank his _glass_ ( direct substitution without comparison)
	- zero-derivation: tango (N) vs. tango (V)

### Distributional Hypothesis
The meaning of a word is defined by the way it is used
- the context surrounding a given word provides information about its meaning;
- words are similar if they share similar linguistic contexts;
- semantic similarity ≈ distributional similarity
- vectors in semantic space, dimensions correspond to features

Context types
- unfiltered word windows: n-words on either side of the lexical term
- filtered word windows: n-words on either side of the lexical term, stop-word filtered
- lexeme window (filtered/ unfiltered):  n-words on either side of the lexical term, but using stems.
- syntactic relations: context for a lexical item is the syntactic dependency structure it belongs to.

Context weighting
1. Binary model: if context co-occurs with word w, value of vector w for dimension c is 1, 0 otherwise.
2. Basic frequency model: the value of vector w for dimension c is the number of times that c co-occurs with w.
3. Characteristic model: Weights given to the vector components express how characteristic a given context is for word w. Example, with pointwise mutual information (PMI)
![[pmi.png | 400]]

Semantic space: entire vocabulary (very sparse), top n words (miss out on infrequent words), dimensionality reduction (not interpretable)
Word frequency follows a Zipfian distribution.

Cosine Similarity
$$
\cos(\theta) = \frac{\mathbf{A}\cdot\mathbf{B}}{||\mathbf{A}||\ ||\mathbf{B}||}= \frac{\sum_{i=1}^nA_{i}B_{i}}{\sqrt{ \sum A_{i}^2 }\sqrt{ \sum B_{i}^2 }}
$$
Distributional methods are a usage representation, and is thus corpus dependent, culture dependent, and register dependent. Example: policeman - cop

### Distributional semantic models
1. Count-based models:
	1. Explicit vectors: dimensions are elements in the context
	2. long sparse vectors with interpretable dimensions
2. Prediction-based models:
	1. Train a model to predict plausible contexts for a word
	2. learn word representations in the process
	3. short dense vectors with latent dimensions

Dense vector benefits: use as features in ML, generalize better, better capturing synonymy

#### Word2Vec
Skip-gram model
- inspired by work on neural language models
- train a neural network to predict neighboring words
- learn dense embeddings for the words in the training corpus in the process

- Sometimes referred to as word2vec because it is implemented in this package
- Given a word $w_{t}$, predict neighbouring words in a context windows of 2L words (for L = 2: wt−2; wt−1; wt+1; wt+2)
- In skip-gram, we learn two representations for every word wj 2 V :
	- word embedding v in word matrix W
	- context embedding c in context matrix C (word in the role as context for other words)
- To learn these embeddings, we take every word w(t) in the corpus (index j in vocabulary), and try to predict w(t + 1)... where we denote this word with index k in the vocabulary:
$$
p(w_{k}|w_{j})
$$

- The idea in skip-gram is that we compute this probability by the similarity between the words wk and wj whereas we use the context matrix C for wk and the word matrix W for wj 
![[word2vec_embeddings.png | 400]]
- Similar to the cosine similarity, we use the dot product for calculating this:
$$
\text{Similarity}(c_{k},v_{j}) \varpropto c_{k}\cdot v_{j}
$$
...

### Properties of Embeddings
- capture similarity
- capture analogy via vector offsets
- capture range of semantic relations
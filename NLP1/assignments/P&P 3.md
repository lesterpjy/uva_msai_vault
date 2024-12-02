## Compositional semantics and discourse

Compositional semantics is concerned with modeling the meaning of phrases and sentences. It relies on the principle of compositionality, which states that the meaning of a phrase or a sentence can be derived from the meanings of its parts, guided by syntactic structure.

In the following examples, identify and explain the issue compositional semantics models might face in each of the examples. Additionally, explain how these issues might complicate compositional semantics.

### 1a) _it snows_
The word _it_ in _it snows_ is a pleonastic pronoun that does not refer to a specific entity, unlike the word _it_ in a sentence like _it barks_ which may refer to a specific referent, like a dog. Compositional semantics assumes the meaning of a sentence can be derived from the meaning of its parts, however, the pronoun _it_ in this case does not contribute any referential meaning in this case.

### 1b) rock the boat
The phrase _rock the boat_ is an idiom, and its meaning of "causing disruption to the status quo" cannot be derived from the meanings of the individual words (rock, the, boat) with compositional semantics. Idioms violate the principle of compositionally because their meanings are not a function of their parts or syntactic structure.

### 1c) enjoy a cigarette/ enjoy a book
These phrases involve logical metonymy, where the verb _enjoy_ implies an associated activity that many be different depending on the objects involved. In this example, _enjoy a cigarette_ means smoking a cigarette, and _enjoy a book_ means to read a book. The activity associated is inferred based on context and world knowledge, rather than the meaning of the word _enjoy_. This poses a challenge to compositional semantics as the meaning of the phrase is not fully contained within the meanings of the phrase's parts.

### 1d) inject life/ inject hydrogen

Similar to the previous example, the example phrases demonstrate logical metonymy and polysemy of the word _inject_. _Inject hydrogen_ may mean the physical action of literally injecting  hydrogen into something, while the phrase _inject life_ metaphorically means to bring energy or vitality to something. This is an complication for compositional semantics as the literal and metaphorical meanings of _inject_ may not be distinguishable from the composition of meanings of the lexical items in the phrase.


Distributional representations of phrases and sentences can be derived by combining, or _composing_, the distributional representations of individual words in them.
### 1e) Describe two limitations of vector mixture models.
1. Vector mixture models do not take word order or syntax in to account: the additive and multiplicative models means the model symmetric, thus "John hit the ball" and "the ball hit John" receives the same representation.
2. By collapsing multiple vectors into one, vector mixture models fail to retain information about how words interact in the phrase or sentence, and function words like conjunctions or prepositions (_some_ dogs, lice _on_ dogs) are lost in the representations.

### 1f) _Briefly_ describe how do lexical function models address these limitations.
Lexical function models address these limitations by incorporating word specific transformation functions, in the form of lexical functions, that capture the contextual and relational information between words. By using lexical functions of specific words like verbs or modifiers to transform the representation of their arguments like objects and subjects, the model can capture word order and relational dependencies.

### 1g) Describe how adjectives can be represented as lexical functions and how such lexical functions can be learned from a text corpus.
Adjectives like "old" or "big" are modeled as lexical functions that can be represented with parameter matrices, $A_{old}$ and $A_{big}$. The lexical functions $f_{adj}(\vec{dog})$ can then be expressed as linear transformations of the vector representation of the noun $\vec{dog}$:
$$
\begin{equation}
\begin{aligned}
\text{big dog} &= A_{big} \cdot \vec{dog}\\
\text{old dog} &= A_{old} \cdot \vec{dog}
\end{aligned}
\end{equation}
$$
The adjective matrices can be learned from a text corpus with the following steps:
1. **Build word representations**: construct distributional vectors using methods like Word2Vec , GloVe for all words in the vocabulary, including vectors for each noun $\vec{n_{j}}$ and adjective-noun pair $\vec{p_{ij}}$, for each adjective noun pairs $(a_{i}, n_{j})$.
2. **Extract training data**: identify the set of tuples $\{(n_{j}, p_{ij})\}_{j}$ that represents the dataset $D(a_{i})$ for the adjective $a_{i}$.
3. **Optimize matrices**: learn matrix $A_{i}$ from $D(a_{i})$ using linear regression, where we minimize the squared error loss, $L(A_{i})=\sum_{j\in D(a_{i})} ||p_{ij} - A_{i}n_{j}||^2$


Suggest a way in which the above lexical function model of adjective meaning can be used to:

### 1h) Paraphrase adjective-noun phrases, obtaining paraphrase pairs such as smart students → clever students
This can be achieved with the lexical function model assuming synonyms like "smart" and "clever" will produce similar transformations on the noun vector "student", then the paraphrasing can be obtained by identifying adjective-noun representations that are close to one another in the vector space. For example, the cosine similarity between "smart student" and all other "adjective-student" vectors in the vocabulary can be computed to identify pairs of high similarity.

### 1i) Disambiguate the meaning of the adjective, for instance, the meaning of _warm_ in warm tea vs. warm person.
The lexical function $f_{\text{warm}}$ can be applied to the noun vectors of tea and person to encode context specific meanings of the adjective "warm". Then the resulting vector representations for the adjective-noun pairs can then be compared with vectors of potential meanings of the phrases. For instance, computing similarities of representations between the adjective-noun pairs with the representations of words like "temperature" and "affection" would then reveal that the phrase "warm tea" is more associated with the temperature sense of the adjective "warm" and "warm person" is more associated with the "affection" sense of the adjective "warm". This method allows lexical function models to disambiguate the meanings of adjective depending on the noun the adjective modifies.
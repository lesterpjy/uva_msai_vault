#### 1. The distributional hypothesis states that the meaning of a word can be defined by its use and, therefore, it can be represented as a distribution of contexts in which the word occurs in a large text corpus. Name **and describe** four different types of contexts that can be used for this purpose.

- Unfiltered word windows: this type of context includes n words on either side of the lexical items as the context. For n=2 and the word "cat" in sentence "The large cat ate the fish", the context is [the 2, large 1, ate 1, fish 0].
- Filtered word windows: this type of context includes n words on either side of the lexical items as the context, but removes some words, such as function words. For n=2 and the word "cat" in sentence "The large cat ate the fish", the context is [large 1, ate 1, fish 0].
- Lexeme window (filtered or unfiltered): this type of context is the similar to the filtered word windows but uses stems of words. For n=2 and the word "cat" in sentence "The large cat ate the fish", the context is [large 1, eat 1, fish 0].
- Syntactic relations (dependencies): this type of context for a lexical item is the syntactic dependency structure it belongs to. For n=2 and the word "cat" in sentence "The large cat ate the fish", the context are: [large 1, eat 1], [large_mod 1, eat_subj 1], [large 1, eat+fish 1]

#### 2. The contexts can be weighted using Pointwise Mutual Information (PMI). Explain how PMI is calculated and how individual probabilities are estimated from a text corpus, giving the formulae.

The point wise mutual information evaluate how much the actual co-occurrence of two events differs from their expected co-occurrence under the assumption of independence. Intuitively, PMI calculates the ratio between the frequency of context and word co-occurrence and the frequency of context and word co-occurrence if they are independent:
$$
\text{PMI(w, c)} = \log \frac{P(w,c)}{P(w)P(c)} = \log\frac{P(c|w)P(w)}{P(w)P(c)} = \log \frac{P(c|w)}{P(c)} 
$$
This ratio tells us how much the actual co-occurrence deviates from the expectation under independence. This allows us to weight the context as PMI tells us how much the specific context $c$ is informative about $w$ compared to a baseline of independence. In practice PMI can be estimated with the following from a corpus,
$$
\text{PMI}(w,c) = \log \frac{f(w,c) \sum_{k}f(c_{k})}{f(w)f(c)}
$$
Since $P(c) = \frac{f(c)}{\sum_{k}f(c_{k})}$ and $P(c|w) = \frac{f(w,c)}{f(w)}$, and $f(w,c)$ is the frequency of word $w$ in context $c$, $f(w)$ is the frequency of word $w$ in all contexts, $f(c)$ is the frequency of context $c$. $\sum_{k}f(c_{k})$ is a constant for the entire corpus, representing the total number of context instances. We calculate these frequencies from the text corpus.

#### 3. Some words occur very rarely in the corpus. Show how this affects their PMI scores as contexts, and explain why this effect occurs.

For rare words their marginal probabilities $P(w)$ will be very low, leading to a very small denominator $P(w)P(C)$ that estimates the expected co-occurrence. As such even a small number of actual co-occurrence $P(w,c)$ would lead to an inflation of the PMI score due the very small denominator. This means that a high PMI score for a rare word-context pair might be a reflection of the rarity of the word, rather than actual semantic or syntactic relationship between the pair.
#### 4. How are the clusters produced in the two experiments different with respect to the similarity they capture? What lexico-semantic relations do the clusters exhibit?

In experiment 1, the clusters captures taxonomic or categorical similarity. For example, "carriage" and "bike" are both hyponyms of transportation vehicles, "official" and "inspector" are both hyponyms of occupations, and "daughter" and "relative" are both hyponyms of relationships. Synonymy is also captured with words like "officer" and "policeman".

In experiment 2, the clusters captures thematic or contextual similarity. Lexico-semantic relations captured in these clusters include collocation of words in similar contexts (like "driver" and "highway" or "concert" and "singer"), thematic associations of words, and functional relations that describe objects and actions associated with activities (like "steering" for the first cluster and "research" in the third cluster).

#### 5. The same clustering algorithm, K-means, was used in both experiments. What was different in the setup of the two experiments that resulted in the different kinds of similarity captured by the clusters? Give 2 such design choices, explain how each of these differed between the two experiments, and how this affects the resulting clusters in each experiment.

One design choice for the clustering algorithm is the feature representation used for representing each of the words. In experiment 1, feature representations derived based on contextual or semantic similarity is more likely used, as words that are hyponyms and synonyms are clustered together, suggesting that the features encode semantic similarity. In experiment 2, co-occurrence based feature representations are more likely used, as words are clustered based on thematic or contextual similarity. Words like "car" and "engine" are likely to co-occur in the same context, thus encoded similarly in their feature representations.

Another design choice may be based on the choice of the similarity metric. To capture semantic similarity the features representation needs to encode abstract relationships between words, and thus experiment 1 likely focused on higher dimensional representations of words where cosine similarity was used to calculate the feature similarities and capture deeper semantic relationships between the words. In experiment 2 for capturing contextual similarity, a simpler metric like Euclidean distance between the context word co-occurrence count can be used. The resulting clusters in this case, as mentioned in the previous question, captures the thematic similarity of the words.

#### 6. Explain what is PP- attachment ambiguity using an example, and explain why it might be challenging for a syntactic parser. Outline how techniques from distributional semantics can be used in conjunction with a syntactic parser to help disambiguate prepositional phrase attachment ambiguities. Explain how such a system could be designed and how you would then use it to assign the correct parsing to a new sentence.




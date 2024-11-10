NGram LMs: Markov assumption + tabular parameterization
NGram LM makes the conditional independence assumption (**Markov assumption**), ie, next word is conditionally independent of all but the $N-1$ preceding words.
$$
P_{X}(w_{1:l}) \overset{\text{ind.}}{=} \prod_{i=1}^l P_{W|H}(w_{i}|\langle w_{i-N+1},\dots,w_{i-1}\rangle)
$$

> [!figure] ![[ngram.png | 400]] 
> *Figure 2*: Top-down: autoregressive LM, unigram LM (N=1), bigram LM (N=2),
trigram LM (N=3)

There is still issue with unseen NGrams, and this can be delt with using Laplace smoothing.

For unknown words not in the vocabulary: augment the vocabulary with a placeholder symbol such as $\text{UNK}$, whenever you encounter an unknown symbol in the future (e.g., “hare”) treat it as $\text{UNK}$.

Note that: 
- Tabular representation is statistically inefficient. Memorizing in tabular cpds along with counts: increasing in order has exponential cost: $V^N \rightarrow V^{N+1}$
- long range dependency is very common in natural language, thus Markov assumption is just for convenience alone.
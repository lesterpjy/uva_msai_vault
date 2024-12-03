## Sequence Labelling

NGram LMs: words are atomic symbols
- large tabular cpds: store prob of conditional outcomes that are possible
- statistical inefficiency (struggle with data sparsity): linguistically related outcomes do not share statistical evidence
Overcome this in 2 ways:
- linguistically motivated change in data we model, new ideas for factorization
- change in parametrization

Method of organizing words into classes: **semantic criteria**, **formal criteria**, **distributional criteria**
![[word_class_criteria.png]]

**Word classes** capture aspects of word relatedness
Example: universal part-of-speech tags
Why? $\Rightarrow$ enable form of delexicalised natural language processing: learn patterns common to all words that share a given property
Example: POS tag captures a word's syntactic function
## Hidden Markov Model

Prescribe a **joint distribution** over the space of **texts annotated with their POS tags**.
Learn to assign probability to sequence pairs $(w_{1:l}, c_{1:l})$
Example: $(\langle \text{a, nice, dog}\rangle, \langle \text{DT, JJ, NN}\rangle)$
Applications: text analysis, language modeling, labeling tasks like entity recognition, semantic labelling

### Formalization
$W$ is a random variable for a word, with outcome $w$ from vocabulary $\mathcal{W}$ of size $V$
$C$ is a random variable for the POS tag, with outcome $c$ from tagset $\mathcal{C}$ of size $K$
$X = \langle W_{1},\dots,W_{L}\rangle$ is a random word sequence with outcome $w_{1:l}$
$Y = \langle C_{1},\dots,C_{L}\rangle$ is a random tag sequence with outcome $c_{1:l}$

Statistical Task:
Mechanism to assign probability $P_{XY}(w_{1:l},c_{1:l})$ to any outcome.
- Factorize $P_{XY}$ with chain rule, conditional independencies
- Parameterize the factors with tabular Categorical cpds
Use MLE to estimate the free parameters of our parameterization

NLP Tasks:
**Predict POS tag sequence** for a given text. For example, via mode-seeking search:
$$
\underset{c_{1:l}\in C^l}{\text{argmax}}\ P_{Y|X}(c_{1:l}|w_{1:l})
$$
**Assign probability to text** that is not annotated with POS tags, via marginalization:
$$
P_{X}(w_{1:l})=\sum_{c_{1:l}\in C^l} P_{XY}(c_{1:l}|w_{1:l})
$$
The outcome assigned largest probability mass is know as the _mode_ of the probability distribution.

Challenge for factorization: $P_{XY}$ is a distribution over a countably infinite space of sequence pairs.
Solution: Re-express the probability of sequence pair using the probability of the "steps" needed to generate it $\rightarrow$ countably finite sample space.

### Illustration

![[hmm_illustration.png | 500]]

The joint probability is then:
![[hmm_joint_prob.png | 300]]

In the chain rule for HMM, we assume conditional independences:
1. $W_{i}$ is independent of all but $C_{i}$
2. $C_{i}$ is independent of all but $C_{i-1}$
leading to:
![[hmm_chain_rule.png | 400]]
Notice that the sequence is padded with:
- BOS tag (context for the 1st transition)
- EOS tag (for the final transition)
- EOS token (for the final emission)


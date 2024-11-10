Language model: a **probability distribution** over the set of all strings in a language. LM can:
1. assign probability to text
2. generate text by drawing samples from LM

### How do design LMs
Requires:
1. A sample space (the set of possible outcomes that the LM can generate)
2. Probability mass function: function mapping each outcome to its probability mass.

Typically for LMs, the sample space is the set of all finite-length sequences (sequence of words/tokens) denoted by $\mathcal{W}^*$. 
$X = \langle W_{1},\dots,W_{L}\rangle$ is a random sequence of of $L$ words, and an outcome of $X$, such as $\langle W_{1},\dots,W_{L}\rangle$ is a sequence in $\mathcal{W}^*$. 

Under this framework, the a LM is a mechanism to assign a probability value $P_{X}(w_{1:l})$ to **each and every** outcome $w_{1:l} \in \mathcal{W}^*$.

**Intuition**: Why is designing $P_{X}$ over the countably infinite space of variable-length sequence a challenge?

## LM Factorization
**Idea**: re-express the probability of outcome $w_{1:l} \in \mathcal{W}^*$ using probabilities assigned to the steps that jointly derives the sequence.

Example
$x = \langle \text{He, went, to, the, store, EOS} \rangle$, and think of $x$ as the result of incrementally expanding an empty sequence $\langle\rangle$, the probability of these ordered steps such that $X = x$ is then given by:
![[factorized_probability.png | center | 300]]
Each time probability is assigned to a word, it is conditioned on the **ordered** history of words that precede it.

This gives us a general LM that has an *autoregressive factorization* of the probability of a sequence:
$$
P_{X}(w_{1:l}) \overset{\vartriangle}{=} \prod_{i=1}^l P_{W|H}(w_{i}|w_{<i})
$$
The pmf of the Categorical random variable $P(W|H=h)$ can be represented by a unit-norm, $V$-dimensional vector $\theta^{(h)}$ of probability masses:
$$
P_{W|H=h} \sim \text{Categorical}(\theta_{1:V}^{(h)})
$$
which implies
$$
P_{W|H}(w|h) = \theta_{w}^{(h)}
$$
Generating outcomes, ie, assigning probability to outcomes, from the LM distribution $P_{X}$ is the procedure known as a **generative story**:
1. Start with an empty history $h_{i} = \langle\rangle$, set $i = 1$.
2. Condition on the available history $h_{i}$ and draw a word $w_{i}$ with probability $P_{W|H}(w_{i}|h_{i})$  and extend the history $h_{i+1}$
3. If $w_{i}$ is $\text{EOS}$, terminate, else increment $i$ and repeat.

Note that:
- The vocabulary is finite, so the pmf of $P_{W|H=h}$ is representable by a tractable $V$-dimensional vector.
- There’s no limit to the set of possible histories (any sequence of any number of words, so long as it does not end in EOS)

Notes on [[Categorical Distributions]]
## LM Parameterization
Parameterization is the design of mechanism to compute $P_{W|H}(w|h)$ for any choice of $(h,w)$. 

Methods of parameterization, ie, method of assigning probability to any $w\in\mathcal{W}$ given any $h\in\mathcal{W}^*$.
1. using the relative frequency of $h\ \circ\ \langle w \rangle$, as observed in a large corpus
2. informed by the count of $h\ \circ\ \langle w \rangle$, and of its subsequences, in a large corpus
3. using a *log-linear model* with features $\phi(h) \in \mathbb{R}^D$
4. using a *non-linear model* to map from $h$ directly to the (parameters of the) pmf.

In method 1 of relative frequency, data sparsity is a problem. Think what happens when we have not seen a given $h$ followed by some $w$ in the following:
$$
P_{W|H}(w|h) \overset{MLE}{=} \frac{\text{count}_{HW}(h,w)}{\sum_{o\in\mathcal{W}} \text{count}_{HW}(h,o)}
$$
Unavoidable truth about empirical methods: not seeing something
is not evidence of it not being possible.
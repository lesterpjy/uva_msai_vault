# Syntax and Morphology

words -> syntactic categories
phrases -> syntactic categories

Syntactic Constituency: group of words can behave as single units, or constituents.
Example
noun phrase: Harry the Horse, the Broadway coppers

Constituents appear in similar syntactic environments.


## Context-Free Grammars
A rewriting system with two types of symbols and a set of symbol-rewriting rules.
Symbols: terminals (words), nonterminals (word and phrasal categories)
Rules: $X \rightarrow \beta$ where $X$ is a nonterminal, and $\beta$ is any string of terminal or nonterminal symbols.
Example
![[cfg_example.png | 400]]

Arity (length of rule's RHS)
- unary: $A \rightarrow B$
- Binary: $X \rightarrow BC$
- n-ary: $X \rightarrow \beta_{1},\dots,\beta_{n}$
If the longest rule as arity $a$, we say the grammar has arity $a$.
_Chomsky normal form_ (CNF): rexpressed to give a grammar of arity $2$

A **derivation** is a sequence of strings, we can use CFGs to derive strings
Steps:
1. start from string $\langle S \rangle$
2. At each step rewrite the leftmost nonterminal $X$ by application of a rule.
3. Until only terminal remain.
If a string $w_{1}\dots w_{l}$ is derivable from $S$, we write $S \overset{*}{\Rightarrow} w_{1}\dots w_{l}$
![[cfg_derivation_example.png | 300]]

The derivation can be denoted with $\delta$, and this "derivability" can be written as $S \overset{\delta}{\Rightarrow} w_{1}\dots w_{l}$

## Probabilistic Context-Free Grammars
A probability distribution over the space of all derivations (including their yields) supported by a grammar.
![[prob_cfg_1.png | 400]]
![[prob_cfg_2.png | 400]]

Example
![[prob_cfg_example.png | 300]]

Generative Story
1. Starts with $D = \langle S \rangle$.
2. If all symboles in $D$ are terminal, stop. Else go to (3)
3. Condition on the left-most nonterminal symbol $v$ in the derivation, draw a RHS string $\beta$ with probability $P_{S|N}(\beta|v)$, replace $v$ in $D$ by $\beta$. Repeat from (2).

**Parameterization**
![[prob_cfg_parameterization.png | 400]]

Estimation with MLE
$$
\theta_{v\rightarrow\beta} = \frac{\text{count}(v \rightarrow \beta)}{\sum_{(v\rightarrow\gamma)\in R} \text{count}(v \rightarrow \gamma)}
$$
Probability of a sentence $w_{1:l}$: the sum over the space of all derivations that have $w_{1:l}$ as yield.
![[pcfg_sentence_prob.png | 400]]

Evaluation as Syntactic Parser
![[pcfg_evaluation.png | 400]]
The set of all derivations of a given sentence $w_{1:l}$ is referred to as a parse forest for $w_{1:l}$. 
Typically work with binary-branching trees (arity 2), then the number of trees for a sentence of $L$ words is the Catalan number:
$$
C_{L} = \frac{(2L)!}{(L+1)!L!}
$$
But we can exploit Markov assumption to avoid enumeration: CKY algorithm

CKY algorithm: construct a graph like view of the forest, compressed to size that is cubic in sentence length.
Limitations:
- Limited use of linguistic context due to generative formulation.
- The context-free assumption is not enough in general: some linguistic constructions violate it.
- Dynamic programming for PCFGs takes time that is cubic in sentence length.

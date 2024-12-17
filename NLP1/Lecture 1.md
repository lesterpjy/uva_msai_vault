![[dos.jpg]]
NLP: computational modelling of human language
Some applications: information retrieval, information extraction, Q&A, sentiment analysis

Linguistics
1. Morphology - structure of words
2. Syntax - way word form phrases 
3. Semantics - lexical semantics, compositional semantics: meaning of words, phrases
4. Pragmatics - meaning in context

Why is NLP hard
- ambiguity
	- Same strings can mean different things
- synonymy/variability
	- different strings can mean the same or similar things

Sentiment analysis
Input: document $d$, fixed set of classes $C = \{c_{1}, c_{1},\dots,c_{J}\}$
Output: predicted class $c \in C$

Classification task
For any document $x$ and any of the $C$ possible classes $y \in Y$, aim to learn to assign probability to $y$ given $x: P_{Y|X}(y|x)$
which we decompose with Bayes rule:
$$
P_{Y|X}(y|x) = 
\frac{P_{Y}(y)P_{X|Y}(x|y)}{\Sigma_{c=1}^C P_{Y}(c)P_{X|Y}(x|c)}
$$
to classify document $d$, we output the most probable class under the model:
$$
\hat{y} = \underset{c\in C}{argmax}\  P_{Y|X}(c|d)
$$

NB for the classification task: [[Naive Bayes]]

For evaluation of the NB (Bag of Words) model on the classification task: [[Model Evaluation#Classification task]]


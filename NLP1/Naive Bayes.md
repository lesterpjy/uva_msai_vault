for: Classification task
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
#### Naive Bayes Classifier
Problem: document $x$ is high dimensional, need large corpus to estimate a useful $P_{X|Y}(x|y)$.

NB assumption (conditional independence assumption): segment $x$ into $l$ 'feature values' (e.g. words) $\langle w_{1}, \dots ,w_{l}\rangle$ and assume these are independent of one another, given the class $y$. Essentially treating the document as a bag of words:
$$
P_{XY}(x, y) \overset{\mathrm{ind}}{=} 
P_{Y}(y) \prod_{i=1}^l P_{F|Y}(w_{i}|y)
$$
With Bayes rule, the document class given the document can then be obtained with
$$
P_{Y|X}(c|x) = 
\frac{P_{Y}(c) \prod_{i=1}^l P_{F|Y}(w_{i}|c)}{P_{X}(x)}
$$
and for any $x$, the denominator is constant, so
$$
\hat{y} = \underset{c\in C}{argmax}\  P_{Y}(c) \prod_{i=1}^l P_{F|Y}(w_{i}|c)
$$
In practice the maximum likelihood estimations are the frequencies in the documents:
$$
P_{Y}(c) \overset{MLE}{=} \frac{\text{count}_{Y}(c)}{N}
$$
$$
P_{F|Y}(w|y) \overset{MLE}{=} 
\frac{\text{count}_{YF}(y,w)}{\Sigma_{f\in \mathcal{F}}\ \text{count}_{YF}(c, f)}
$$
where $\mathcal{F}$ is all words in the training data (the vocabulary).

Laplace smoothing for handling data sparsity (think what happens when a word has not been seen in a class)
$$
P_{F|Y}(w|y) \overset{\text{add 1}}{=} 
\frac{\text{count}_{YF}(y,w) + 1}{ |\mathcal{F}|+ \Sigma_{f\in \mathcal{F}}\ \text{count}_{YF}(c, f)}
$$

Source of errors for Bag of Words:
- Negation: "Ridely Scott has never directed a bad film"
- Overfitting on training data
- Documents with comparisons and contrast

For evaluation of the NB (Bag of Words) model on the classification task: [[Model Evaluation#Classification task]]
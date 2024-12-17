### Classification task
Precision: % of selected items that are correct (how much of the captured are positives)
Recall: % of correct items that are selected (how much of the positives are captured)
![[precision_recall.png]]

F-score (harmonic mean of precision and recall)
$$
F_{1} = \frac{2PR}{P+R}
$$
### Intrinsic Evaluation
#### Perplexity
todo
We assess the average surprisal (negative log probability) that our model assigns to heldout texts {x(1), . . . , x(S)}:
$$
\frac{1}{S} \sum_{s=1}^S \log P_{X}(X^{(s)})
$$
For ease of interpretation, we re-express it in terms of perplexity per token, a measure of average confusion
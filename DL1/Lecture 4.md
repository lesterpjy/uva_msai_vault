
## Overfitting

Bias-Variance tradeoff
**Bias**
“The difference between an estimator’s expected value and the true value of the parameter being estimated”.
The bias error is an error from erroneous assumptions in the learning algorithm.

**Variance**
The amount that the estimate of the target function will change if different training data was used.
The variance is an error from sensitivity to small fluctuations in the training set.

Overfitting
Overfitted models perform poorly on new data from the same domain.
Low/zero training error is not automatically overfitting!
Only in combination with worse generalization as a function of training error.
Why overfitting happens?
1. Complexity / parameter count >> problem / data.
2. Overfitting especially common when dealing with co-occurrences.
3. Memorization (i.e., learning individual samples instead of their distribution).
4. Silly things you might have missed in your data

## Regularization
- Weight regularization needed to prevent overfitting
- $l_{2}$-regularization: Introduce objective term for minimizing weights
$$
w^* = \text{argmin}_{w} \mathcal{L} + \frac{\lambda}{2} \sum_{l}||w_{l}||_{2}^2
$$
- When using simple (stochastic) gradient descend, then $l_{2}$ regularization is the same as weight decay:
$$
w_{t+1} = (1-\lambda\eta_{t})w_{t} - \eta_{t}\nabla_{\theta}\mathcal{L}
$$
- $l_{1}$-regularization: use $l_{1}$ objective, introduces sparse weights
$$
w^* = \text{argmin}_{w} \mathcal{L} + \frac{\lambda}{2} \sum_{l}||w_{l}||
$$
- $l_{1}$-regularization added to the gradient descent update rule
$$
w_{t+1} = w_{t} - \eta_{t}\left(\nabla_{\theta}\mathcal{L} + \lambda\frac{ w_{t}}{\text{sign}(w_{t})}\right)
$$
- $l_{1}$-regularization lead to sparse weights

**Early stopping**: stop the training when test error increases but training loss continues to decrease. Can be counted to regularization as training steps are reduced.

**Dropout**: setting activations randomly to 0 during training with probability $p$ (mostly between 0:1 and 0:5)
- During test time, every activation is reweighted by $\frac{1}{p}$
- Reduces co-adaptations/-dependencies between neurons because none can solely depend on the other
- Neurons get more robust, reduces overfitting
- Effectively, a different network architecture is used every iteration. Testing can be seen as using model ensemble ($2^n$ models)

## Augmentation
By performing augmentation, we can prevent neural networks from learning or memorizing irrelevant patterns, essentially boosting overall performance.

Data augmentation = pre-defined invariance
Essentially a form of injecting prior knowledge to instill invariance: ie, a dog flipped vertically is still a dog, so a network should still predict that label.

Other forms of data augmentations
- Noise robustness: adding noise to weights (uncertainty), adding noise to outputs (label smoothing).
- Semi or self-supervised learning: introducing a particular form of prior belief about the solution.
- Multitask learning: Shared input and parameters – improve statistical strength. Requires statistical relationship between tasks.

## Normalization
![[normalization.png | 400]]
Data pre-processing
- Center data around 0 (activation functions are designed for that)
- Scale input variables to have similar diagonal covariances (not if features are differently important)
- De-correlate features if there is no inductive bias (e.g. sequence over time)

**Batch Normalization**
 Ensure Gaussian distribution of features over batches at every module input
 $$
\begin{equation}
\begin{aligned}
\mu_{j} &= \frac{1}{m} \sum_{i=1}^m x_{ij} \\
\sigma_{j}^2 &= \frac{1}{m} \sum_{i=1}^m(x_{ij}-\mu_{j})^2 \\
\hat{x}_{ij} &= \frac{x_{ij}-\mu_{j}}{\sqrt{ \sigma_{j}^2 + \epsilon }} \\
\hat{x}_{ij} &= \gamma \hat{x}_{ij} + \beta
\end{aligned}
\end{equation}
$$
1. Compute mini-batch mean ($i$ runs over mini-batch samples, $j$ over the feature dimensions)
2. Compute mini-batch variance
3. Normalize feature to $\hat{x}_{ij}\sim\mathcal{N}(0,1)$
4. Scale and shift input with trainable parameters $\gamma$ (variance) and $\beta$ (mean).

 - Helps the optimizer to control mean and variance of input distribution, and reduces effects of 2nd order between layers ) easier, faster learning
 - Acts as regularizer as distribution depends on mini-batch and therefore introduces noise
 - During testing, take a moving average of the last training steps and use those for $\mu_{j}$ and $\sigma_{j}^2$
Drawbacks of batchnorm
- Requires large mini-batch: memory intense (store batch stat), performance sensitive to batch size, discrepancy between training and test data, breaks training example independence.
- Awkward to use with RNN
- Bug prone

**Layer Normalization**
- statistics (mean and variance) are computed across all channels and spatial dimensions.
-  statistics are independent of the batch.
![[layernorm.png | 400]]

### Question 1.1
Given a decoder $f_{\theta}$, describe the steps needed to sample an image.

For a pixel $m$ in the image, sample from the prior distribution $p(z_{n})$, which is a Gaussian specified by $\mathcal{N}(0, \mathbf{I}_{D})$. The sampled $D$-dimensional $z_{n}$ can be mapped to the Categorical distribution $\mathbf{p}_{m}=(p_{m_{1}},\dots,p_{mk})$ with the learned decoder $f_{\theta}$. The largest event probability in the distribution, in the range $[1,k]$, can be selected as the intensity of the pixel. The sampled image is formed by repeating these steps for each pixel in the image.

### Question 1.2
Although Monte-Carlo Integration with samples from $p(z_{n})$ can be used to approximate $\log p(x_{n})$, it is not used for training VAE type of models, because it is inefficient. In a few sentences, describe why it is inefficient and how this efficiency scales with the dimensionality of z. (Hint: you may use Figure 2 in your explanation.)

As shown in the figure, the prior distribution $p(z)$ is centered around $(0,0)$ while the posterior distribution $(p(z|x))$ is peaked around $(1,1)$. This means that most samples drawn from $p(z)$ do not overlap in the blue posterior distribution $p(x)$, suggesting that many samples drawn will contribute little to approximating the integral. Specifically, the efficiency of this sampling approach diminishes exponentially relative to the dimensionality of $z$ because $p(x|z)$ is significant only in the region where $p(z)$ and $p(z|x)$ overlap, and this overlap decrease exponentially with the dimensionality of $z$.


### Question 1.3
Explain how you can see from Equation 10 that the right hand side has to be a lower bound on the log-probability $\log p(x_{n})$?

Equation 10:
$$
\log p(\mathbf{x}_{n}) - KL(q(\mathbf{z}_{n}|\mathbf{x}_{n})||p(\mathbf{z}_{n}|\mathbf{x}_{n})) = \mathbb{E}_{q(\mathbf{z}_{n}|\mathbf{x}_{n})}[\log p(\mathbf{x}_{n}|\mathbf{z}_{n})] - KL(q(\mathbf{z}_{n}|\mathbf{x}_{n})||p(\mathbf{z}_{n}))
$$
The right hand side must be a lower bound on the log probability $\log p(\mathbf{x}_{n})$ because KL divergence is a non-negative value. Proving this is simple with Jensen's inequality($*$) in the following:
$$
\begin{equation}
\begin{aligned}
-D(p||q) &= -\sum_{x}p(x) \left[\log \frac{p(x)}{q(x)}\right] \\
&= \sum_{x} p(x) \left[\log \frac{q(x)}{p(x)}\right] \\
&\overset{*}{\leq} \log \sum_{x} p(x) \frac{q(x)}{p(x)} \\
&= \log 1 \\
&= 0
\end{aligned}
\end{equation}
$$
Thus the left hand side of equation 10 is smaller than $\log p(\mathbf{x}_{n})$, and right hand size is a lower bound of $\log p(\mathbf{x}_{n})$.

### Question 1.4
Describe what happens to the ELBO as the variational distribution $q(z_{n}|x_{n})$ approaches the true posterior $p(z_{n}|x_{n})$?

Equation 10:
$$
\log p(\mathbf{x}_{n}) - KL(q(\mathbf{z}_{n}|\mathbf{x}_{n})||p(\mathbf{z}_{n}|\mathbf{x}_{n})) = \underbrace{\mathbb{E}_{q(\mathbf{z}_{n}|\mathbf{x}_{n})}[\log p(\mathbf{x}_{n}|\mathbf{z}_{n})] - KL(q(\mathbf{z}_{n}|\mathbf{x}_{n})||p(\mathbf{z}_{n}))}_{\text{ELBO}}
$$
As the variational distribution $q(\mathbf{z}_{n}|\mathbf{x}_{n})$ approaches the true posterior $p(\mathbf{z}_{n}|\mathbf{x}_{n})$, the KL divergence term on the left hand side approaches $0$, and thus it would cause the ELBO to approach the log-probability $\log p(\mathbf{x}_{n})$.

### Question 1.5
Explain shortly why the names reconstruction and regularization are appropriate for these two losses.

- $-\mathbb{E}_{q_{\phi}(\mathbf{z}_{n}|\mathbf{x}_{n})}[\log p_{\theta}(\mathbf{x}_{n}|\mathbf{z}_{n})]$ measures how well the decoder $p_{\theta}(\mathbf{x}_{n}|\mathbf{z}_{n})$ reconstructs the input $\mathbf{x}_{n}$ from the latent representation $\mathbf{z}_{n}$, which is sampled from the variational distribution $q_{\phi}(\mathbf{z}_{n}|\mathbf{x}_{n})$. As the variational distribution provides an approximation of the posterior that encodes $\mathbf{x_{n}}$ into the latent space $\mathbf{z}_{n}$, the expectation penalizes mismatch between the input data and its reconstruction, and can be viewed as the reconstruction loss.
- $D_{KL}(q_{\phi}(\mathbf{z}_{n}|\mathbf{x}_{n})||p_{\theta}(\mathbf{z}_{n}))$ regularizes the latent space by enforcing the variational distribution $q_{\phi}(\mathbf{z}_{n}|\mathbf{x}_{n})$ to be close to the prior $p_{\theta}(\mathbf{z}_{n})$. By aligning the learned posterior with the prior, this loss term ensures that the latent space does not overfit, and can be sampled from for generative purposes.

### Question 1.6
The above derivation of closed form expression for the regularization term requires Gaussian prior and variational distributions. Assume that we want to model the prior $p(\mathbf{z})$ with a more complex distribution — it is likely the closed form expression would not exist. Keeping in mind that $D_{KL}(q(\mathbf{z}|\mathbf{x})|p(\mathbf{z})) = \mathbb{E}_{z\sim q(\mathbf{z}|\mathbf{x})}\left[\log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})} \right]$, propose an alternative way of estimating the regularization term for a given sample $\mathbf{x}_{n}$.

If the closed form solution does not exist for the regularization term, it is possible to estimate its values using Monte-Carlo sampling in a similar manner to the reconstruction term. The KL divergence can first be rewritten as,
$$
\begin{equation}
\begin{aligned}
D_{KL}(q(\mathbf{z}_{n}|\mathbf{x}_{n})|p(\mathbf{z}_{n})) &= \mathbb{E}_{z\sim q(\mathbf{z}_{n}|\mathbf{x}_{n})}\left[\log \frac{q(\mathbf{z}_{n}|\mathbf{x}_{n})}{p(\mathbf{z}_{n})} \right] \\
&= \mathbb{E}_{z\sim q(\mathbf{z}_{n}|\mathbf{x}_{n})}\left[\log q(\mathbf{z}_{n}|\mathbf{x}_{n})-\log p(\mathbf{z}_{n}) \right] \\
&\approx \frac{1}{L}\sum_{l=1}^L \left[\log q(\mathbf{z}_{n}^{(l)}|\mathbf{x}_{n})-\log p(\mathbf{z}_{n}^{(l)})\right], \ \ \ \ \mathbf{z}_{n}^{(l)}\sim q(\mathbf{z}_{n}|\mathbf{x}_{n})
\end{aligned}
\end{equation}
$$
Similar to the reconstruction term, we can sample $L$ samples of $\mathbf{z}_{n}^{(l)}$ from the variational posterior, and evaluate the KL divergence with the Monte Carlo estimate shown above.

### Question 1.7
Passing the derivative through samples can be done using the reparameterization trick — the process of sampling $z$ directly from $\mathcal{N}(\mu(x),\Sigma(x))$ is commonly replaced by calculating $z = \Sigma(x)\epsilon+\mu(x)$, where $\epsilon\sim\mathcal{N}(0,1)$. In a few sentences, explain why the act of direct way of sampling usually prevents us from computing $\nabla_{\phi}\mathcal{L}$, and how the reparameterization trick solves this problem.

The process of sampling from $z$ directly from a the distribution $\mathcal{N}(\mu(x),\Sigma(x))$ prevents us from computing $\nabla_{\phi}\mathcal{L}$ because the non-deterministic operation of random drawing from $\mathcal{N}(\mu(x),\Sigma(x))$ is non-differentiable with respect to parameters $\phi$ of the variational distribution. The reparameterization trick solves this issue by isolating the stochasticity in the random variable $\epsilon$. $z$ now becomes a deterministic transformation of the random variable $\epsilon$, making $z$ differentiable with respect to $\phi$, through $\mu$ and $\Sigma$.
### Question 1.8

> [!figure] 
> ![[train_bpd.png | 400]] 
> ![[val_bpd.png | 400]]
> Figure 1: Tensorboard plots of training (top) and validation (bottom) bit per dimension curves.

### Question 1.9

> [!figure] 
> ![[epoch_0_samples_zdim20.png]] 
> ![[epoch_10_samples_zdim20.png]] 
> ![[epoch_80_samples_zdim20.png]]
> *Figure 2: 64 samples from the VAE model trained on MNIST at 0 (top), 10 (middle), 80 (bottom) epochs.

Some samples images generated from the VAE after 80 epochs appear to look like combinations of numbers or some are strokes that resembles numerics, but not actually numbers. This demonstrates the generative properties of the model, as number-like images are generated from the latent space.
### Question 1.10

> [!figure] 
> ![[vae_manifold.png | 400]] 
> Figure 3: Visualized manifold of VAE with a 2-dimensional latent space.


### Question 2.1

Configuration: 
```
General arguments for training
--------------------------------------------------
batch_size: 64
valid_ratio: 0.75
augmentations: [True, False]
pretrained: [True, False]
num_epochs: 30
train_strats: ['standard', 'FGSM', 'PGD']
visualise: False
epsilon_fgsm: 0.1
alpha_fgsm: 0.5
epsilon_pgd: 0.01
alpha_pgd: 2
num_iter_pgd: 10
```

| Experiment set | Pre-trained | Attack Strategy | Defense Strategy | Data Augmentations | Test Accuracy | Acc. Decrease (%) |
| -------------- | ----------- | --------------- | ---------------- | ------------------ | ------------- | ----------------- |
| Set 1          | True        | -               | -                | -                  | 0.92          | -                 |
|                | True        | FGSM            | -                | -                  | 0.4188        | 54.5              |
|                | True        | -               | FGSM             | -                  | 0.90          | 2.17              |
|                | True        | FGSM            | FGSM             | -                  | 0.588         | 34.7              |
|                | True        | -               | -                | True               | 0.93          | -                 |
|                | True        | FGSM            | -                | True               | 0.4244        | 54.4              |
|                | True        | -               | FGSM             | True               | 0.88          | 5.38              |
|                | True        | FGSM            | FGSM             | True               | 0.5504        | 37.5              |
|                | -           | -               | -                | -                  | 0.67          | -                 |
|                | -           | FGSM            | -                | -                  | 0.1384        | 79.3              |
|                | -           | -               | FGSM             | -                  | 0.62          | 7.5               |
|                | -           | FGSM            | FGSM             | -                  | 0.2288        | 63.1              |
|                | -           | -               | -                | True               | 0.68          | -                 |
|                | -           | FGSM            | -                | True               | 0.1304        | 80.8              |
|                | -           | -               | FGSM             | True               | 0.58          | 14.7              |
|                | -           | FGSM            | FGSM             | True               | 0.1848        | 68.1              |
| Set 2          | True        | -               | -                | -                  | 0.92          | -                 |
|                | True        | FGSM            | -                | -                  | 0.402         | 56.3              |
|                | True        | -               | FGSM             | -                  | 0.89          | 3.26              |
|                | True        | FGSM            | FGSM             | -                  | 0.556         | 37.5              |
|                | True        | -               | -                | True               | 0.92          | -                 |
|                | True        | FGSM            | -                | True               | 0.4156        | 54.8              |
|                | True        | -               | FGSM             | True               | 0.89          | 3.26              |
|                | True        | FGSM            | FGSM             | True               | 0.5588        | 37.2              |



Why would adding a random perturbation with size ϵ not have a similar effect as a FGSM perturbation of the same size?

A random perturbation of size $\epsilon$ is not guaranteed to move the input data in the opposite direction of the largest gradient descent. By following the sign of the gradient, FGSM ensures that even a small perturbation is highly effective at push the input example across the model's decision boundary, leading to misclassification.


Say we split the training data into two sets A and B, and train models A and B on those datasets, respectively. Then, we have two models trained for the same task but using different subsets of the same dataset. Now for an instance x in the test set, a perturbation built using the gradients from model A, will likely have a similar effect on model B, even though the models don’t share weights or the exact training data. What is likely to be the cause of this phenomenon?

This phenomenon can be attributed to adversarial transferability, which arise because different models that are trained on different subset of the same dataset tend to learn similarly oriented decision boundaries and latent representations for the same task. The two models are both optimizing for the same classification objective and learning from data that are drawn from the same underlying distribution. As a result the perturbation derived from model A will likely work on model B as well.


What is the effect on using data augmentation compared to using no data augmentation, how would you explain this effect?

Using data augmentation during training should lead to a model that is more robust to FGSM attacks. Theoretically, training with data augmentation should cause the model to rely on more generalizable patterns, and the learned boundaries in the loss landscape should be smoother and less vulnerable to small changes to the input. Practically, in the result above we observe that the accuracy differs across different training runs, in experiment set 1, augmentation led to a small decrease, while in experiment set 2, it led to a small increase. This suggest that while data augmentation generally improves robustness, each training run might resulting in variations in the final accuracy and the exact level of robustness to FGSM attacks, due to the more complex high-dimensional loss landscape of our deep network model.


### Question 2.2

Describe the tradeoff between defending against the attack or not, why does this tradeoff occur?

The result shows that with pretraining and without augmentation, defense with FGSM loss works the best, as the accuracy on adversarial test data decreased by only 34.67% and the a small trade-off of 2.17% on the unperturbed data. Results without pretraining is significantly worse, and results for pretraining with data augmentation is only slightly worse.

There is a tradeoff on the accuracy of the clean, unperturbed data when adversarial defense is incorporated in training. This behavior is explicitly observed in the results table of Q2.1, where we see the accuracy decreased by 2.17% without data augmentation, and by 5.38% with data augmentation. The competing objective in the adversarial loss means that the model is forced to optimize on both the clean data loss with $\alpha J(\mathbf{\theta},x,y)$ and on the perturbed data loss with $(1-\alpha)J(\mathbf{\theta}, x+\epsilon \text{sign}(\nabla_{x}J(\mathbf{\theta},x,y)))$. The $\alpha$ parameter in the loss function can be used to control this trade off between the competing objectives.
### Question 2.3


| Experiment set | Pre-trained | Attack Strategy | Defense Strategy | Data Augmentations | Test Accuracy | Acc. Decrease (%) |
| -------------- | ----------- | --------------- | ---------------- | ------------------ | ------------- | ----------------- |
| Set 1          | True        | -               | -                | -                  | 0.92          | -                 |
|                | True        | PGD             | -                | -                  | 0.4424        | 51.91             |
|                | True        | -               | PGD              | -                  | 0.88          | 4.35              |
|                | True        | PGD             | PGD              | -                  | 0.5032        | 42.82             |
|                | True        | -               | -                | True               | 0.93          | -                 |
|                | True        | PGD             | -                | True               | 0.4424        | 52.43             |
|                | True        | -               | PGD              | True               | 0.80          | 13.98             |
|                | True        | PGD             | PGD              | True               | 0.466         | 41.75             |
|                | -           | -               | -                | -                  | 0.67          |                   |
|                | -           | PGD             | -                | -                  | 0.1264        |                   |
|                | -           | -               | PGD              | -                  | 0.57          |                   |
|                | -           | PGD             | PGD              | -                  | 0.1352        |                   |
|                | -           | -               | -                | True               | 0.68          |                   |
|                | -           | PGD             | -                | True               | 0.1812        |                   |
|                | -           | -               | PGD              | True               | 0.60          |                   |
|                | -           | PGD             | PGD              | True               | 0.20          |                   |
| Set 1          | True        | -               | -                | -                  | 0.92          |                   |
|                | True        | PGD             | -                | -                  | 0.42          |                   |
|                | True        | -               | PGD              | -                  | 0.87          |                   |
|                | True        | PGD             | PGD              | -                  | 0.5028        |                   |
|                | True        | -               | -                | True               | 0.92          |                   |
|                | True        | PGD             | -                | True               | 0.432         |                   |
|                | True        | -               | PGD              | True               | 0.82          |                   |
|                | True        | PGD             | PGD              | True               | 0.4844        |                   |

In this implementation, how do "using an adversarial loss" and "adding adversarial examples to the batch" compare? Are they equivalent or different? Provide reasoning for your answer, including under what conditions they might align or diverge.

The two approaches should produce equivalent or comparable models if the loss weight parameter $\alpha$ on the adversarial loss is set to $0.5$. Since in the adding adversarial examples case, we are adding equal number of duplicates of the input data into the training batch, the importance weighting of the examples seen by the model is equivalent for perturbed and unperturbed inputs. If the parameter $\alpha$ deviates significantly from 0.5, the adversarial loss is penalizing perturbation or non-perturbation more significantly than the other, leading to deviation from the "adding adversarial example to batch" approach.

Describe a tradeoff between using FGSM and PGD. For each method, identify one advantage it has over the other.

A tradeoff between the two approaches is the computational cost and the strength of the adversarial attack:
- FGSM requires only a single step to generate adversarial example, which makes the approach computationally efficient and easy to implement.
- PGD takes multiple steps of gradient-based perturbations, which is computationally more involved, but produces stronger attacks that are more reliable in measuring a model's robustness to adversarial attacks.

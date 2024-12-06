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
The above derivation of closed form expression for the regularization term requires Gaussian prior and variational distributions. Assume that we want to model the prior $p(\mathbf{z})$ with a more complex distribution — it is likely the closed form expression would not exist. Keeping in mind that $D_{KL}(q(\mathbf{z}|\mathbf{x})|p(\mathbf{z})) = \mathbb{E}_{z\sim q(\mathbf{z}|\mathbf{x})}\left[\log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{x})} \right]$, propose an alternative way of estimating the regularization term for a given sample $\mathbf{x}_{n}$.



### Question 1.7







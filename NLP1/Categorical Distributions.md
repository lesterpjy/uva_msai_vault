### Tabular Case

With categorical distribution of categories $\{1,\dots,K\}$, the categorical distribution is denoted as,
$$
\text{Categorical} (\theta_{1},\dots,\theta_{K}) = \text{Categorical} (\theta_{1:K})
$$
where the parameters $\theta_{1},\dots,\theta_{K}$ specifies the categorical probability mass function, constrained on:
$$
\begin{equation}
\begin{aligned}
0 \leq \theta \leq 1\ \text{for any} \ j\in[K] \\

\sum\limits_{j=1}^K = 1
\end{aligned}
\end{equation}
$$
We can write $\mathbb{C}\sim\text{Categorical}(\theta_{1:K})$ and $P(C=j)=\theta_{j}$
$\theta_{1:K}$ is a vector where,
$$
\vec{\theta} = (\theta_{1},\dots,\theta_{K})^T \in \triangle_{K-1} \subset \mathbb{R}
$$
$\triangle_{K-1}$ is the probability simplex.

**Conditional probability distribution** (cpd)

$$
W|C=\text{positive} \sim \text{Categorical}(\theta_{1:V}^{(pos)})
$$
$$
W|C=\text{neutral} \sim \text{Categorical}(\theta_{1:V}^{(neu)})
$$
$$
W|C=\text{negative} \sim \text{Categorical}(\theta_{1:V}^{(\neg)})
$$
Tabular CPD: $P_{W|C}$ as a table:
> [!figure] ![[tabular_cpd.png]] 
> *Figure 1*: Tabular cpd

**Estimation**
Estimate the numerical values of the parameters $\vec{\theta}$ with Maximum Likelihood Estimation (MLE).
1. gather data $D = \{ (c_{n}, w_{n})_{n=1}^N \}$
2. Log likelihood function $L(\vec{\theta}|D) \overset{iid}{=} \sum_{n=1}^N \log \theta_{w_{n}}^{(c_{n})}$

The model of the data is given by the cpds: $W|C=\text{c} \sim \text{Categorical}(\theta_{1:V}^{(c)})$, example: $P(W=dog|C=\text{pos}) \sim \text{Categorical}(\theta_{dog}^{(pos)})$

We look for a local/global optimum of $L(\vec{\theta}|D)$. For the tabular model there is an exact solution:
$$
\theta_{w}^{(c)} = \frac{\text{count}_{CW}(c, w)}{\sum_{\theta=1}^V \text{count}_{CW}(c, \theta)}
$$
Example
> [!figure] ![[tabular_cat_dist_example.png]] 
> *Figure 2: Tabular categorical distribution complete example.



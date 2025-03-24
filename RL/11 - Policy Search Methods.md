## Why Stay Close to Previous Policies?

A fundamental idea in modern policy optimization is taking small, controlled steps when updating policies. This approach leads to more stable learning for several reasons:

1. Value estimates and returns can be noisy or inaccurate
2. Large policy changes might lead to catastrophic performance drops
3. We want to maximize improvement while minimizing the risk

Mathematically, a policy update can be viewed as:

- Current parameters: $\theta$
- New parameters: $\theta' = \theta + x$
- Goal: Find $x$ such that $\theta'$ has high expected reward while keeping the difference between $\pi_\theta$ and $\pi_{\theta'}$ small

## The Problem with Standard Policy Gradients

Standard ("vanilla") policy gradient methods find the direction of steepest improvement per unit of parameter change using the Euclidean norm in parameter space. This is mathematically represented as:

$$x^* = \arg\max_x J(\theta + x) \text{ s.t. } x^T x = c$$

Which can be approximated as:

$$x^* \approx \arg\max_x J(\theta) + (\nabla_\theta J(\theta))^T x \text{ s.t. } x^T x = c$$

The solution is proportional to the gradient: $x^* \propto \nabla_\theta J(\theta)$

However, this approach has several limitations:

- It's sensitive to parameterization (the same policy can be represented in different ways)
- Parameters may have different "meanings" (e.g., mean vs. variance in a Gaussian policy)
- A norm in parameter space doesn't accurately reflect how much the actual policy behavior changes
- Different parameters might have different sensitivities, making uniform step sizes inefficient
- Parameters may be correlated, but this is ignored in standard gradients

## Covariant Policy Representation with KL Divergence

Instead of measuring distance in parameter space, we can use the Kullback-Leibler (KL) divergence, which measures the difference between probability distributions:

$$D_{KL}(\pi||\pi') = \int_{-\infty}^{\infty} \pi(a) \log \frac{\pi(a)}{\pi'(a)} da$$

KL divergence has important properties for RL:

- It equals 0 when $\pi = \pi'$
- It's invariant to parameter transformations (covariant)
- It naturally captures how different two policies behave

The goal becomes: find the direction of maximal improvement per unit of KL divergence between policies.

## Natural Policy Gradient (NPG)

The Natural Policy Gradient, introduced by Kakade (2002) and expanded by Bagnell (2003), addresses the parameterization sensitivity issue.

Starting with the KL divergence between the current policy and a slightly modified one:

$$c = \mathbb{E}_s[D_{KL}(\pi(a|s;\theta)||\pi(a|s;\theta + x))] = \text{EKL}(x)$$

We can approximate this using a second-order Taylor expansion:

$$c \approx \frac{1}{2}x^T (\nabla^2_x \text{EKL}(x)) x$$

The Hessian matrix $F = \nabla^2_x \text{EKL}$ turns out to be the Fisher Information Matrix, which can be written as:

$$F = \mathbb{E}_{a,s} [(\nabla \log \pi_\theta(a|s)) (\nabla \log \pi_\theta(a|s))^T]$$

This matrix characterizes how sensitive the policy is to parameter changes. The natural gradient update rule becomes:

$$\theta_{t+1} = \theta_t + \alpha F^{-1} \nabla_{\theta_t} J(\theta_t)$$

The natural gradient $\tilde{\nabla}_\theta J(\theta) = F^{-1}\nabla_\theta J(\theta)$ has several important properties:

- It's invariant to parameterization
- It automatically adjusts step sizes based on parameter sensitivity
- It accounts for parameter correlations
- It never points more than 90° away from the steepest ascent direction, ensuring improvement

![[npg.png | 400]]
### Example: Bernoulli Policy

Consider a Bernoulli policy with a single parameter $\theta$: $p(A=1) = \theta, p(A=0) = 1-\theta$

The Fisher information matrix is a scalar: $$F(\theta) = \frac{1}{\theta} + \frac{1}{1-\theta}$$

When $\theta$ is close to 0 or 1, $F(\theta)$ becomes large (around 11 for $\theta=0.1$ or $\theta=0.9$). When $\theta=0.5$, $F(\theta)=4$.

This means the natural gradient automatically takes smaller steps when $\theta$ approaches 0 or 1, which makes sense because small changes in $\theta$ have larger effects on the policy in these regions.

## Trust Region Policy Optimization (TRPO)

TRPO, developed by Schulman et al. (2015), builds on the natural gradient concept but takes a different theoretical approach. The key insight is defining a "trust region" where policy approximations are valid.

The theoretical foundation of TRPO comes from maximizing a lower bound on policy improvement while constraining the policy change:

1. Approximate the return function
2. Apply a KL-based penalty term to yield a lower bound
3. Maximize this lower bound within a constraint

In practice, TRPO maximizes a linearized approximation of the objective under a KL constraint:

$$\max_{\theta'} \mathbb{E}_s[\mathbb{E}_a[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A^{\pi_{\theta}}(s,a)]] \text{ s.t. } \mathbb{E}_s[D_{KL}(\pi_{\theta}(\cdot|s)||\pi_{\theta'}(\cdot|s))] \leq \delta$$

The connection to natural gradients becomes clear when we write the update as $x = \beta\tilde{\nabla}J$ and solve for $\beta$ using the KL constraint:

$$c \approx \frac{1}{2}\beta^2(\tilde{\nabla}J)^T F(\tilde{\nabla}J)$$

Solving for $\beta$: $$\beta = \sqrt{\frac{2c}{(\tilde{\nabla}J)^T F(\tilde{\nabla}J)}}$$

The key advantages of TRPO over NPG:

- Step size is automatically determined from the KL constraint
- Larger policy updates while still ensuring stability
- Strong empirical performance on complex tasks

## Advantages and Limitations

### Natural Policy Gradient:

- **Advantages**: Needs less training than regular policy gradients; Parameter-invariant
- **Limitations**: Requires computing the Fisher information matrix; Computationally costly to invert; Inherits high variance issues from standard policy gradients

### TRPO:

- **Advantages**: Even larger stable steps than NPG; Automatic step size adaptation; Strong theoretical guarantees
- **Limitations**: Computationally expensive; Complex implementation; Approximations needed for practical use with neural networks

## Conclusion

Both NPG and TRPO represent significant advances in policy optimization by addressing the fundamental question: how can we take large improvement steps while ensuring stability?

The key insights are:
1. Policy distances should be measured in behavior space (using KL), not parameter space
2. Policy updates should account for parameter sensitivities and correlations
3. Trust regions or constraints provide a principled way to determine step sizes

These methods form the foundation for many modern deep reinforcement learning algorithms and have been crucial for applying RL to complex control problems.

![[methods_summary.png | 500]]
## Need to know

- Advantage of covariant representation of distances?
- Advantage of specifying constraint instead of stepsize?
- Why do we need a constraint / penalty / stepsize?
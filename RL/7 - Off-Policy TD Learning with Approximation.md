
**Off-Policy TD Learning with Function Approximation**

In standard _on-policy_ TD learning, the agent’s behavior policy is the same as the target policy. In _off-policy_ learning, we collect data (transitions) under a _behavior policy_ $b$, while we want to learn about or improve a _target policy_ $\pi$. This introduces a mismatch between the distribution of visited state-action pairs and the distribution under the target policy. **Importance sampling** weights $\rho_t$ compensate for this mismatch.

When we also use function approximation (e.g., linear or nonlinear), we combine these ideas into **off-policy TD learning with approximation**. Below is a more detailed treatment, starting from the fundamental concepts up to the known risks of divergence.

---
## On-policy control with approximation

General gradient-descent update for action value:
$$
\begin{equation}
\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha\left[U_{t} - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})\right] \nabla \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})
\end{equation}
$$
where the update target $U_{t}$ can be any approximation of $q_{\pi}(S_{t}, A_{t})$. For example, the _episodic semi-gradient one-step Sarsa_:
$$
\begin{equation}
\mathbf{w}_{t+1} = \mathbf{w}_{t} + \alpha\left[R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_{t}) - \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})\right] \nabla \hat{q}(S_{t}, A_{t}, \mathbf{w}_{t})
\end{equation}
$$
If the action set is discrete and not too large, then the control method can be formed by coupling the above action-value prediction method with existing policy improvement and action selection methods. Such as an on-policy with a $\epsilon$-greedy policy:

![[epi-semi-grad-sarsa.png]]


## Off-policy control with approximation

Challenge: 1. target of the update, 2. distribution of the updates.
1. Target of the update (not target policy, but the target of update at that step): dealt with using importance sampling. Increase variance, but necessary for tabular and approximate.
2. Distribution of updates: necessary for function approximation because distribution of update in off-policy $\neq$ on-policy distribution, and on-policy distribution is important to stability of semi-gradient methods.

### Semi-Gradient Methods
These methods address the first part of the challenge, but not the second part. As such, these methods may diverge in some cases. However, these methods are guaranteed to stable and asymptotically unbiased for the tabular case, so it may still be possible to combine them with feature selection methods in such a way that the combined system could be assured stable.

#### Behavior vs. Target Policy, and the Importance Sampling Ratio

We assume we have:

- A behavior policy $b(a \mid s)$, which is the policy that _generates the data_ $\{(S_t, A_t, R_{t+1}, S_{t+1}), \dots\}$.
- A target policy $\pi(a \mid s)$, which is the policy _whose value function_ we want to estimate or improve.

Mathematically, when an action $A_t$ is drawn from $b(\,\cdot \mid S_t)$, the probability that $\pi$ would also have selected that action is $\pi(A_t \mid S_t)$. The ratio
$$
\rho_t \;\;=\;\; \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}
$$
is called the **importance sampling ratio**. If $\rho_t > 1$, that means the sampled action was _less likely_ under $b$ than under $\pi$ (so we “up-weight” that sample). If $\rho_t < 1$, the sampled action was _more likely_ under $b$ (so we “down-weight” that sample).

In standard off-policy learning theory, we usually require _coverage_, meaning if $\pi(a \mid s) > 0$, then we need $b(a \mid s) > 0$ as well, so $\rho_t$ is finite.
#### Approximating Value Functions

We introduce a parametric function $\hat{v}(s, \mathbf{w})$ to approximate the state-value function $v_\pi(s)$. A common choice is **linear approximation**:
$$
\hat{v}(s, \mathbf{w}) \;=\; \mathbf{w}^\top \boldsymbol{\phi}(s),
$$
where $\boldsymbol{\phi}(s)$ is a fixed feature vector for state $s$, and $\mathbf{w}$ is a parameter vector to be learned. Similarly, if we approximate action-values, we write:
$$
\hat{q}(s, a, \mathbf{w}) \;=\; \mathbf{w}^\top \boldsymbol{\phi}(s,a).
$$
Our goal is to adjust $\mathbf{w}$ so that $\hat{v}(\cdot, \mathbf{w})$ (or $\hat{q}(\cdot,\cdot,\mathbf{w})$) is as close as possible to the true $v_\pi(\cdot)$ (or $q_\pi(\cdot,\cdot)$).

#### Semi-Gradient TD(0) Update (Off-Policy)

Recall that with ordinary _on-policy_ TD(0) for state-value prediction, the update for parameter $\mathbf{w}$ at time $t$ is (in a semi-gradient form):
$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t \;+\; \alpha \,\bigl[R_{t+1} + \gamma \,\hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)\bigr]\, \nabla_{\mathbf{w}}\,\hat{v}(S_t, \mathbf{w}_t).
$$
To make it _off-policy_, we multiply by the importance sampling ratio $\rho_t$. Thus, we have:
$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t \;+\; \alpha\,\rho_t \bigl[R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t)\bigr]\, \nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t).
$$
Here:

1. $\alpha$ is the learning rate (step-size).
2. $\rho_t = \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$.
3. $\hat{v}(S_t, \mathbf{w}_t)$ is the _current estimate_ of the value for $S_t$.
4. $\nabla_{\mathbf{w}}\hat{v}(S_t, \mathbf{w}_t)$ is the gradient of the value function approximation with respect to parameters $\mathbf{w}$.

Because we use the semi-gradient, we only differentiate the _estimated_ part $\hat{v}(S_t, \mathbf{w}_t)$, not the _bootstrapped_ part that involves $\hat{v}(S_{t+1}, \mathbf{w}_t)$.
#### Action-Value Approximation Variant

Often, we want to learn an _action-value function_ $q_\pi(s,a)$ off-policy. One example is semi-gradient _off-policy SARSA_ or _Q-learning_ with function approximation. The general shape of the update is similar:

$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t \;+\; \alpha\,\rho_t\, \bigl[R_{t+1} + \gamma\,\hat{q}(S_{t+1},A_{t+1},\mathbf{w}_t) - \hat{q}(S_t, A_t,\mathbf{w}_t)\bigr]\, \nabla_{\mathbf{w}}\hat{q}(S_t, A_t,\mathbf{w}_t)
$$

An alternative (like in Q-learning) might replace $\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}_t)$ with $\max_{a}\hat{q}(S_{t+1},a,\mathbf{w}_t)$. However, that generally does _not_ preserve exact off-policy _policy evaluation_ of $\pi$ but rather aims to improve a greedy policy (leading to a control method).
#### Where Does the Importance Sampling Ratio Come From?

The intuitive reason is: we want an _unbiased_ sample of the update as if we were following $\pi$. However, our samples are _actually_ coming from $b$. The expected TD update under \pi can be matched by weighting each sample by $\frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$. Symbolically, for any function $f$,
$$
\mathbb{E}_{\substack{S_t \sim d^\pi \\ A_t \sim \pi}} [f(S_t, A_t)] \;=\; \mathbb{E}_{\substack{S_t \sim d^b \\ A_t \sim b}} \!\Bigl[\rho_t \,f(S_t, A_t)\Bigr],
$$
where $d^\pi$ is the stationary distribution of states under policy $\pi$, and $d^b$ is that of $b$. Hence, the ratio $\rho_t$ _corrects_ for the difference in distributions between $\pi$ and $b$.
#### Gradient Derivation (Sketch)

Using **linear** function approximation for the state-values, $\hat{v}(s,\mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)$, the gradient is:
$$
\nabla_{\mathbf{w}}\,\hat{v}(S_t,\mathbf{w}_t) \;=\; \boldsymbol{\phi}(S_t).
$$
So the update rule is:
$$
\mathbf{w}_{t+1} \;=\; \mathbf{w}_t \;+\; \alpha\,\rho_t \,\delta_t \,\boldsymbol{\phi}(S_t),
$$
where
$$
\delta_t \;=\; R_{t+1} \;+\; \gamma\, \mathbf{w}_t^\top \boldsymbol{\phi}(S_{t+1}) \;-\; \mathbf{w}_t^\top \boldsymbol{\phi}(S_t).
$$
The ratio $\rho_t$ simply multiplies this TD error $\delta_t$, ensuring the update’s expectation matches the _target-policy_ distribution rather than the _behavior-policy_ distribution.

---
### Examples of Off-policy Divergence

Off-policy + Bootstrapping + Function Approximation can lead to divergence.
#### A simple example

![[a_simple_example_off-policy_divergence .png | 150]]

- Because the approximate values for $S_0$​ and $S_1$​ are _coupled_ (they both depend on $w$), repeated updates from the "wrong" distribution of transitions (the behavior policy) can push $w$ away from any stable solution.
- "Wrong" distribution here refers to the fact that in off-policy training, the behavior policy might select actions on transitions out of $S_{1}$ which the target policy never would. In such a case $\rho_{t}$ is zero ($\frac{\pi(A_{t}|S_{t})}{b(A_{t}|S_{t})} = 0$ ), and no updates would be made to $w$.
- If the transition $S_{0} \rightarrow S_{1}$, which increases $w$ because $\hat{v}(S_{1},w) > \hat{v}(S_{0},w)$, keeps occurring without the other transitions (such as one that transit out of $S_{1}$ to reduce $w$) making updates to $w$, $w$ is not stable and can diverge to infinity.

#### Baird's counterexample

![[bairds_counterexample.png]]

- **Two actions**: a “dashed” action (behavior policy might choose it with high probability) and a “solid” action (the target policy always takes this action).
- This means:
	- behavior policy chooses the dashed action almost always, which cycles through the states in a certain pattern.
	- target policy only choose the solid action.
	- Mismatch leads to a very large or very small _importance sampling ratio_ $\rho_{t}$
- Reward is 0 on all transitions: true value of the function is 0 for all $s$, which is representable exactly with $\mathbf{w} = \mathbf{0}$. In fact there are many solutions since $\mathbf{w} \in \mathbb{R}^8 > \text{no. of states}$, and set of feature vectors is a linearly independent set. This is all to show that representation of the exact solution is in principle favorable. 
- If semi-gradient TD(0) is applied to this problem, the weights diverge. Even for semi-gradient DP, the weights diverge, ie, not caused by randomness of sampling.

Does this mean off-policy learning with function approximation is always hopeless?  No. In many practical cases, it can work if:
- The coverage of the behavior policy is not too drastically different from the target policy (as is the case for Q-learning with $\epsilon$-greedy policy)
- The features are well-chosen (and sometimes the dimension is large enough to represent the needed values),
- Or one uses **convergent** algorithms such as **Gradient TD** or **Emphatic TD** methods introduced in research literature.

---

### The “Deadly Triad” and Risk of Divergence

**off-policy TD learning can diverge** when combined with:

1. **Function approximation** (especially linear or non-linear),
2. **Bootstrapping** (like in TD(0), where we update from our own estimates $\hat{v}(S_{t+1}, \mathbf{w}_t)$),
3. **Off-policy data** (i.e., $\rho_t$ can be large or unbounded).

These three together are called the **deadly triad**. The danger is not due to _control_ or _generalized policy iteration_ (instability arise in simpler prediction as well), nor is it due to _learning_ or _uncertainties about the environment_ (also arise in DP).

### Linear Value-function Geometry

![[linear-value-func-geometry.png]]

Policy evaluation adds the Bellman error to the previous function, represented by:
- $B_{\pi}v_{\mathbf{w}}$, where the Bellman operator is applied to the value func $v_{\mathbf{w}}$
- TD(0) on average follows the same path (gray arrows)
- However, this brings the value function out of the representable subspace (above plane parameterized by $w_{1}$ abd $w_{2}$).


### Gradient Descent in the Bellman Error

SGD: updates are made that in expectation are equal to the negative gradient of an objective function. Monte Carlo methods are true SGD methods, converge robustly for:
- on-policy, off-policy, nonlinear (differentiable) function approximators
- slower than semi-gradient methods with bootstrapping (not SGD methods), which may diverge.


### Bellman Error is not Learnable

![[objectives-distributions-mdps.png]]


### Gradient TD Methods






# Off-policy TD Learning with Approximation

## Introduction to the Problem

This lecture explores the challenges of combining three elements that create what's known as the "deadly triad" in reinforcement learning:

1. Function approximation
2. Bootstrapping (TD methods)
3. Off-policy learning

The lecture identifies that while each of these components is useful on its own, their combination can lead to instability and divergence in learning algorithms.

## Review of Semi-Gradient TD Learning

In previous lectures, semi-gradient TD methods were introduced for prediction (value function estimation) and control. For prediction, the update rule is:

$$w_{t+1} \leftarrow w_t + \alpha [R + \gamma\hat{v}(S', w_t) - \hat{v}(S_t, w_t)] \nabla\hat{v}(S_t, w_t)$$

And for control (SARSA), the update rule is:

$$w_{t+1} \leftarrow w_t + \alpha (R_t + \gamma\hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)) \nabla\hat{q}(S_t, A_t, w_t)$$

The term "semi-gradient" indicates that these methods only partially account for how the target values depend on the weight vector $w$.

## Off-Policy Learning with Importance Sampling

The lecture then asks if semi-gradient TD methods can be extended to off-policy learning, where we learn about one policy (the target policy) while following a different policy (the behavior policy).

Off-policy learning uses importance sampling with a ratio:

$$\rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$$

Where:

- $\pi$ is the target policy (the policy we want to learn)
- $b$ is the behavior policy (the policy we're actually following)
- $\rho_t$ is the importance sampling ratio

This ratio gives higher weight to transitions that are more likely under the target policy than the behavior policy, and lower weight to transitions that are more common under the behavior policy.

## Applying Importance Sampling to TD

The natural extension of semi-gradient TD to off-policy learning would be:

$$w_{t+1} \leftarrow w_t + \alpha\rho_t (R_{t+1} + \gamma\hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t)) \nabla\hat{v}(S_t, w_t)$$

Where $\rho_t$ is the importance sampling ratio.

## Instability in Off-Policy TD with Function Approximation

To illustrate the potential instability, the lecture presents a simple two-state example where the value function is approximated with a simple linear function.

The example shows that:
1. With on-policy learning, the TD updates tend to shrink the TD error over time
2. However, with off-policy learning (especially when $\rho \ll 1$), the TD error can actually grow without bound

This demonstrates that the off-policy semi-gradient TD method can diverge, particularly when the behavior policy is very different from the target policy.

## The Deadly Triad

The lecture emphasizes that this divergence is not due to:

- The value function being unrepresentable
- Random effects
- Dependent features

Rather, it's the combination of three elements:
1. Function approximation (inability to represent arbitrary value functions)
2. Semi-gradient bootstrapping (using current estimates as targets)
3. Off-policy training (learning about a policy different from what generated the data)

This combination is referred to as the "deadly triad" because it can lead to instability and divergence.

## Value Function Geometry

To better understand the issues, the lecture introduces a geometric perspective on value functions. For a system with $n$ states, the value function can be viewed as a point in an $n$-dimensional space. However, with function approximation (especially linear), we can only represent a lower-dimensional subspace of all possible value functions.

In this geometric view:

- The true value function $v_\pi$ is a point in the full state-space
- Our function approximation creates a subspace of representable functions
- TD methods try to move within this subspace to minimize some form of error

![[linear-value-func-geometry.png | 400]]

## Different Types of Errors

1. **Value Error (VE)** - The mean squared difference between the true value and the approximated value: $$\text{VE}(w) = \sum_{s \in S} \mu(s) [v_\pi(s) - \hat{v}(s, w)]^2$$
2. **TD Error (TDE)** - The mean squared TD error: $$\text{TDE}(w) = \sum_{s \in S} \mu(s) \mathbb{E}[\delta_t^2 | S_t = s, A_t \sim \pi]$$ where $\delta_t = R_{t+1} + \gamma\hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t)$
3. **Bellman Error (BE)** - The mean squared expected TD error: $$\text{BE}(w) = \sum_{s \in S} \mu(s) \mathbb{E}[\delta_t | S_t = s, A_t \sim \pi]^2$$
![[geometry_bellman.png | 400]]

Note that at $v_{\pi}$, TDE $\delta$ is not zero (small green vectors), but they cancel out in expectation. Blue trajectory is not followable with approximation. Need to project into representable space.

![[still_bellman_error.png | 400]]

![[projection_geometry.png | 400]]


1. **Projected Bellman Error (PBE)** - The Bellman error projected onto the space of representable functions: $$\text{PBE}(w) = \sum_{s \in S} \mu(s) (\Pi\mathbb{E}[\delta_t | S_t = s, A_t \sim \pi])^2$$ where $\Pi$ is the projection operator onto the space of representable functions
    

## Issues with These Error Measures

The lecture examines problems with minimizing each error type:

2. **TD Error (TDE)**: Using the A-split example (a simple MDP with states A, B, C), the lecture shows that minimizing TDE can lead to poor value approximations. The optimal value at state A would be the average of the optimal values at B and C, which might not be desirable.
    
3. **Bellman Error (BE)**: While minimizing BE seems reasonable, it can't be done directly from samples. The lecture demonstrates this with examples of MDPs that generate the same sample transitions but have different optimal values, making it impossible to determine the correct BE from samples alone.
    

## Gradient TD Methods

The lecture then presents gradient TD methods that can safely perform off-policy learning with function approximation. These methods minimize the Projected Bellman Error (PBE) and include:
1. **GTD2**: A true gradient-descent method with update rules:
    - For a secondary weight vector $v$: $v_{t+1} = v_t + \beta\rho_t (\delta_t - v_t^T x_t) x_t$
    - For the main weight vector: $w_{t+1} = w_t + \alpha\rho_t (x_t - \gamma x_{t+1}) x_t^T v_t$
    - converge to 0 mean squared projected Bellman error with linear features
    - with non-linear feature approx, converge to local optimum
2. **TDC** (Temporal Difference with Correction): An improved version of GTD2

These methods are stable because they perform true gradient descent on the PBE, unlike semi-gradient methods.

## Convergence Properties

![[convergence_properties_td.png]]

1. **Gradient Monte Carlo**:
    - Converges globally to minimize VE
    - Works with all types of function approximation and policy settings
2. **Semi-gradient TD**:
    - Converges to minimize PBE in the on-policy case with linear function approximation
    - Can diverge in off-policy scenarios
    - Often faster than gradient TD methods when it does converge
3. **Gradient TD (GTD2, TDC)**:
    - Converges to minimize PBE in all cases with linear function approximation
    - Converges to local optima with non-linear function approximation
    - Stable even with off-policy learning
    - Requires additional parameters and computations
4. **LSTD (Least Squares TD)**:
    - Converges to minimize PBE
    - Not applicable to non-linear function approximation
    - More data-efficient but computationally expensive

## Key Takeaways
1. The combination of function approximation, bootstrapping, and off-policy learning (the deadly triad) can lead to instability.
2. Standard semi-gradient TD methods can diverge in off-policy scenarios with function approximation.
3. True gradient TD methods (like GTD2 and TDC) provide stable alternatives by minimizing the projected Bellman error.
4. Different error measures (VE, TDE, BE, PBE) represent different objectives in RL, and each has its advantages and limitations.
5. Understanding the geometry of value functions helps explain why certain methods work or fail in different scenarios.

## Need to know

What is the deadly triad?
What are the VE, TDE, BE, PBE and how can we think of them
in terms of value function geometry?
Which of these errors should be minimized by a gradient TD
method and why?
How do the properties of GMC, semi-gradient TD, gradient TD,
and LSTD compare?
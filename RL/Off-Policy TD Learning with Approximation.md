
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



### Practical Algorithmic Steps

A typical algorithmic loop for semi-gradient off-policy TD(0) with state-values is:

2. Initialize parameters w\mathbf{w} (e.g., all zeros or small random values).
3. **For each time step** tt:
    - Observe current state StS_t.
    - Choose action AtA_t according to the behavior policy b( ⋅∣St)b(\,\cdot\mid S_t).
    - Execute AtA_t, observe next state St+1S_{t+1} and reward Rt+1R_{t+1}.
    - Compute importance sampling ratio: ρt  =  π(At ∣ St)b(At ∣ St).\rho_t \;=\; \frac{\pi(A_t \,\mid\, S_t)} {b(A_t \,\mid\, S_t)}.
    - Compute TD error: δt  =  Rt+1  +  γ v^(St+1,w)  −  v^(St,w).\delta_t \;=\; R_{t+1} \;+\; \gamma\,\hat{v}(S_{t+1}, \mathbf{w}) \;-\; \hat{v}(S_t, \mathbf{w}).
    - Update parameters (semi-gradient): w  ←  w  +  α  ρt  δt  ∇wv^(St,w).\mathbf{w} \;\leftarrow\; \mathbf{w} \;+\; \alpha\;\rho_t\;\delta_t\;\nabla_{\mathbf{w}}\hat{v}(S_t,\mathbf{w}).
4. Continue until convergence or for a predetermined number of episodes.

If we are dealing with action-values q^(s,a,w)\hat{q}(s,a,\mathbf{w}), we replace v^(St,w)\hat{v}(S_t,\mathbf{w}) with q^(St,At,w)\hat{q}(S_t,A_t,\mathbf{w}) and similarly for the next state-action pair.

---

### 9. Summary of Key Concepts

- **Behavior Policy  b\,b, vs. Target Policy  π\,\pi**: Off-policy means we learn about π\pi from data generated by bb.
    
- **Importance Sampling Ratio  ρt\,\rho_t**:
    
    ρt  =  π(At∣St)b(At∣St),\rho_t \;=\; \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)},
    
    which re-weights each transition to reflect what would have happened under π\pi.
    
- **Off-Policy Semi-Gradient TD(0)**:
    
    wt+1  =  wt  +  α ρt[Rt+1  +  γ v^(St+1,wt)  −  v^(St,wt)]∇w v^(St,wt).\mathbf{w}_{t+1} \;=\; \mathbf{w}_t \;+\; \alpha \,\rho_t \Bigl[ R_{t+1} \;+\; \gamma\,\hat{v}(S_{t+1}, \mathbf{w}_t) \;-\; \hat{v}(S_t, \mathbf{w}_t) \Bigr] \nabla_{\mathbf{w}}\,\hat{v}(S_t, \mathbf{w}_t).
- **Potential for Divergence**: The _deadly triad_ of (1) function approximation, (2) bootstrapping, and (3) off-policy data can cause the parameters to diverge or behave unstably, unless we use special care (e.g. smaller step-sizes, truncated ρt\rho_t, or specialized algorithms like Emphatic TD).
    

---

### 10. References

- **Sutton, R.S. and Barto, A.G.**: _Reinforcement Learning: An Introduction_, especially Chapters 10–11 for off-policy methods and function approximation.
- **Lecture slides** on off-policy TD approximation (slides by Herke van Hoof, or your local RL course slides).
- For **emphatic TD**, see Sutton & Barto Chapter 11 for a deeper look at alternative algorithms that address divergence.

---

These notes incorporate more detailed explanations, derivations, and the underlying mathematics of **off-policy TD learning with approximation**. All key symbols and equations are presented in LaTeX form to facilitate direct transfer into your Obsidian notebook.
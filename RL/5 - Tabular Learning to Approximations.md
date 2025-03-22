## The Big Picture of Reinforcement Learning Methods

Reinforcement learning algorithms can be broadly categorized based on whether they have access to the underlying MDP dynamics or only to sampled data:

1. **When MDP dynamics are known** (having access to transition probabilities $p(s',r|s,a)$):
    - Dynamic programming methods like Policy Iteration and Value Iteration can be used
2. **When only sampled data is available** (having a dataset $D = {(s_i, a_i, r_i, s'_i)}_{i=1...N}$):
    - Monte Carlo methods (learn from complete episodes)
    - Temporal Difference methods like TD(0), SARSA, and Q-learning (learn from partial episodes)

All these methods aim to learn the value function directly without learning the transition model.

## Recap of Key Algorithms: Q-learning and SARSA

### Q-learning (Off-policy TD Control)

Q-learning follows this update rule: $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

Key characteristics:

- It's an off-policy algorithm (the behavior policy can differ from the target policy)
- The behavior policy doesn't need to converge to greedy
- In the target calculation, we use a greedy policy ($\max_{a'}$) regardless of what action is actually taken next
- Q-learning converges to $q^*$ under usual conditions (appropriate step-size schedule and sufficient exploration)
- At termination, $Q(s',a')$ is defined to be 0

### SARSA (On-policy TD Control)

SARSA follows this update rule: $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

Key characteristics:

- It's an on-policy algorithm (learns about the policy being followed)
- For SARSA to converge to $q^*$, the policy needs to converge to a greedy policy (e.g., by gradually reducing $\epsilon$ in $\epsilon$-greedy)
- At termination, $Q(s',a')$ is defined to be 0

## Why No Importance Sampling in Off-policy Q-learning?

An interesting question arises: why doesn't Q-learning use importance sampling weights like off-policy Monte Carlo methods?

To understand this, let's revisit off-policy Monte Carlo learning: $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \rho_{t+1:T-1}[G_t - Q(S_t, A_t)]$$
In Monte Carlo methods:

- The (S,A) pairs don't change the function being learned
- However, the returns G (which define the targets) depend on the behavior policy
- We can't know what the return would have been under the target policy, so we must use the returns we have and correct them with importance weights

In Q-learning:

![[q-learning_no-importance-weights.png]]

- We still don't care which policy generated the (S,A) pairs
- What matters is that the target is appropriate for the Q-function we want to learn
- The target depends on the next state-action pair (red)
- Instead of taking an action from the behavior policy and correcting with importance weights, Q-learning takes actions directly from the target policy (the greedy policy)
- This is easier and possible because we can calculate the target for any alternative action

In short:

- If we know what the value-function target is under the target policy π, use it directly (no importance weights needed)
- If we don't know the target under π, use the target under behavior policy b and correct using importance weights

## Which TD Method to Choose?

The choice between on-policy methods like SARSA and off-policy methods like Q-learning depends on several factors:

|On-policy (SARSA)|Q-learning (Off-policy with greedy π)|
|---|---|
|Only needs sampled action in update|Considers all actions per update (more computationally expensive)|
|Specific case (b=π)|Can use any behavior policy|
|Only data gathered with current policy|Can reuse data, use data from other sources|
|Generally needs non-greedy policy|Uses greedy target policy|

General guideline:

- If you want a greedy target policy, use Q-learning
- If you want some other fixed target policy, use SARSA

## The Broader Perspective: MC and TD Methods

Both TD(0) and Monte Carlo methods have advantages:

- MC can quickly back-up values from a single episode
- TD(0) can exploit learned values at intermediate states

An intermediate approach called n-step TD can balance these advantages.

Main difference between methods:
- prediction (estimate V) vs. control (estimate Q)
- off-policy vs. on-policy
- "wide" vs. "narrow" updates
- "deep" vs. "shallow" updates

![[methods_comparison.png | 400]]
## The Problem with Tabular Methods and Need for Approximation

Tabular methods face several limitations:

1. The Q/V table might be too large to fit in memory
2. It might take too long to collect enough experiences to fill the table
3. The state space might be continuous (infinitely many states)

Fortunately, we can generalize our experience because similar states often have similar values. Function approximation allows us to:

- Represent the value function in a compact way
- Generalize experiences to nearby states
- Maintain approximation accuracy

This is similar to supervised learning:

- Very flexible functions might not generalize well (overfitting)
- Inflexible functions may not be expressive enough
- We need to balance expressivity with data availability

## Value Function Approximation

Instead of storing a separate value for each state, we can represent the value function as a parameterized function: $$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \phi(s) = \sum_i w_i \cdot \phi_i(s)$$Where:
- $\mathbf{w}$ is a vector of parameters
- $\phi(s)$ is a feature vector representing state $s$

A simple example is to divide states into groups (buckets) and use a one-hot encoding:
- $\phi_1(s) = 1$ if $s$ is in group one, 0 otherwise
- $\phi_2(s) = 1$ if $s$ is in group two, 0 otherwise, etc.

## Objective Function for Approximation

We want to find the best parameters $\mathbf{w}$ that minimize the error between the true and approximate value functions. The objective is:

$$VE(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) [v_\pi(s) - \hat{v}(s, \mathbf{w})]^2$$

Where $\mu(s)$ is a state distribution that weights the importance of different states. For on-policy learning, $\mu$ is the fraction of time spent in each state when following policy $\pi$.

For continuing tasks, this is the stationary distribution: $$\mu_\pi(s) = \sum_{\bar{s}}\sum_a p(s|\bar{s},a)\pi(a|\bar{s})\mu_\pi(\bar{s})$$For episodic tasks, we first find the average number of visits to each state per episode: $$\eta_\pi(s) = h(s) + \gamma \sum_{\bar{s}} \eta_\pi(\bar{s})\sum_a \pi(a|\bar{s})p(s|\bar{s},a)$$Then normalize to get the fraction: $$\mu_\pi(s) = \frac{\eta_\pi(s)}{\sum_{s'} \eta_\pi(s')}$$
## Gradient Monte Carlo Algorithm

To find the optimal parameters $\mathbf{w}$, we can use stochastic gradient descent:

$$\nabla_\mathbf{w} VE(\mathbf{w}) = \nabla_\mathbf{w} \sum_{s \in \mathcal{S}} \mu(s) [v_\pi(s) - \hat{v}(s, \mathbf{w})]^2$$
But how to approximate?
- As $\mu$ is on-policy distribution, simple pick random visited state to sample from $\mu(s)$
- Use the return $G$ to generate unbiased estimate of target $v_{\pi}$
$$
\begin{equation}
\begin{aligned}
\nabla_\mathbf{w} VE(\mathbf{w}) &\approx \nabla_\mathbf{w} \frac{1}{T}\sum_{t=1}^T (G_{t} - \hat{v}(s_{t}, \mathbf{w}))^2 \\&= -\frac{2}{T}\sum_{t=1}^T (G_{t} - \hat{v}(s_{t}, \mathbf{w}))\nabla_\mathbf{w}\hat{v}(s_{t}, \mathbf{w})
\end{aligned}
\end{equation}
$$
This leads to the update rule: $$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [G_t - \hat{v}(S_t, \mathbf{w}_t)] \nabla_\mathbf{w}\hat{v}(S_t, \mathbf{w}_t)$$
Where:

- $G_t$ is the return from state $S_t$, serving as an unbiased estimate of $v_\pi(S_t)$
- $\nabla_\mathbf{w}\hat{v}(S_t, \mathbf{w}_t)$ is the gradient of the approximate value function
- For linear approximation, this gradient is simply $\phi(S_t)$

The complete Gradient Monte Carlo algorithm:

![[gradient-mc.png | 500]]

## Worked Example: Steady State Distribution

A worked example of calculating the steady-state distribution for a simple 3-state system. 

![[steady-state-example.png | 500]]

1. Finding the values $\eta$ (number of visits to each state)
2. Solving a system of linear equations using matrix inverse
3. Normalizing to find the distribution $\mu$

## Key Takeaways
1. Understand the advantages of TD(0) and Monte Carlo methods
2. Know how tabular value-based methods can be categorized
3. Understand Gradient Monte Carlo and why function approximation is useful for handling large state spaces

The transition from tabular methods to function approximation is essential for tackling real-world reinforcement learning problems with continuous or high-dimensional state spaces.

## Need to know

What are advantages of TD(0) and Monte-Carlo
How can tabular value-based methods be categorized?
What is gradient MC and why is it useful?
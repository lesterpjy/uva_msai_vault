## Introduction to Monte Carlo Methods

Monte Carlo (MC) methods form a fundamental class of algorithms in reinforcement learning that learn directly from experience without requiring a model of the environment. Unlike dynamic programming, which we covered previously, Monte Carlo methods don't need the transition probabilities or reward functions to be known beforehand.

The key idea behind Monte Carlo methods is simple yet powerful: to evaluate a state, we average the returns (cumulative rewards) observed after visiting that state. By collecting enough samples and averaging them, we can approximate the expected return, which is precisely what the value function represents.

## Monte Carlo for Prediction

Monte Carlo prediction aims to estimate the value function $v_\pi(s)$ for a given policy $\pi$. The value function is defined as:

$$v_\pi(s) = \mathbb{E}_\pi[G_t|S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t = s\right]$$

This is the expected return starting from state $s$ and following policy $\pi$ thereafter. The beauty of Monte Carlo methods is that we can estimate this expectation by simply averaging the returns we observe.

### Key Characteristics of Monte Carlo Methods:

1. **Episodic Tasks**: MC methods are typically used for episodic tasks where experience is divided into episodes that eventually terminate. Examples include games won/lost or finding the exit in a maze.
2. **Complete Episodes**: MC methods require complete episodes before learning, as they update values based on the final return.
3. **First-visit vs. Every-visit**: In first-visit MC, we only consider the first time a state is visited in an episode for updates. In every-visit MC, we consider every visit to the state.

### Monte Carlo Algorithm for Prediction:

![[mc_prediction.png | 500]]

1. Initialize $V(s)$ arbitrarily for all states
2. Initialize an empty list Returns(s) for each state
3. For each episode:
    - Generate an episode following policy $\pi$
    - For each state $s$ in the episode:
        - Calculate the return $G$ following the first (or every) visit to $s$
        - Add $G$ to Returns(s)
        - Update $V(s)$ to be the average of Returns(s)

The algorithm loops backward through the episode, calculating returns and updating state values. This backward iteration enables us to efficiently compute returns using the recursive relationship.

## Monte Carlo vs. Dynamic Programming

The two approaches represent different strategies for solving reinforcement learning problems:

**Dynamic Programming**:
- Requires complete knowledge of the environment (transition probabilities)
- Looks one step ahead at all possible next states (full width)
- Has lower depth of calculation (one step)

**Monte Carlo**:
- Requires no prior knowledge of the environment
- Samples a single trajectory (narrow width)
- Looks all the way to the end of the episode (high depth)
- Only applicable to episodic tasks
- Has higher variance, especially with long episodes

## Monte Carlo for Control

Control refers to finding the optimal policy. While Monte Carlo prediction helps us evaluate a policy, Monte Carlo control aims to improve the policy iteratively.

The challenge with Monte Carlo control is that we need to ensure all state-action pairs are visited. There are several approaches to guarantee this:

### 1. Exploring Starts

One approach is to use "exploring starts," where we start each episode from a random state-action pair. This guarantees that all state-action pairs have a chance of being visited.

However, exploring starts may not be practical in many environments where we don't have control over the starting conditions.

### 2. On-Policy Methods with Soft Policies

An alternative is to use a policy that maintains some exploration. Instead of using a deterministic greedy policy, we use a "soft" policy like ε-greedy, which takes the greedy action with probability $1-\epsilon + \frac{\epsilon}{A(s)}$ and random actions with probability $\frac{\epsilon}{A(s)}$.

An ε-soft policy guarantees that every action has at least a probability of $\frac{\epsilon}{|A(s)|}$ of being selected.

Using the Generalized Policy Iteration (GPI) framework:
1. Policy Evaluation: Estimate $Q(s,a)$ for the current policy
2. Policy Improvement: Make the policy ε-greedy with respect to the current $Q$-values

For on-policy control, it can be proven that when the policy is updated to be ε-greedy with respect to the current $Q$-values, the new policy is at least as good as the old one (policy improvement theorem). This guarantees that the algorithm will converge to the optimal ε-soft policy.

![[soft_policy_convergence_proof.png]]

In the last line of the derivation, we have:

$$v_\pi(s) = \frac{\varepsilon}{|\mathcal{A}(s)|}\sum_a q_\pi(s,a) + (1-\varepsilon)\sum_a \frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}(s)|}}{1-\varepsilon}q_\pi(s,a)$$

The second term contains a convex combination of q-values, where the weights are: $$\frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}(s)|}}{1-\varepsilon}$$
This is a convex combination because:
1. These weights sum to 1 (can be verified algebraically)
2. Each weight is non-negative (assuming π is an ε-soft policy)

A fundamental property of convex combinations is that they're always less than or equal to the maximum element. That is:

$$\sum_a w_a q_\pi(s,a) \leq \max_a q_\pi(s,a)$$

where $w₁, w₂, ..., wₙ$ are weights that sum to 1 and are non-negative.
This property is crucial because it allows us to establish:

$$\sum_a \frac{\pi(a|s) - \frac{\varepsilon}{|\mathcal{A}(s)|}}{1-\varepsilon}q_\pi(s,a) \leq \max_a q_\pi(s,a)$$

And consequently:

$$v_\pi(s) \leq \frac{\varepsilon}{|\mathcal{A}(s)|}\sum_a q_\pi(s,a) + (1-\varepsilon)\max_a q_\pi(s,a)$$

This is exactly the expected value under the ε-greedy policy π', which proves that the ε-greedy policy improves over the ε-soft policy with respect to $q_{\pi}$.

![[epsilon-greedy-mc.png]]

The algorithm alternates between:
- Estimating $Q(s,a)$ by averaging returns
- Improving the policy by making it ε-greedy with respect to $Q$

### 3. Off-Policy Methods

Off-policy methods allow us to learn about one policy (the target policy) while following another (the behavior policy). This separates exploration from exploitation.

The key idea is:

- Behavior policy $b(a|s)$ (used for generating episodes) is exploratory (e.g., ε-soft)
- Target policy $\pi(a|s)$ (being learned) is greedy with respect to $Q$

The major challenge is that we're collecting data from $b$ but want to estimate expected returns for $\pi$.

How can we approximate the expectation of the return of following policy $\pi$ using samples from following policy $b$?

$$
v_{b}(s) = \mathbb{E}[G(\tau_{t})|S_{t}=s,A_{t}\sim b] = \sum_{\tau_{t}}p(\tau_{t}|S_{t}=s, A_{t}\sim b)G(\tau_{t})
$$
$$
v_{\pi}(s) = \mathbb{E}[G(\tau_{t})|S_{t}=s,A_{t}\sim \pi] = \sum_{\tau_{t}}p(\tau_{t}|S_{t}=s, A_{t}\sim \pi)G(\tau_{t})
$$
## Importance Sampling

Importance sampling is a technique for estimating expected values under one distribution using samples from another distribution. In reinforcement learning, we use it to estimate the expected return under the target policy using episodes generated by the behavior policy.

![[importance_sampling_derivation.png]]

For a trajectory $\tau_t = (S_t, A_t, ..., S_T)$ starting at time $t$, the importance sampling ratio is:

$$
\begin{equation}
\begin{aligned}
\rho_{t:T-1} &= \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_{k},A_{k})}{\prod_{k=t}^{T-1} b(A_k|S_k)p(S_{k+1}|S_{k},A_{k})} \\&=

\frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} b(A_k|S_k)}
\end{aligned}
\end{equation}
$$

This ratio helps reweight the returns to account for the difference between the two policies. There are two main types of importance sampling:
### 1. Ordinary Importance Sampling

$$V(s) = \frac{\sum_{t\in\mathcal{T}(s)} \rho_{t:T(t)-1}G_t}{|\mathcal{T}(s)|}$$

where $\mathcal{T}(s)$ is the set of time steps where state $s$ is visited. This estimator is unbiased but can have high variance.

### 2. Weighted Importance Sampling

$$V(s) = \frac{\sum_{t\in\mathcal{T}(s)} \rho_{t:T(t)-1}G_t}{\sum_{t\in\mathcal{T}(s)} \rho_{t:T(t)-1}}$$

This estimator has lower variance but is biased. However, the bias decreases as more samples are collected, making it often preferred in practice.

## Incremental Implementation

Rather than storing all returns and recalculating the average, we can use incremental updates:
1. For simple average (where $k_s$ is the number of times state $s$ has been visited):$$\hat{V}(S_t) \leftarrow \hat{V}(S_t) + \frac{1}{k_s + 1}(G_t - \hat{V}(S_t))$$
2. For a fixed learning rate $\alpha$:$$\hat{V}(S_t) \leftarrow \hat{V}(S_t) + \alpha(G_t - \hat{V}(S_t))$$
Using a fixed learning rate puts more weight on recent experiences and allows the algorithm to adapt to non-stationary environments. However, for convergence guarantees, learning rates often need to follow specific schedules where they decrease over time.

## Off-Policy Monte Carlo Control

![[off-policy_mc_control.png | 500]]

The off-policy Monte Carlo control algorithm works as follows:
1. Initialize $Q(s,a)$ arbitrarily and $C(s,a) = 0$ for all states and actions
2. Initialize target policy $\pi$ to be greedy with respect to $Q$
3. For each episode:
    - Choose a behavior policy $b$ (e.g., ε-soft)
    - Generate an episode using $b$
    - For each step of the episode (backward):
        - Update importance sampling weight $W$
        - Update $C(s,a)$ and $Q(s,a)$ using weighted importance sampling
        - Update $\pi$ to be greedy with respect to $Q$
        - If the action taken doesn't match the greedy action, break the loop

The algorithm stops updating the weight once it encounters an action that wouldn't have been taken by the target policy. This is because the importance sampling weight would become zero for all subsequent steps.

## Summary of Key Concepts
1. **Monte Carlo methods**:
    - Learn directly from experience
    - Require complete episodes
    - Don't need a model of the environment
    - High variance, especially with long episodes
2. **Monte Carlo prediction**:
    - Estimates value functions by averaging returns
    - Can use first-visit or every-visit techniques
3. **Monte Carlo control**:
    - Requires exploration strategies (exploring starts, soft policies, or off-policy learning)
    - Follows the generalized policy iteration framework
4. **On-policy vs. Off-policy**:
    - On-policy: Learn about the policy being used for decision-making
    - Off-policy: Learn about one policy while following another
5. **Importance sampling**:
    - Required for off-policy learning
    - Ordinary importance sampling: unbiased but high variance
    - Weighted importance sampling: biased but lower variance

Each approach has its strengths and weaknesses, and the choice depends on the specific problem and constraints at hand.

## Need to know
How do Monte Carlo methods learn value functions and what are their properties?
Why do we need ‘exploring starts’, non-greedy policies, or offline learning, and what are the advantages of each?
What are ordinary and weighted importance sampling and when are they used?
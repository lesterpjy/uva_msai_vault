## Refresher on MDPs and Value Functions

A Markov Decision Process (MDP) formally consists of:

- A finite set of states
- A finite set of actions for each state
- A dynamics function: 
- $p(s', r|s, a) = \text{Pr}\{S_t = s', R_t = r|S_{t-1} = s, A_{t-1} = a\}$
    - Often broken down into transition function $p(s'|s,a)$ and reward function $p(r|s,a,s')$
- A discount factor $\gamma \in [0,1)$

The return in RL is defined as the sum of rewards. For episodic tasks: $$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T$$For continuing tasks, we use discounted return: $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$For simplicity, we use the discounted return formulation and allow $\gamma = 1$ for episodic tasks.

## Policies and Value Functions

A policy $\pi$ is a mapping from states to probability distributions over actions. For each state, it specifies how likely the agent is to take each possible action: $\pi(a|s)$.

The state-value function $v_\pi(s)$ represents how good it is to be in state $s$ when following policy $\pi$: $$v_\pi(s) = \mathbb{E}_\pi[G_t|S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t = s\right]$$The action-value function $q_\pi(s,a)$ represents how good it is to take action $a$ in state $s$ and then follow policy $\pi$: $$q_\pi(s,a) = \mathbb{E}_\pi[G_t|S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}|S_t = s, A_t = a\right]$$
## Optimal Policies and Value Functions

A policy $\pi$ is defined as optimal if: $v_\pi(s) \geq v_{\pi'}(s) \text{ for all } s \text{ and all policies } \pi'$

The optimal value functions are defined as: $$v_{*}(s) = \max_\pi v_\pi(s)$$ $$q_{*}(s,a) = \max_\pi q_\pi(s,a)$$Multiple policies can be optimal, and they all share the same optimal value functions.

## Bellman's Principle of Optimality

Richard Bellman's principle (1957) states: "An optimal sequence of controls in a multistage optimization problem has the property that whatever the initial stage, state and controls are, the remaining controls must constitute an optimal sequence of decisions for the remaining problem with stage and state resulting from previous controls considered as initial conditions."

This suggests working backwards from the last step, which is the foundation of dynamic programming.

## Bellman Equations

The Bellman equation expresses a relationship between the value of a state and the values of its successor states:

$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s] = \mathbb{E}_{a\sim\pi,s',r}[r + \gamma v_\pi(s')|S_{t}=s]$

For the optimal value function, we have the Bellman optimality equations:

$v_{*}(s) = \max_a q_{*}(s,a) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1})|S_t = s, A_t = a]$

$q_{*}(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_{*}(S_{t+1},a')|S_t = s, A_t = a]$

These equations are crucial for deriving algorithms to find optimal policies.

## Dynamic Programming Algorithms

### 1. Value Iteration

![[value_iteration_backup.png | 200]]

Value iteration applies the Bellman optimality operator repeatedly:

1. Initialize $V_0(s) = 0$ for all states
2. For $k = 0, 1, 2, \ldots$ until convergence:
    - Update $$\begin{equation}\begin{aligned}
v_{k+1}(s) &= \max_a \mathbb{E}[r(s,a) + \gamma v_k(s')]\\&= \max_a \sum_{s',r}p(s',r|s,a)[r + \gamma v_k(s')]
\end{aligned}\end{equation}$$
At convergence, $v_k$ approaches $v_*$, and we can extract the optimal policy by acting greedily with respect to the value function. Another way of understanding value iteration is by reference to the Bellman optimality equation. Note that value iteration is obtained simply by turning the Bellman optimality equation into an update rule.

![[value_iteration.png]]

### 2. Policy Evaluation

![[policy_eval_backup.png | 200]]

Policy evaluation computes the value function for a given policy:

1. Initialize $V_0(s) = 0$ for all states
2. For $k = 0, 1, 2, \ldots$ until convergence:
    - Update $$\begin{equation}\begin{aligned}v_{k+1}(s) &= \mathbb{E}_{a\sim\pi}\mathbb{E}_{s',r}[r + \gamma v_k(s')|s] \\&=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{k}(s')]\end{aligned}\end{equation}$$
This converges to $v_\pi$, the true value function for policy $\pi$.

### 3. Policy Improvement
The policy improvement theorem is a fundamental concept in reinforcement learning that guarantees we can improve our agent's behavior by making specific adjustments to its policy. 

The policy improvement theorem establishes that if we have:
- A current policy $\pi$
- Its value function $v_{\pi}$
- A new policy $π'$ that always selects actions that maximize expected return according to $v_π$

Then one of two things must be true:
1. The new policy π' will yield strictly better performance (higher value) in at least one state
2. The current policy π is already optimal, so no improvement is possible

#### Mathematical Understanding
If $\pi'$ is equal to $\pi$ except at $s_{t}$, then:
$$
\mathbb{E}_{a_{t}\sim\pi'(s_{t})}q_{\pi}(s_{t},a_{t}) > v_{\pi}(s_{t}) \Rightarrow v_{\pi'}(s) \geq v_{\pi}(s) \ \forall\  s
$$
When we say $π'$ is "greedy with respect to $v_π$", we mean that for each state $s$, the new policy selects the action that maximizes the expected return:

$$π'(s) = argmax_a q_π(s,a)$$

Where $q_π(s,a)$ is the action-value function that tells us the expected return when taking action $a$ in state $s$, then following policy $π$ thereafter.

If we find an action that has higher expected return than what our current policy would do, and we update our policy to select that action, our overall performance cannot get worse—it must either improve or stay the same (if we're already optimal).


### 4. Policy Iteration

![[policy_iteration.png]]

Policy iteration alternates between policy evaluation and policy improvement:
1. Start with an initial policy $\pi_0$
2. Repeat until convergence:
    - Policy Evaluation: Compute $v_{\pi_i}$
    - Policy Improvement: Derive $\pi_{i+1}$ by acting greedily with respect to $v_{\pi_i}$

This algorithm is guaranteed to converge to an optimal policy.

## Value Iteration vs. Policy Iteration

The main difference is:

- **Value Iteration**: Updates the value function once for each state, then improves the policy
- **Policy Iteration**: Updates the value function until convergence, then improves the policy

Both are examples of **Generalized Policy Iteration** (GPI), which describes the general idea of interleaving policy evaluation and improvement.

## Limitations of Dynamic Programming

Dynamic programming requires complete knowledge of the MDP, including all transition probabilities. In reinforcement learning, we typically don't have this knowledge and must learn from experience.

This leads to three main approaches in RL:
1. **Model-free, value-based methods**: Learn value functions directly (lectures 3-7)
2. **Model-free, policy-based methods**: Learn policies directly (lectures 8-10)
3. **Model-based methods**: Learn a model of the environment (lecture 11)

## Key Takeaways
1. **Optimal policy and value function**: The optimal policy achieves the maximum expected return; the optimal value function gives the expected return when following the optimal policy.
2. **Algorithms**:
    - Value Iteration: Compute the optimal value function directly
    - Policy Evaluation: Compute the value function for a given policy
    - Policy Iteration: Alternate between evaluation and improvement
3. **Time horizons**: Each iteration in value iteration effectively computes values for a longer time horizon.
4. **Terminal states**: States from which no further transitions occur; they terminate an episode and have zero value for all future time steps.

These dynamic programming methods form the conceptual foundation for many reinforcement learning algorithms, even those that don't require explicit knowledge of the environment's dynamics.

## Need to know
What is an optimal value function and an optimal policy?
What are the value iteration (VI), policy evaluation (PE) and
policy iteration (PI) algorithms and what do they compute?
How do iterations of VI and PI relate to different time horizons?
What is a terminal state?
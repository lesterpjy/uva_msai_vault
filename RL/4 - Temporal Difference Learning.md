## Introduction to Temporal Difference Learning

Temporal Difference (TD) learning represents a major advancement in RL that bridges the gap between two previously covered approaches: Dynamic Programming (DP) and Monte Carlo (MC) methods. TD learning combines the advantages of both:
- Like Monte Carlo methods, TD learning can learn directly from experience without requiring a model of the environment.
- Like Dynamic Programming, TD learning uses bootstrapping (updating estimates based on other estimates) and exploits the recursive structure of the Bellman equation.

## Key Limitations of Monte Carlo Methods

Before diving into TD learning, it's important to understand the limitations of Monte Carlo methods:

1. **Complete episodes needed**: MC methods require complete episodes to be finished before learning can occur, making them unsuitable for continuing tasks.
2. **Variance issues**: MC methods can have high variance, especially with long episodes, since returns depend on many random actions, transitions, and rewards.
3. **Structural inefficiency**: MC methods don't exploit the recursive structure between consecutive states (the Bellman equation).

## The Temporal Difference Approach

TD learning addresses these limitations by using a fundamentally different update approach. Recall the value function definition:

$$v_\pi(s) = \mathbb{E}_\pi[G_t|S_t = s]$$

The Monte Carlo update rule uses the actual return $G_t$ as the target:

$$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$$

TD learning, however, replaces this with an estimated return based on the next state's value:

$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

The expression $[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ is called the **TD error** (δ), representing the difference between the estimated value of the current state and the reward plus discounted value of the next state.

## TD(0)

![[tabular_td.png]]

TD(0) is the simplest form of TD learning, which looks only one step ahead:

1. Initialize value function $V(s)$ arbitrarily, except $V(\text{terminal}) = 0$
2. For each episode:
    - Initialize state $S$
    - For each step of episode:
        - Choose action $A$ using policy π
        - Take action $A$, observe reward $R$ and next state $S'$
        - Update $V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]$
        - $S \leftarrow S'$
    - Until $S$ is terminal

This algorithm continuously updates value estimates without waiting for episodes to complete.

## Properties of TD(0)

- **Bootstrapping**: Updates value estimates based on other value estimates
- **Online learning**: Updates occur at each step, not just at the end of episodes
- **Not unbiased**: Unlike MC, TD(0) is biased because initial value estimates influence later updates
- **Convergence**: For fixed policy π, TD(0) converges to true value function $V_\pi$ with probability 1 when learning rate decreases appropriately
- **Lower variance**: TD methods typically have lower variance than MC methods, though they introduce bias

## Comparing TD(0) and MC: An Example

The lecture presented Example 6.4 from Sutton & Barto, considering state sequences with states A and B:

```
A, 0, B, 0    B, 1
B, 1          B, 1
B, 1          B, 1
B, 1          B, 0
```

- **MC approach**: Looks at complete returns, averaging 0 for state A and averaging outcomes for state B ($V(A)=0$ and $V(B)=\frac{3}{4}$)
- **TD approach**: Uses the relationship between states, resulting in different value estimates that often generalize better ($V(A)=\frac{3}{4}$ and $V(B)=\frac{3}{4}$)
- Verify the results above and explain why TD might generalize better.

## SARSA: On-Policy TD Control

SARSA is an on-policy TD control method for learning action-value functions. The name comes from the sequence of events used in the update: (State, Action, Reward, next State, next Action).

The update rule for SARSA is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

SARSA is on-policy because it uses the same policy for both acting and updating. For convergence to optimal policy, the behavior policy must gradually become greedy (e.g., by reducing ε in ε-greedy over time).

![[sarsa.png]]

## Q-Learning: Off-Policy TD Control

Q-learning is an off-policy TD control method that directly learns the optimal action-value function $Q^*$ regardless of the policy being followed:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

Key properties of Q-learning:

- **Off-policy**: Can learn about the optimal policy while following an exploratory policy
- **Behavior policy**: Can use any policy that sufficiently explores the state-action space
- **Target policy**: Always uses the greedy policy in its update
- **Convergence**: Converges to $Q^*$ with probability 1 under standard conditions

![[q-learning.png | 400]]

## Comparing SARSA and Q-Learning

While these algorithms may seem similar, they have key differences:

1. **Policy use**:
    - SARSA uses the same policy for behavior and targets (on-policy)
    - Q-learning uses a greedy policy for targets regardless of behavior policy (off-policy)
2. **What they learn**:
    - SARSA learns $Q^\pi$ of the current policy (e.g., ε-greedy)
    - Q-learning learns $Q^*$ of the optimal policy
3. **Behavior in risky environments**:
    - SARSA tends to learn safer paths that account for exploration mistakes
    - Q-learning learns optimal paths assuming perfect execution

An important insight: even if both algorithms use a greedy behavior policy, they are still different algorithms because Q-learning's update always uses the maximum Q-value of the next state, while SARSA uses the Q-value of the actually selected next action.

![[sarsa-vs-q.png | 500]]

## Backup diagrams

![[backup_td.png|500]]

TD methods offer several advantages:
- No model required (unlike DP)
- Can learn online, step-by-step (unlike MC which needs complete episodes)
- Can learn from incomplete sequences
- Can be applied to continuing problems
- Generally faster convergence with lower variance than MC
- Can incorporate function approximation more easily

## The Bigger Picture: Approaches to Reinforcement Learning

The slides present a taxonomy of RL approaches:
1. **Model-free RL**:
    - **Value-based methods**: Learn value functions to implicitly represent policies (TD learning, MC)
    - **Policy-based methods**: Directly optimize policy parameters
2. **Model-based RL**:
    - Learn a model of the environment (transitions and rewards)
    - Use the model to learn values or policies

TD learning fits into the value-based, model-free category and represents an important stepping stone toward more sophisticated RL algorithms.

## Need to know

How do temporal difference (TD) methods compare to dynamic programming (DP) and Monte Carlo (MC) methods?
What are SARSA, and Q-learning and what are their properties?
## Core Concepts of Reinforcement Learning

Reinforcement learning addresses a fundamental question: **How can an agent learn to sequentially interact with an environment to maximize a long-term objective?**

This is distinct from other machine learning paradigms in several important ways:

### Reinforcement Learning vs. Other ML Paradigms

The slides illustrate RL in the context of broader machine learning:

1. **Supervised Learning**: Learning to predict from labeled examples (e.g., classifying images as "2")
    - Uses datasets with human-provided answers
    - Data doesn't depend on what is learned
    - Data points are independent and identically distributed (i.i.d.)
2. **Unsupervised Learning**: Learning patterns without labels (e.g., clustering data points)
    - Discovers structure in unlabeled data
3. **Reinforcement Learning**: Learning to act to maximize rewards
    - Improves over human performance through trial and error
    - Data depends on current strategy/policy
    - Data occurs in sequential form, not i.i.d.
    - Fundamentally about "learning to do" rather than "learning to predict"

### Real-World RL Applications

- Game playing (AlphaGo/AlphaZero)
- Robotics (cooking and assistance robots)
- Self-driving vehicles
- Smart grids and energy management

## K-Armed Bandits: A Simple RL Model

### K-Armed Bandit Problem Definition

- There are k "slot machines" (arms) to choose from
- Each machine has an unknown distribution of payoffs
- Goal: maximize cumulative payoff over time

Formally:
- There are k actions available at each time step
- After selecting action $a_t$, you receive reward $r_t$ from unknown distribution $p(r_t|a_t)$
- Objectives can be:
    - Finite horizon: Maximize total reward over T actions $\sum_{t=1}^{T} r_t$
    - Infinite horizon: Maximize total discounted rewards $\sum_{t=1}^{\infty} \gamma^t r_t$ with discount factor $\gamma \in [0,1)$

If $\gamma = 1$ with an infinite horizon, the sum could be unbounded, making optimization impossible. That's why $\gamma < 1$ is necessary for infinite horizons.

### Estimating Arm Values

The value of an action is its expected reward: $Q_t(a) = E[r_t|a_t]$

If action $a$ has been chosen $k_a$ times, yielding rewards $r_1,...,r_{k_a}$, we can estimate:
$$\hat{Q}_t(a) = \frac{\sum_{i=1}^{k_a} r_i}{k_a}$$

This requires storing all rewards, which is inefficient. Instead, we can use an incremental update rule:

$$\hat{Q}_{t+1}(a) = \hat{Q}_t(a) + \frac{1}{k_a + 1}[r_t - \hat{Q}_t(a)]$$

This formula adjusts our estimate by a fraction of the difference between the new reward and our current estimate.

### The Exploration-Exploitation Dilemma

A central challenge in RL is balancing:
- **Exploitation**: Choosing the arm with highest estimated value to maximize immediate reward
- **Exploration**: Trying arms with uncertain or lower estimated values to potentially discover better options

The slides describe several exploration strategies:
1. **Greedy Strategy**: Always select the action with highest estimated value
    - No exploration, just exploitation
2. **ε-Greedy Strategy**: Choose best action with probability 1-ε, random otherwise
    - Simple, but doesn't distinguish between slightly worse and much worse actions
3. **Softmax/Boltzmann Exploration**: Probability of choosing action proportional to its estimated value
    - $$p(a) = \frac{e^{\hat{Q}(a)/\tau}}{\sum_{a'} e^{\hat{Q}(a')/\tau}}$$
    - Parameter $\tau$ (temperature) controls exploration level
4. **Upper Confidence Bound (UCB)**: Select actions based on optimistic estimates
    - Consider both mean value and uncertainty
    - Automatically explores actions that haven't been tried much

## Markov Decision Processes (MDPs)

K-armed bandits are limited because they don't model how actions affect future states. For more complex sequential decision problems, we use Markov Decision Processes.

### MDP Structure

![[mdp.png | 400]]

In an MDP:
- An agent interacts with an environment over discrete time steps
- At each step, the agent observes the current state $s_t$, takes action $a_t$, and receives reward $r_t$
- The environment transitions to a new state $s_{t+1}$

The interaction forms a loop:
1. Agent observes state $s_t$
2. Agent selects action $a_t$
3. Environment provides reward $r_t$ and next state $s_{t+1}$
4. Process repeats

### MDP Assumptions

An MDP makes several key assumptions:
1. **Markov property**: The next state depends only on the current state and action, not on any previous states or actions
2. **Reward dependence**: Rewards depend only on the current state, action, and next state
3. **Discrete time steps**: Interaction happens in discrete steps
4. **Full observability**: The agent can fully observe the current state (no hidden information)

These assumptions imply that the environment is **stationary** - the transition dynamics don't change over time.

### Formal MDP Definition

A finite MDP consists of:
- A finite set of states
- A finite set of actions for each state
- A dynamics function: $$p(s',r|s,a) = Pr\{S_t=s', R_t=r|S_{t-1}=s, A_{t-1}=a\}$$
    - Often split into transition function $p(s'|s,a)$ and reward function $p(r|s,a,s')$
- A discount factor $\gamma \in [0,1)$

### Return and Horizons

The **return** is the cumulative reward that the agent aims to maximize:

For episodic tasks (with finite episodes): $$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T$$For continuing tasks (potentially infinite): $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
The discount factor $\gamma$ ensures that the infinite sum converges and encourages the agent to prioritize earlier rewards.

## Why is RL Challenging?

The lecture highlights several key challenges:
1. **Credit assignment problem**: When a good or bad outcome occurs, it's difficult to know which of the many previous actions were primarily responsible.
2. **Non-i.i.d. data**: The data distribution changes as the agent's policy improves, violating the i.i.d. assumption common in other ML approaches.
3. **Exploration-exploitation tradeoff**: Balancing between exploiting known good strategies and exploring to discover potentially better ones.
4. **State representation**: The decision process depends heavily on how states are defined and represented.

## Conclusion

This introduction lays the foundation for understanding reinforcement learning by introducing:
- The core distinction between RL and other ML paradigms
- The k-armed bandit as a simple RL model
- The exploration-exploitation dilemma
- Markov Decision Processes as a framework for sequential decision problems
- The notion of return in both episodic and continuing tasks
- Key challenges that make RL difficult

The course will build on these concepts to develop algorithms for finding optimal policies in various reinforcement learning settings.
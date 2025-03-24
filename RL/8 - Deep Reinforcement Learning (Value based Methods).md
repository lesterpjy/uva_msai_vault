## Introduction to Deep Q Networks (DQN)

DQN was a groundbreaking algorithm developed by DeepMind (published in Nature, 2015) that demonstrated how deep neural networks could be effectively combined with reinforcement learning to solve complex tasks. It was particularly notable for learning to play Atari games directly from pixel inputs, in many cases achieving human-level performance or better.

### Q-Function Approximation Architectures

Two main approaches to approximating the Q-function:
1. **State to value mapping**: $s \rightarrow \hat{v}(s)$
    - One network that outputs the value of a state
2. **State-action to Q-value mapping**: Two possibilities:
    - One network taking both state and action as input: $(s,a) \rightarrow \hat{q}(s,a)$
        - Useful for large action spaces where you need to evaluate individual actions
    - One network taking just the state and outputting Q-values for all possible actions: $s \rightarrow [\hat{q}(s,a_1), \hat{q}(s,a_2), ..., \hat{q}(s,a_n)]$
        - Useful for modest-sized discrete action spaces
        - Essentially learns a mapping from states to "lookup tables" over actions
        - This is the approach used in the original DQN paper

### DQN Architecture

DQN uses convolutional neural networks (CNNs) to process visual input. This has several advantages:
1. **Feature learning**: Unlike linear function approximation (which requires hand-designed features), CNNs can learn useful representations directly from raw data.
2. **Task-agnostic preprocessing**: The same network architecture can work across many different tasks without domain-specific feature engineering.
3. **CNNs are well-suited for visual data** because:
    - They use the same filters at each location in the image (parameter sharing)
    - They only connect neurons locally, reducing the number of parameters
    - They impose translation equivariance, meaning objects are recognized regardless of position

For Atari games, DQN uses the following preprocessing:
- Downscaling images and converting to grayscale to reduce computational requirements
- Stacking multiple recent frames to provide temporal context (giving the agent a "short memory")
- Using the increment in game score as the reward

### Key Innovations in DQN

DQN introduced two major innovations to stabilize training of neural networks with Q-learning:
1. **Experience replay buffer**:
    - Transitions $(s, a, r, s')$ are stored in a replay buffer
    - Training happens on random batches from this buffer, not just the most recent transition
    - This breaks the correlation between consecutive samples
    - It allows experiences to be used multiple times for training
    - It improves GPU efficiency by allowing batch processing
    - Learning from behavior policy n-time steps ago, so is off-policy. Current $b \neq b$ of experience 
2. **Target network**:
    - In standard Q-learning, the target depends on the current Q-network weights: $\underbrace{R + \gamma \max_{a'} \hat{q}(s', a', w)}_{\text{target}} - \hat{q}(s, a, w)$
    - This target depends on $w$
    - This can cause instability as the target keeps changing during training
    - DQN uses a separate "target network" with parameters $\tilde{w}$ that are periodically copied from the main network
    - The target becomes: $R + \gamma \max_{a'} \hat{q}(s', a', \tilde{w}) - \hat{q}(s, a, w)$
    - This creates more stable training targets
3. **Error clipping**: DQN also clips the TD error to the range [-1, 1], which provides additional stability and makes the learning rate easier to set.

The tricks make the problem more ‘supervised’:
- Target network: fixed target value
- Experience replay: closer to iid
Still semi-gradient used off-policy
- Hence the need to stabilise learning
- Lack theoretical guarantees, but good empirical performance
Non-linear function approximation
- Risk of local optima seems not so bad in practice for deep nets

## Offline Reinforcement Learning
### Challenges of Offline RL

Offline RL is about learning from a fixed dataset without additional environment interaction. This is useful when:

- Exploration is unsafe (e.g., in healthcare, robotics)
- Data collection is expensive
- We want to leverage existing datasets

However, offline RL presents unique challenges:

- The behavior policy that collected the data may be far from the optimal policy
- Certain states, actions, or transitions may be missing from the dataset
- Function approximation errors for out-of-distribution actions can be particularly problematic

In offline RL, estimation errors for actions that were rarely chosen by the behavior policy can lead to catastrophic outcomes when the agent tries to exploit these erroneously high Q-values.

### Conservative Q-Learning (CQL)

Conservative Q-Learning (CQL) is an approach to offline RL that addresses the overestimation problem by adding a penalty to the Q-learning objective.

The core idea is to lower-bound the true Q-values to prevent harmful overestimation, making the agent "conservative" in its predictions.

Mathematically, if we denote the behavior policy as $\beta$ and the target policy as $\pi$, CQL adds a penalty term:
$$-\alpha \frac{\pi(a|s) - \beta(a|s)}{\beta(a|s)}$$
This penalizes the agent for choosing actions that deviate too much from the behavior policy.

The resulting Q-value update has the form:

$$\hat{V}^{k+1}(s) := \mathbb{E}_{a\sim\pi(a|s)}[\hat{Q}^{k+1}(s,a)] = \hat{\mathcal{B}}^\pi \hat{V}^k(s) - \alpha\mathbb{E}_{a\sim\pi(a|s)}\left[\frac{\pi(a|s) - \beta(a|s)}{\beta(a|s)}\right]$$

CQL guarantees that:

$$\hat{V}^\pi(s) \leq V^\pi(s)$$

That is, the estimated value function is a lower bound on the true value function. This conservatism prevents the agent from exploiting erroneously high Q-values for out-of-distribution actions.

The final CQL loss function combines:
1. A standard TD learning term (Bellman error)
2. The CQL penalty term that encourages conservatism
3. A regularization term on the policy (not shown in the equation below):

$$\min_Q \max_\pi \alpha \left( \mathbb{E}_{s\sim\mathcal{D},a\sim\pi(a|s)}[Q(s,a)] - \mathbb{E}_{s\sim\mathcal{D},a\sim\beta(a|s)}[Q(s,a)] \right) + \frac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}\left[\left(Q(s,a) - \hat{\mathcal{B}}^{\pi_k}\hat{Q}^k(s,a)\right)^2\right]$$

CQL is particularly effective with small datasets, where the risk of overestimation is highest. It provides theoretical guarantees while being easy to implement (only about 20 lines of code added to a standard Q-learning implementation).

## Weaknesses of Value-Based Methods

The lecture concludes by noting some limitations of value-based approaches:
1. **Handling continuous or large action spaces**: Value-based methods like Q-learning require finding $\arg\max_a Q(s,a)$, which is problematic when the action space is continuous or very large.
2. **Fixed policy randomness**: Value-based methods typically use deterministic policies or simple exploration strategies like ε-greedy, which cannot learn the optimal degree of randomness required in some tasks (particularly those with perceptual aliasing).

## Summary

Deep Q Networks combine deep neural networks with Q-learning, enabling reinforcement learning directly from high-dimensional inputs like images. Key innovations like experience replay and target networks stabilize training.

Conservative Q-Learning extends these ideas to offline RL by introducing a penalty that ensures conservative Q-value estimates, preventing the exploitation of erroneously high values for out-of-distribution actions.

Deep Q networks (DQN)
- Semi-gradient version of Q learning
- Deep neural networks to learn directly from ‘raw’ data
- Good empirical performance

Conservative Q learning
- Off-line RL setting
- Theoretical guarantee of lower bound
- Improved performance on Atari & other domains, especially with small datasets
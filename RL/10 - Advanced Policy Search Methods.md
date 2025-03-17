
I'll cover the key concepts from the lecture on advanced policy search methods, focusing on actor-critic methods, deterministic policy gradients, and considerations for empirical evaluation.

## Overview of Policy Learning Approaches

Reinforcement learning methods can be broadly categorized based on their approach to learning policies:

1. **Value-based methods**: Learn value functions (V(s) or Q(s,a)) and derive policies from them
2. **Policy-based methods**: Directly optimize policy parameters
3. **Model-based methods**: Learn environment dynamics and use them for planning/policy learning

Within model-free methods, we have three subcategories:

- **Critic-only methods**: Traditional value-based approaches like SARSA, Q-learning
- **Actor-only methods**: Pure policy gradient methods like REINFORCE
- **Actor-critic methods**: Combine policy optimization with value function approximation

## Policy Gradient Review

In the previous lecture, we covered stochastic policy gradient methods like REINFORCE. The REINFORCE algorithm uses Monte Carlo returns to estimate the gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t+1:T} r_{t'} \right]$$

This can be approximated using samples:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t+1:T} r_{t'}$$

## Policy Gradient Theorem (PGT)

The Policy Gradient Theorem extends this by showing that we can use expected returns (the action-value function $q_\pi$) rather than just Monte Carlo samples:

$$\nabla J = \mathbb{E}_\tau \left[ \sum_{t'=1}^T \nabla \log \pi(a_{t'}|s_{t'})q_\pi(s_{t'}, a_{t'}) \right]$$

This can be rewritten as an expectation over state-action pairs from the on-policy distribution:

$$\nabla J \propto \mathbb{E}_{\mu(s)\pi(a|s)} \nabla \log \pi(a|s)q_\pi(s, a)$$

This formulation shows that state-action pairs at each time step contribute equally to the total gradient, which simplifies implementation.

## Actor-Critic Methods

Actor-critic methods combine policy gradient approaches with value function approximation. They have two main components:

- An **actor** (the policy) that determines actions
- A **critic** (the value function) that evaluates those actions

### PGT Actor-Critic

In a PGT Actor-Critic, we replace the true $q_\pi$ with an estimate, typically using TD learning:

$$\nabla J = \mathbb{E}_{\mu(s)\pi(a|s)} \nabla \log \pi(a|s)q_\pi(s, a)$$

The update rule becomes:

$$\theta_{t+1} = \theta_t + \alpha (R_{t+1} + \gamma\hat{v}(s_{t+1}, w)) \nabla \log \pi(a_t | s_t, \theta_t)$$

Where:

- $\theta$ represents policy parameters
- $w$ represents value function parameters
- $\hat{v}(s, w)$ is the estimated state-value function

**Advantages**:

- Lower variance compared to Monte-Carlo returns
- Can be formulated for continuing (non-episodic) settings

**Disadvantages**:

- Introduces bias in the gradient estimate
- More complex to implement with multiple moving parts

### Actor-Critic with Baseline

We can further reduce variance by using a baseline:

$$\theta_{t+1} = \theta_t + \alpha (R_{t+1} + \gamma\hat{v}(s_{t+1}, w) - \hat{v}(s_t, w)) \nabla \log \pi(a_t | s_t, \theta_t)$$

The term $(R_{t+1} + \gamma\hat{v}(s_{t+1}, w) - \hat{v}(s_t, w))$ is the TD error $\delta$, which serves as an unbiased estimate of the advantage function.

## Deterministic Policy Gradients (DPG)

All the policy gradients discussed so far work with stochastic policies. However, in some cases (particularly with continuous action spaces), deterministic policies might be more efficient.

### Challenges with Deterministic Policies

For discrete actions, deterministic policies are problematic for gradient-based methods because:

- Small parameter changes either don't change the action (gradient = 0)
- Or cause a jump to a different action (gradient is undefined)

For continuous actions, however, small parameter changes can lead to small action changes, making gradient-based approaches viable.

### DPG Formulation

DPG utilizes two policies:

- A deterministic policy $\pi_\theta$ (the actor) that we want to learn
- A stochastic behavior policy $\beta$ (for exploration) that collects experiences

The objective function is:

$$J_\beta(\pi_\theta) = \int_S \mu^\beta(s)V^\pi(s)ds = \int_S \mu^\beta(s)Q^\pi(s, \pi_\theta(s))ds$$

The off-policy deterministic policy gradient is:

$$\nabla_\theta J_\beta(\pi_\theta) = \int_S \mu^\beta(s)\nabla_\theta Q^\pi(s, \pi_\theta(s))ds = \mathbb{E}_{s\sim\mu^\beta} \left[ \nabla_\theta \pi_\theta(s)\nabla_a Q^\pi(s,a)|_{a=\pi_\theta(s)} \right]$$

Note that $\beta$ only appears in the sampling distribution, not in the gradient calculation itself.

### Deep Deterministic Policy Gradient (DDPG)

DDPG applies DPG with neural network function approximators. It addresses several challenges:

4. **Correlated samples**: Uses experience replay to break correlations
5. **Unstable Q-learning**: Uses target networks for both actor and critic
6. **Input scaling issues**: Applies batch normalization
7. **Exploration**: Uses correlated noise (like Ornstein-Uhlenbeck process) instead of independent noise

DDPG combines insights from DQN (experience replay, target networks) with deterministic policy gradients, making it effective for continuous control tasks.

## Empirical Evaluation of RL Methods

When evaluating and comparing RL methods, several important factors should be considered:

### Environment Selection

- Performance on a single environment isn't sufficient to claim an algorithm is generally superior
- Choose environments that specifically test the aspects your method addresses
- Use multiple environments to test generalization

### Parameter Sensitivity

RL methods can be highly sensitive to:

- Neural network architecture (layers, neurons)
- Activation functions
- Hyperparameters (learning rates, discount factors)
- Reward scaling
- Implementation details

### Reproducibility Considerations

- Run experiments with multiple random seeds (at least 5, preferably 15+)
- Report central tendency (mean/median) with appropriate error metrics
- Clearly specify what is being measured (final performance, learning curve, etc.)
- Document implementation details and hyperparameter settings

### Debugging RL Methods

Debugging complex RL systems presents unique challenges:

- Loss functions don't always decrease due to changing data distribution
- Isolate components and test them separately
- Try batch mode or freeze policies to test specific components
- Start with simple problems before scaling up
- Be aware that "actor loss" in frameworks is often a proxy for the true objective

## Advantages of Policy-Based Methods

Policy search methods are preferable in several scenarios:

8. Problems with continuous action spaces
9. When stochastic policies are needed
10. When prior knowledge about policies is available
11. When small, controlled policy updates are important

Many of these aspects are particularly relevant for physical systems like robots, explaining why policy-based methods are popular in robotics.

This lecture provides an overview of advanced policy search methods that build upon basic policy gradient approaches, with a particular focus on actor-critic methods and deterministic policy gradients. These methods address various limitations of simpler approaches and have been successfully applied to complex control problems with continuous action spaces.
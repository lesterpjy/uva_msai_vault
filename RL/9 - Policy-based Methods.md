## Introduction to Policy-based Methods

Policy-based methods represent a fundamental approach in reinforcement learning (RL) that directly optimizes the policy function without an intermediate value function. This lecture focuses on understanding these methods, particularly policy gradient techniques.

In reinforcement learning, we collect data in the form of transitions: $D = {(s_i, a_i, r_i, s'_i)}_{i=1...N}$, where:

- $s_i$ is the state
- $a_i$ is the action taken
- $r_i$ is the reward received
- $s'_i$ is the next state

From this data, we can take different approaches:

1. **Learn a value function** $V(s)$ or $Q(s,a)$ and derive a policy from it
2. **Directly optimize a policy** $\pi(a|s)$
3. **Learn a dynamics model** $p(s'|s,a)$ and $r(s,a)$, then derive a policy

The first two approaches are model-free RL methods, while the third is model-based. Policy-based methods take the second approach by directly optimizing policy parameters.

## Limitations of Action-Value Methods

Value-based methods have several significant limitations:
1. **Handling continuous actions is difficult**
    - The max operator in $\arg\max_a Q(s,a)$ becomes inefficient or intractable for continuous action spaces
2. **Difficult to ensure policy smoothness**
    - Small changes in value function can lead to drastic changes in policy
    - Hard to control policy update size directly
3. **Hard to incorporate prior knowledge about solutions**
    - Knowledge about good policies is difficult to encode into value functions
4. **Cannot learn truly stochastic policies**
    - Value-based methods typically lead to deterministic policies (take the max-value action)
    - Stochastic policies are important when:
        - States are aliased (same features but different optimal actions)
        - In partially observable environments
        - For exploration

## Policy Representation

In policy-based methods, we directly parameterize a policy function $\pi_\theta(a|s)$ with parameters $\theta$. Desirable properties for policy representations include:

- Representing a distribution over all possible actions
- Having differentiable (log)probabilities with respect to parameters

Common policy representations include:
1. **Linear Gaussian policy** (for continuous actions):
    - $a \sim \mathcal{N}(\theta^T x(s); \sigma)$
    - Mean is linear in state features, with fixed or learnable variance
2. **Neural network policy** (for continuous actions):
    - $a \sim \mathcal{N}(NN_{\theta_\mu}(s); NN_{\theta_\sigma}(s))$
    - Both mean and variance can be functions of the state
3. **Softmax policy** (for discrete actions):
    - $\pi_\theta(a|s) = \frac{\exp f_\theta(s, a)}{\sum_{a' \in A} \exp f_\theta(s, a')}$
    - $f$ can be linear, neural network, etc.

## Optimization Objective

Every policy defined by parameter vector $\theta$ has an expected return $J(\theta) = \mathbb{E}[G]$ where $G$ is the total reward.

We can visualize the implicit mapping from $\theta$ to $J$ as a surface to optimize. The goal is to find parameters $\theta$ that maximize $J(\theta)$.

## Optimization Approaches

Different approaches to finding the optimal policy:
1. **0-order methods**:
    - Grid search
    - Random search
    - Meta-heuristics (like evolutionary algorithms)
    - These don't use gradient information
2. **1st order methods**:
    - Policy Gradients
    - Use gradient information to climb the performance surface
3. **2nd order methods**:
    - Use information about how gradient changes

## Finite Difference Gradients

One approach to estimate policy gradients is finite difference:

1. Consider two policies with parameters $\theta + \epsilon$ and $\theta - \epsilon$
2. Estimate gradient as: $\nabla J \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$
3. Update parameters: $\theta_{t+1} = \theta_t + \alpha \hat{\nabla} J$

This approach has limitations:

- Must repeat for each dimension of $\theta$
- High variance when policy evaluation is stochastic
- Inefficient for high-dimensional parameter spaces

## REINFORCE Algorithm (Policy Gradient)

The REINFORCE algorithm provides an analytical gradient for policy-based methods:

$\nabla_\theta J = \nabla_\theta \mathbb{E}_\tau[G(\tau)]$ $= \int \nabla_\theta p_\theta(\tau) G(\tau) d\tau$ $= \int \frac{p_\theta(\tau)}{p_\theta(\tau)} \nabla_\theta p_\theta(\tau) G(\tau) d\tau$ $= \mathbb{E}_\tau \left[ \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} G(\tau) \right]$ $= \mathbb{E}_\tau [\nabla_\theta \log p_\theta(\tau) G(\tau)]$

This can be further expanded to:

$\nabla_\theta J = \mathbb{E}_\tau \left[ G(\tau) \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$

The gradient of the trajectory probability simplifies because:

1. Initial state distribution $p(s_0)$ doesn't depend on $\theta$
2. Transition dynamics $p(s_{t+1}|a_t, s_t)$ don't depend on $\theta$
3. Only the policy $\pi_\theta(a_t|s_t)$ depends on $\theta$

This gives us the REINFORCE update rule:

1. Sample trajectories using the current policy
2. For each trajectory, compute $G(\tau)$ and $\nabla_\theta \log \pi_\theta(a_t|s_t)$ for each step
3. Update: $\theta_{t+1} = \theta_t + \alpha \hat{\nabla} J$ where $\hat{\nabla} J = \frac{1}{N} \sum_{i=1}^N \left[ G(\tau_i) \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \right]$

![[reinforce.png | 400]]
### Example Gradient Calculations

For a softmax-like policy: If $p(A = a_1|S) = (1+\exp(-\theta_1))^{-1}$ for $S = s_1$, then: $\nabla_{\theta_1} \log p(A = a_1|S = s_1) = \frac{1}{1 + \exp(\theta_1)}$

For a Gaussian policy with linear mean: If $a \sim \mathcal{N}(\theta^T x(s); \sigma)$, then: $\nabla_\theta \log \pi(a_t|s_t) = \frac{a-\theta^T x(s)}{\sigma^2} x(s)$

## Variance Reduction in REINFORCE

The basic REINFORCE algorithm has high variance, making learning inefficient. Two main improvements:

### 1. Baselines

We can subtract a baseline $b$ from the returns:
$$\nabla_\theta J = \mathbb{E}_\tau [(G(\tau) - b) \nabla_\theta \log \pi_\theta(\tau)]$$

This doesn't bias the gradient estimate because:
$$\mathbb{E}_\tau [b\nabla_\theta \log \pi_\theta(\tau)] = b\mathbb{E}_\tau [\nabla_\theta \log \pi_\theta(\tau)] = b \nabla_\theta \int \pi_\theta(\tau) d\tau = b \nabla_\theta 1 = 0$$
A good baseline is the expected return (e.g., observed average) or a state-dependent value function.

### 2. Credit Assignment Improvement (REINFORCE v2)

In the basic REINFORCE, all actions get credit for all rewards in a trajectory. This introduces unnecessary variance.

A better approach is to assign credit more carefully: $$\nabla_\theta J = \mathbb{E}_\tau \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t+1:T} r_{t'} \right]$$
That is, multiply the policy gradient at time $t$ only with the sum of rewards received after time $t$: $\sum_{t'=t+1}^T r_{t'} = G_t$

This cuts out variance from rewards received earlier in the trajectory.

#### Reinforce V2 with baseline

![[reinf_v2_baseline.png | 500]]
This approach learns both a policy and a value function simultaneously, using the value function as a baseline to reduce gradient variance.
## Advantages and Limitations of Policy Gradient Methods

### Advantages:
1. Can handle continuous action spaces naturally
2. Can ensure smooth policy updates with small step sizes
3. Can incorporate prior knowledge through policy form or initialization
4. Can learn stochastic policies when needed
### Limitations:
1. High variance from Monte Carlo rollouts
2. Typically specific to episodic settings
3. Require differentiable policies
4. Variance from stochastic transitions increases with trajectory length

## Comparison of Methods

The lecture presents a comparison framework:

**Exploration axis**: Episode-based to Step-based

- Finite differences (blackbox methods) are episode-based
- Original REINFORCE is step-based

**Evaluation axis**: Episode-based to Step-based

- REINFORCE v2 uses step-based evaluation

Each approach makes different tradeoffs in terms of variance, bias, and computational efficiency.


## Introduction to Model-based RL

Model-based reinforcement learning is a major approach in RL that focuses on learning a model of the environment dynamics to help an agent make decisions. This contrasts with model-free approaches, which learn policies or value functions directly from experience without explicitly modeling the environment.

The big picture of reinforcement learning approaches can be categorized into:

1. **Model-free RL**:
    - **Value-based methods**: Learn value function V(s) or Q(s,a), then derive policy
    - **Policy-based methods**: Directly optimize policy π(a|s)
2. **Model-based RL**:
    - Learn dynamics model p(s'|s,a) and reward function r(s,a)
    - Use this model to learn values/policies through planning

## Why Use Model-based RL?

There are several compelling reasons to use model-based approaches:
1. **Data efficiency**: Real-world data collection can be expensive, risky, or time-consuming (e.g., robotics, autonomous vehicles, medical applications). A model allows us to generate synthetic experiences.
2. **Access to additional information**:
    - Counterfactuals (what would happen if...?)
    - Internal gradients for optimization
    - Full probability distributions (not just samples)
    - Ability to simulate many trajectories
3. **On-policy limitations**: Some environments don't allow arbitrary exploration for safety reasons.

For instance, autonomous vehicles must learn safely without endangering passengers or other road users. A model allows testing risky scenarios in simulation before deploying in the real world.

## Planning, Learning, and Acting Cycle

![[planning-acting-learning.png | 300]]

Model-based RL involves three key processes that form a cycle:
1. **Model Learning**: Learning a model of the environment dynamics from experience
2. **Planning**: Using the model to improve policies/values
3. **Acting**: Using the improved policy to interact with the environment

## Dyna-Q: A Simple Model-based RL Algorithm

Dyna-Q combines direct RL with model-based planning in a simple yet powerful way:

![[dyna-q.png | 500]]

The key insight here is that Dyna-Q performs both:

- Direct RL updates based on real experience (step d)
- Planning updates using simulated experiences from the model (step f)

This provides better sample efficiency by reusing past experience through the model.

### Dyna-Q Environment Model Example

![[dyna-q_example.png | 400]]

### Dyna-Q Performance

Empirical results demonstrate that increasing the number of planning steps (N) in Dyna-Q greatly reduces the number of real environment steps needed to find optimal policies. This illustrates the core advantage of model-based methods: reduced sample complexity.

However, model-based RL typically requires more computation per real experience:

- Compute time for learning the model
- Compute time for planning with the model

## Model Learning Approaches

Potential choices:

![[dyna-q_choices.png | 400]]


Several approaches exist for learning environment models:
1. **Table-based** (like in Dyna-Q):
    - Store observed (s,a) → (s',r) transitions
    - For deterministic environments, simply record the observed result
    - For stochastic environments, can track frequencies/probabilities
2. **Function approximation**:
    - For large or continuous state/action spaces, use neural networks or other function approximators
    - For discrete states: can use classification loss
    - For continuous states: can use regression loss

## Planning Methods
Different planning methods can be used with learned models:
1. **Sample-based planning** (like in Dyna-Q):
    - Generate samples from the model
    - Update value functions or policies using model-free methods
2. **Distribution-based planning**:
    - Use full distribution information for more powerful updates
    - Examples include value iteration-like updates
3. **Analytical methods**:
    - Use gradients of the model for policy optimization
    - Backpropagate reward information through the model

An important caveat: **Don't trust the model too much** - model errors can lead to poor policies.

## When to Plan

There are two major approaches to timing the planning process:
1. **Planning ahead of acting**:
    - Learn a good policy for any state before interacting with the environment
    - Example: Dyna-Q does this by learning a complete policy before execution

2. **Planning while acting**:
    - Focus planning on the current state and its likely successors
    - Think of chess players planning their next moves from the current board position
    - Examples: Monte Carlo Tree Search (MCTS) and AlphaGo

## Rollout algorithm

![[roll-out-algo.png | 400]]

- Play $N$ games using a policy $b$ (can be random policy) from $s_{t}$
- Observe $N$ games/rollouts $\tau_{1} = (s_{t,a_{t+1}, s_{t+1},\dots}), \dots, \tau_{N}$
- Reward/winner/value: $v_{i} \in \{-1,1\}, i=1,\dots,N$
- Estimate action value as mean of trajectories starting with action $a$:
$$
q_{\pi}(s,a) \approx Q(s,a) = \frac{\sum_{i:\tau_{i}=(s_{t},\mathbf{a}..)}v_{i}}{\sum_{i:\tau_{i}=(s_{t},\mathbf{a}..)}1}
$$
![[rollout_algo.png | 400]]

## Monte Carlo Tree Search (MCTS)

MCTS is a powerful planning algorithm that incrementally builds a search tree focused on the most promising paths. It's especially useful for domains with large state spaces, like board games.

![[mcts.png | 400]]

MCTS consists of four main steps:
1. **Selection**: Starting from the root, select successive child nodes until reaching a leaf node. The selection policy balances exploration and exploitation, often using formulas like Upper Confidence Bounds (UCB1):$$\pi_{tree}(s) = \arg\max_a \left[Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\right]$$Where:
    - N(s) = number of visits to state s
    - N(s,a) = number of times action a was selected in state s
    - Q(s,a) = estimated value of taking action a in state s
    - c = exploration parameter
2. **Expansion**: Add one or more child nodes to the tree (expand the leaf node)
3. **Simulation**: Perform a rollout from the new node(s) using a simple policy (often random) until reaching a terminal state or a set depth
4. **Backup**: Update the values of nodes in the path from the expanded node to the root based on the simulation result

This process is repeated many times within a computational budget, gradually improving the estimates of action values.

### AlphaGo and AlphaZero

![[alphazero.png | 400]]

AlphaGo and its successor AlphaZero enhance MCTS with neural networks:
1. **Policy network** ($\pi_θ(a|s)$): Guides tree search by providing prior probabilities for actions
2. **Value network** ($V_θ(s)$): Evaluates leaf nodes without requiring full simulations
In AlphaZero, the tree selection formula becomes:

$$\pi_{tree}(s) = \arg\max_a \left[Q(s,a) + \frac{\pi_θ(a|s)}{N(s,a) + 1}\right]$$

This creates a powerful combination:
- Neural networks provide general knowledge about good moves and positions
- MCTS provides precise tactical calculations for the current position

The neural networks are trained by self-play, with the MCTS search results acting as a teacher for the networks in a reinforcement learning loop.

## The Big Picture

![[big_picture.png | 300]]

Model-based RL combines the strengths of different approaches:
- Models allow sample-efficient learning
- Planning uses models to improve decisions
- Learning from real experience keeps the process grounded

We can view approaches like AlphaGo as sitting at the intersection of:
- Parametrized value functions (critic)
- Parametrized policies (actor)
- Parametrized or given transition models

## Key Takeaways

1. Model-based RL is especially valuable when real-world data is expensive, risky, or limited.
2. The general structure involves learning a model, using it for planning, and applying the resulting policy.
3. Models can be learned in various ways: tables for small MDPs, neural networks for complex environments.
4. Planning can be done "ahead of time" (Dyna-Q) or "in the moment" (MCTS, AlphaGo).
5. Neural networks can enhance planning algorithms by providing priors and evaluations, as in AlphaGo/AlphaZero.
6. Model errors can compound, so care must be taken when relying on learned models.
7. Modern approaches often combine model-based and model-free techniques for the best of both worlds.

## Need to know

Why do model-based reinforcement learning?
What is the general structure of model-based learning
What are some answers to the questions:
- How to learn model
- When to update
- How to update
How does Alpha-Go leverage planning both ‘ahead of’ and ‘while’ acting in the world?

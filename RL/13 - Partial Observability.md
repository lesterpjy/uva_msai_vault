## Introduction to Partial Observability

In standard Markov Decision Processes (MDPs), we assume that the agent has complete access to the true state of the environment. However, in many real-world scenarios, this assumption doesn't hold. This is where partially observable MDPs (POMDPs) come into play.

In a POMDP, the agent doesn't directly observe the true underlying state of the environment. Instead, it receives observations that provide incomplete or noisy information about the true state. This creates significant challenges for decision-making, as the agent must now reason about uncertainty over states.

## The Partial Observability Problem

There are several common ways in which partial observability manifests:

1. **Aliasing**: Different states may appear identical to the agent. For example, an agent might observe "I am in front of a door" but not know which specific door it is facing. In games like Pong, just seeing the current frame doesn't tell you the ball's direction or velocity.
2. **Noise**: Sensors might provide imprecise readings. For instance, GPS signals might tell you approximately where you are, but with some error margin.
3. **Missing information**: Some aspects of the environment state might be entirely hidden from the agent.

In these situations, the single current observation is insufficient for optimal decision-making. The agent needs to somehow accumulate and process information from its history of interactions with the environment.

## The Formal POMDP Framework

![[pomdp_variables.png | 350]]

In a POMDP setting, we distinguish between:

- **x**: The latent "true" state of the environment (which the agent cannot directly observe)
- **o**: The observation the agent receives
- **a**: The action the agent takes
- **r**: The reward the agent receives

The environment now has three key components:

- Transition function: $p(x'|x,a)$ (how the true state evolves)
- Reward function: $r(x',a)$ (rewards based on states and actions)
- Observation function: $p(o'|a,x')$ (what the agent observes)

The agent must now make decisions based on observations rather than the true state.

## Approaches to Handling Partial Observability

![[pomdp_approaches.png | 400]]

When dealing with partially observable environments, we need a way to represent the agent's internal belief or knowledge about the environment. This internal representation, often denoted as **s**, serves as input to the policy and value functions.

### 1. Full History Approach

The simplest approach is to use the entire history of interactions:

$H_t = A_0, O_1, ..., A_{t-1}, O_t$

The agent's policy then becomes: $A_t = \pi(H_t)$

**Advantages**:

- Extremely simple to implement
- Guarantees a Markov representation (contains all available information)

**Disadvantages**:

- Not compact; memory requirements grow over time
- Representing policies over all possible histories becomes intractable
- Particularly problematic for continuing (non-episodic) problems

### 2. Belief States

A more sophisticated approach is to maintain a probability distribution over the possible true states given the history. This is known as a **belief state**:

$s = p(x|h)$ for all possible x

When new actions are taken and observations received, the belief state can be updated using Bayes' rule:

$$p(x'|h') = p(x'|o',a,h) = \frac{p(o'|x',a,h)p(x'|a,h)}{p(o'|a,h)}$$
This can be expanded to:
$$p(x'|h') = \frac{p(o'|x',a)\sum_x p(x'|x,a)p(x|h)}{\sum_{x'} p(o'|x',a)\sum_x p(x'|x,a)p(x|h)}$$
Where:

- $p(o'|x',a)$ is the observation model
- $p(x'|x,a)$ is the transition model
- $p(x|h)$ is the current belief state

**Advantages**:

- Interpretable: clear meaning as probabilities over true states
- Relatively compact: dimensionality depends on state space size, not history length
- Recursive update possible without storing full history

**Disadvantages**:

- Requires known observation and transition models
- These models can be difficult to learn from data
- Only practical for discrete state spaces

#### Example: Tiger Problem

In the Tiger problem:

- There are two doors; behind one is a tiger, behind the other is treasure
- The agent can: listen for the tiger, open the left door, or open the right door
- Listening gives a noisy observation (correct 85% of the time)
- Opening the door with the tiger is heavily penalized (-100), while finding treasure is rewarded (+10)
- Listening has a small cost (-1)

Initially, the belief might be 50% probability the tiger is on the left. After hearing the tiger on the left (which is 85% accurate), the belief updates to approximately 85% probability the tiger is on the left.

Planning in belief space yields an optimal policy: listen until the belief becomes strong enough, then open the appropriate door.

### 3. Predictive State Representations (PSRs)

Instead of tracking beliefs over latent states, PSRs focus on predictions about future observations:
$$f(h) = f_{oa}(h) := Pr\{O_{t+1} = o|H_t = h, A_t = a\}$$

Or more generally, we can consider "tests" (sequences of actions and observations): $\tau = a_1o_1a_2o_2a_3o_3$

And define the probability that a test "succeeds" given history: 

$$p(\tau|h) = Pr\{O_{t+1} = o_1, O_{t+2} = o_2, O_{t+3} = o_3|H_t = h, A_t = a_1, A_{t+1} = a_2, A_{t+2} = a_3\}$$

For certain sets of "core tests," the vector of their probabilities forms a Markov state.

**Advantages**:

- Test probabilities can be learned from data
- Can be as compact or more compact than belief states
- Can still be updated recursively

**Disadvantages**:

- Still primarily applicable to tabular settings (though extensions exist)

### 4. Recent Observations (Frame Stacking)

A simpler, approximate approach is to use the k most recent observations: $S = (O_{t-k}, A_{t-k}, ..., O_{t-1}, A_{t-1}, O_t)$

Example: In the Atari DQN paper (Mnih et al., 2013), the authors used a stack of 4 recent frames as input.

**Advantages**:

- Very simple to implement
- Works well for many practical applications

**Disadvantages**:

- Potentially loses information from observations beyond the window
- Not necessarily Markov (may not capture all relevant history)
- Might not be compact if observations are high-dimensional

### 5. End-to-End Learning with Recurrent Networks

Modern deep learning approaches often use recurrent neural networks (RNNs) like LSTMs to process the history:

$o_t, a_t \rightarrow \text{RNN} \rightarrow s_t \rightarrow \text{Policy/Value}$

Example: Deep Recurrent Q-Learning (Hausknecht & Stone, 2015) combines convolutional layers with LSTM layers to handle partial observability.

**Advantages**:

- Can theoretically capture dependencies over the entire history
- No fixed window size like in frame stacking
- Automatically learns relevant features from raw observations

**Disadvantages**:

- RNN training can be challenging (local optima, vanishing/exploding gradients)
- Requires more data and computation than simpler approaches
- Still not guaranteed to be Markov

## Trade-offs and Practical Considerations

When choosing an approach to handle partial observability, consider these trade-offs:
1. **Compactness**: How efficiently does the representation scale with history length?
2. **Markov property**: Does the representation capture all relevant information?
3. **Interpretability**: Is the representation meaningful to humans?
4. **Computational complexity**: How expensive are updates and learning?
5. **Ease of implementation**: How difficult is it to implement and debug?

In practice, approximate methods like frame stacking or recurrent networks are often preferred for complex environments, even though they don't theoretically guarantee optimal performance.

Even when using function approximation, there's typically no guarantee that features define a perfect Markov state. The key is finding a representation that's "close enough" to Markov to enable effective learning and decision-making.
## Conclusion

Partial observability introduces significant challenges to reinforcement learning. The agent must maintain some internal representation that summarizes relevant aspects of history to make good decisions.

While exact methods like belief states and PSRs provide theoretical guarantees, they often have practical limitations. Approximate methods like frame stacking and recurrent networks offer pragmatic alternatives that work well in many applications.

The choice of approach depends on the specific requirements of the task, available computational resources, and whether known models are available or must be learned from data.

Partial observable MDP’s do not have all relevant information from history in the observations. Thus, an internal state has to be extracted from the history.
Trade-off between various factors:
- Compactness
- Markov property
- Interpretability
- Computational complexity of updates, learning
- Ease of implementation

## Need to know
- What is a state update function and why do we need it?
- What are the advantages and disadvantages of the discussed state update functions?
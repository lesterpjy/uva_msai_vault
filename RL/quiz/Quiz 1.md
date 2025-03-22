---
sources:
  - "[[12 - Model-based Reinforcement Learning]]"
---
> [!question] Briefly explain the core advantage of model-based reinforcement learning methods.
>> [!success]- Answer
>> The core advantage is reduced sample complexity, meaning fewer real-world interactions are needed to learn an optimal policy.  This is achieved by using a model of the environment to generate synthetic experiences for learning and planning.

> [!question] Which of the following are key processes in model-based reinforcement learning?
> a) Model Learning
> b) Planning
> c) Acting
> d) Data Collection
> e) Policy Evaluation
>> [!success]- Answer
>> a) Model Learning
>> b) Planning
>> c) Acting

> [!question] Which of the following is NOT a compelling reason to use model-based reinforcement learning approaches?
> a) Data efficiency
> b) Access to additional information such as counterfactuals
> c) On-policy limitations in certain environments
> d) Guaranteed optimal policies
>> [!success]- Answer
>> d) Guaranteed optimal policies

> [!question] Model-free RL methods learn policies or value functions directly from experience without explicitly modeling the environment.
>> [!success]- Answer
>> True

> [!question] Dyna-Q combines direct RL updates with model-based planning, improving sample efficiency by reusing past experience through the `____`.
>> [!success]- Answer
>> model

> [!question] Compare and contrast the 'planning ahead of acting' and 'planning while acting' approaches in model-based RL, providing examples of each and discussing their relative strengths and weaknesses.
>> [!success]- Answer
>> In 'planning ahead of acting,' a complete policy is learned using the model before any interaction with the real environment occurs. Dyna-Q is an example; it pre-computes a policy using a model before execution.  This approach is efficient in simulation but might not perfectly generalize to the real world if the model is inaccurate.  In 'planning while acting,' planning is done dynamically, focusing on the current state and likely successors. MCTS and AlphaGo exemplify this approach; they plan only for the current game state.  This approach adapts better to the real world, since actions are informed by the current state, but it is computationally more demanding, because planning needs to be done in real time.


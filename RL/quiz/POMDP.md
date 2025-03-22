---
sources:
  - "[[13 - Partial Observability]]"
---
> [!question] The simplest approach to representing an agent's internal belief in a POMDP is to use the entire history of interactions, which can be represented as `____` where At represents actions and Ot represents observations.
>> [!success]- Answer
>> H_t = A_0, O_1, ..., A_{t-1}, O_t

> [!question] Compare and contrast the advantages and disadvantages of using belief states, Partially Observable Sequential Representations (PSRs), and methods like frame stacking or recurrent neural networks for handling partial observability in reinforcement learning.  Discuss the trade-offs involved in choosing an approach, considering compactness, the Markov property, interpretability, computational complexity, and ease of implementation.
>> [!success]- Answer
>> Belief states offer interpretability and relative compactness by representing uncertainty as a probability distribution over states, but require known transition and observation models and are often impractical for large state spaces. PSRs offer a way to learn from data and can be compact, but remain primarily applicable to tabular settings. Frame stacking and recurrent networks are practical approximations, simple to implement, and effective in many applications, but may lose information, not strictly adhere to the Markov property, and lack interpretability.  The choice depends on the specific task and available resources.  If models are known and the state space is small, belief states may be suitable.  If data is abundant, PSRs or recurrent networks might be preferable for their capacity to learn from experience. For simpler scenarios or when computational resources are limited, frame stacking could be a good balance between performance and complexity.  Ultimately, the trade-off is between theoretical guarantees (belief states and, to some extent, PSRs) and practical applicability and scalability (frame stacking and recurrent networks).  The ideal method will often be context-specific, prioritizing what aspects are most important given the problem at hand.

> [!question] Which of the following are advantages of using belief states to represent an agent's knowledge in a POMDP?
> a) Interpretable
> b) Relatively compact
> c) Recursive update possible without storing full history
> d) Requires no known observation or transition models
> e) Works well with continuous state spaces
>> [!success]- Answer
>> a) Interpretable
>> b) Relatively compact
>> c) Recursive update possible without storing full history

> [!question] In a Partially Observable Markov Decision Process (POMDP), the agent has complete access to the true state of the environment.
>> [!success]- Answer
>> False

> [!question] Which of the following is NOT a common way in which partial observability manifests in POMDPs?
> a) Aliasing
> b) Noise
> c) Perfect Information
> d) Missing Information
>> [!success]- Answer
>> c) Perfect Information

> [!question] What is a belief state in the context of POMDPs, and why is it a more sophisticated approach compared to using the entire history of interactions?
>> [!success]- Answer
>> In POMDPs, a belief state is a probability distribution over possible true states given the history of observations and actions. It's more sophisticated than using the entire history because it's more compact, representing the agent's uncertainty using probabilities instead of the full history, which becomes intractable with long histories.  This compactness allows for more efficient planning and decision making.


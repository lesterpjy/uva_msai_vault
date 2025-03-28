## Foundation of Adversarial Machine Learning

Adversarial machine learning explores vulnerabilities in machine learning models by studying how intentionally crafted inputs can cause models to make errors. This field is important for understanding model robustness and security implications.

### The Basic Concept of Adversarial Examples

An adversarial example is created by taking an input signal and adding a carefully crafted perturbation to produce an adversarial example:

Input Signal + Adversarial Corruption = Adversarial Example

The key insight here is that these adversarial corruptions aren't random noise—they're "learned" and specifically designed to trick the model while being imperceptible or subtle to humans.

## Classification of Adversarial Attacks

Adversarial attacks can be categorized along several dimensions:

### 1. Target Knowledge (Attack Setting)

- **White-box attacks**: Attackers have full access to the model parameters and can leverage gradients directly to generate perturbations
- **Gray-box attacks**: Attackers have partial information about the model
- **Black-box attacks**: Attackers can only query the model for outputs and have no access to internal parameters or gradients

### 2. Attack Objective

- **Targeted attacks**: The goal is to make the model predict a specific incorrect target class
- **Untargeted attacks**: The goal is simply to make the model predict any incorrect class
Untargeted attacks <span style="color:rgb(172, 96, 230)">maximize the loss for the correct class</span> (pushing away from correct classification), while targeted attacks <span style="color:rgb(172, 96, 230)">minimize the loss for the target class</span> (pulling toward the desired incorrect classification)

### 3. Perturbation Constraints

In the context of adversarial attacks, <span style="color:rgb(172, 96, 230)">threat models formalizes</span>:

- **Objective** of the attack
- **Imperceptibility** requirements (how noticeable the changes are)
- **Strength** of perturbation (usually constrained by some norm)
- **Budget** (time/iteration/compute/query constraints)

A common formalization is the <span style="color:rgb(172, 96, 230)">Lp-norm threat model</span>, which <span style="color:rgb(172, 96, 230)">constrains the perturbation's magnitude using different norms</span>:

- L∞-norm: Limits the maximum change to any feature
- L2-norm: Limits the Euclidean distance between original and perturbed examples
- L1-norm: Limits the sum of absolute changes across all features

## White-Box Attack Methods

### Fast Gradient Sign Method (FGSM)

FGSM (Goodfellow et al., 2015) is a simple one-step method for generating adversarial examples:

1. Calculate the gradient of the loss with respect to the input: ∇ₓJ(θ, x, y)
2. Take the sign of this gradient: sign(∇ₓJ(θ, x, y))
3. Create the adversarial example: x' = x + ε·sign(∇ₓJ(θ, x, y))

Where:

- x is the original input
- y is the true label
- θ represents the model parameters
- J is the loss function
- ε is a small constant that controls perturbation magnitude

### Basic Iterative Method (BIM)

BIM (Kurakin et al., 2017) extends FGSM by applying it iteratively in smaller steps:

1. Initialize x₀' = x
2. For i = 1...k:
    - Calculate gradient: δ = sign(∇ₓJ(f, θ, xᵢ₋₁', y))
    - Update: xᵢ' = xᵢ₋₁' + α·δ
    - Project back onto constraint region: xᵢ' = clip(xᵢ', x-ε, x+ε)

Here α is a smaller step size (typically α < ε).

### Projected Gradient Descent (PGD)

PGD (Madry et al., 2018) is considered one of the strongest first-order attacks and is widely used as a benchmark for adversarial robustness:

1. Initialize x₀' = x + random noise (within constraints)
2. For i = 1...k:
    - Calculate gradient: δ = sign(∇ₓJ(f, θ, xᵢ₋₁', y))
    - Update: xᵢ' = xᵢ₋₁' + α·δ
    - Project back to Lp ball: xᵢ' = Πₚ(xᵢ', ε) (enforces the Lp threat model constraints)

The <span style="color:rgb(172, 96, 230)">key difference from BIM is the random initialization</span>, which helps explore different regions of the loss landscape and find stronger adversarial examples.

## Black-Box Attacks

Black-box attacks are more challenging as they don't have access to gradients. Examples include:

### One-Pixel Attack

The One-Pixel Attack (Su et al., 2019) shows that modifying just a single pixel can fool deep neural networks. The slides show examples where changing one pixel can cause a classifier to misidentify objects like "SHIP" as "CAR" with 99.7% confidence or "HORSE" as "FROG" with 99.9% confidence.

## Defending Against Adversarial Attacks

### Requirements for Defenses in IR
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Be applicable during training or inference phase
- Maintain or enhance performance of IR models
- Guarantee stability of top-K results
### Defense Techniques
**General**
- <span style="color:rgb(172, 96, 230)">Adversarial Training</span>: Incorporating adversarial examples during training
    - Data augmentation with adversarial examples
    - Curriculum learning approaches (gradually increasing attack strength)
- <span style="color:rgb(172, 96, 230)">Wide Decision Boundaries</span>: Making models less sensitive to small perturbations
    - Stochastic features to add robustness
    - Theoretical loss function adjustments
**IR**
- <span style="color:rgb(172, 96, 230)">Empirical Defenses</span> (training phase): <span style="color:rgb(172, 96, 230)">developed and validated through experimental data and observation, make models empirically robust to known adversarial attacks</span>.
    - Data augmentation: Generate multiple variations of training documents
    - Traditional adversarial training: Train models on adversarial docs that have random synonyms mixed in.
    - <span style="color:rgb(172, 96, 230)">Simple & low cost but non-targeted and limited in effectiveness</span>.
- Certified Defenses (training phase)
    - Provide provable guarantees of robustness within certain constraints
- <span style="color:rgb(172, 96, 230)">Attack Detection</span> (inference phase)
    - Learning-based detection: Train models to identify adversarial inputs
    - <span style="color:rgb(172, 96, 230)">Threshold-based detection</span>:
        - Perplexity: Adversarial texts often have abnormally high perplexity
        - L2 norm: Gradient-based attack on bi-encoder retriever relying on dot-product causes increase in L2 norm.

## Adversarial Information Retrieval
Imperceptibility on adversarial IR: if user find the adversarial document, is the attack successful?

### Information Retrieval Pipeline

A typical IR system has two main stages:

1. **Retrieval**: Find an initial set of candidate documents for a query
2. **Ranking**: Determine the relevance degree of each candidate

- **Attack objective**: <span style="color:rgb(172, 96, 230)">Make irrelevant documents appear at the top of results</span>
- **Defense objective**: <span style="color:rgb(172, 96, 230)">Protect the stability of top-K results</span>

### Classification of IR Attacks

1. **Adversarial retrieval attack**: Get a target document <span style="color:rgb(172, 96, 230)">outside top-K to appear within top-K</span>
2. **Adversarial ranking attack**: <span style="color:rgb(172, 96, 230)">Promote the ranking of a document already in top-K</span>

### Attacks

#### Typical attack (Imperceptible)
Topic-oriented adversarial retrieval/ ranking attack
- <span style="color:rgb(172, 96, 230)">Paired attack: Promotes a target document for a specific query</span>
	- ![[paired_attack.png | 400]]
- <span style="color:rgb(172, 96, 230)">Topic-oriented attack</span>: Promotes a target document in rankings on each query in the group with the same topic.
	- ![[target-oriented-attack.png | 500]]
#### Steps of Adversarial Attacks
General workflow for creating adversarial attack in IR:

1. <span style="color:rgb(172, 96, 230)">Steal knowledge from black-box models</span>
    - Train surrogate models by querying the target model
2. <span style="color:rgb(172, 96, 230)">Identify vulnerable positions in documents</span>
    - Gradient-guided approach: Calculate gradients to find important tokens
    - Output-guided approach: Analyze model outputs to identify sensitive areas
3. <span style="color:rgb(172, 96, 230)">Add perturbations to identified positions</span>
    - Perturbation type (word-level, sentence-level, etc.)
    - Perturbation strategy (static greedy search or dynamic approaches)

For example, in a document about Star Trek, replacing "begin" with "began" might change the document's ranking from 98th to 54th position, while replacing a sentence could improve it to 22nd position.

#### Surrogate Model Training
A common approach for black-box attacks.
<span style="color:rgb(172, 96, 230)">Objective: Train a surrogate white-box model to steak target model knowledge</span>
<span style="color:rgb(172, 96, 230)">Approach: Continuously query target model to gather input-output pairs</span>

1. Query the target model to gather input-output pairs
2. Train a surrogate model that mimics the target model's behavior
3. Perform white-box attacks on the surrogate model
4. Transfer these adversarial examples to attack the target model

Surrogate model training can effectively transfer vulnerabilities from the target model.

#### Gradient Guided Position

<span style="color:rgb(172, 96, 230)">Gradient-guided position</span>: Calculate the gradient on the surrogate model to backpropagate to document tokens and identify important positions by large gradients.
1. Input the target document (with query) into the surrogate model.
2. Calculate gradients by the loss function and back-propagate to the token embedding layer.
3. Find tokens with large gradients as vulnerable positions in the document

#### Perturbation types
- Selecting perturbation type is a <span style="color:rgb(172, 96, 230)">trade-off between attack effectiveness and naturalness</span>
- Different scenarios and different query-document pairs suit different types of perturbations.
![[perturbation_types.png | 400]]

#### Greedy-based strategy
For each perturbation position, candidate perturbations are tried in turn, and the <span style="color:rgb(172, 96, 230)">one with the highest rank improvement is selected as the final perturbation</span> for the current position.

![[greedy-adversarial-search.png | 280]]

## Attacking LLMs

- Prompt-based attack: find triggers by gradient to jailbreak LLMs.

## Attacks on Retrieval-Augmented Generation (RAG)

### Attack Objectives on RAG

1. <span style="color:rgb(172, 96, 230)">Jailbreak</span>: Bypass LLM's safety mechanisms to generate forbidden content
2. <span style="color:rgb(172, 96, 230)">System Disruption</span>: Cause RAG to generate incorrect responses

### Attack Vectors

1. <span style="color:rgb(172, 96, 230)">Query Manipulation</span>: Modify user queries to trigger unsafe responses
2. <span style="color:rgb(172, 96, 230)">Corpus Poisoning</span>: Inject malicious content into the knowledge base

Corpus poisoning is particularly concerning because it's hard to defend against. An example shown is injecting false information that "Tim Cook is the CEO of OpenAI" into the knowledge base, which can cause the RAG system to confidently provide this incorrect information when asked about OpenAI's CEO.

## Conclusion

Adversarial machine learning highlights the vulnerabilities in current AI systems. Understanding these vulnerabilities is crucial for:

1. Building more robust models
2. Ensuring deployed systems are trustworthy
3. Advancing the field toward more secure AI

The field continues to evolve, especially as systems grow more complex with the integration of retrievers and large language models in frameworks like RAG.
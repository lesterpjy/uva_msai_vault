## 1. Introduction and Motivation

Information Retrieval (IR) systems have become increasingly complex, especially with the integration of neural networks and deep learning approaches. This complexity has created a need to explain and interpret how these systems make decisions.

The lecture begins with several motivating examples that demonstrate the importance of explainability:

1. **The "Husky vs Wolf" example** - A model correctly distinguished between huskies and wolves in images, but for the wrong reasons. The model was actually detecting snow in the backgrounds (wolves were typically photographed in snowy settings) rather than the animal features.
    
2. **The Argument Reasoning Comprehension task** - BERT achieved 77% accuracy (close to human performance), but further investigation revealed it was exploiting spurious statistical cues in the dataset rather than understanding the arguments. Words like "not," "is," or "do" could predict the correct answer better than random chance, and when these cues were removed, performance dropped to chance level.
    

These examples highlight that models might perform well for the wrong reasons, emphasizing why explainability is crucial.

## 2. Goals and Users of Explainability

### Goals of Explainability:

- Gain insight into the decision-making process of AI models
- Determine whether to trust model predictions
- Debug and improve existing models and training processes
- Maintain control as AI research progresses rapidly
- Audit AI systems deployed in real-world settings

### Users of Explanations:

- End users who interact with the system
- Managers and policy makers who make decisions about deployment
- Engineers who build and maintain the systems

The appropriate type of explanation depends on who will be using it, as different stakeholders have different needs and technical backgrounds.

## 3. Types of Explainability in IR

The lecture identifies three main approaches to explainability in IR:

1. **Interpretable by design** - Creating models that are inherently interpretable
2. **Understanding model decisions based on input data** - Explaining how specific inputs affect outputs (feature attribution)
3. **Understanding the inner workings of the model** - Analyzing internal components and representations (mechanistic interpretability)

Additionally, the lecture distinguishes between different levels of explanations in ranking:

- **Pointwise explanations**: Why does a document receive a high ranking score?
- **Pairwise explanations**: Why is document A ranked higher than document B?
- **Listwise explanations**: Why are documents ranked in a particular order?

## 4. Feature Attribution Explanations (Part 1)

Feature attribution explanations aim to determine which input features contribute most significantly to a model's prediction.

### Definition:

A feature attribution explanation is a dictionary that assigns importance values ($\phi_i$) to each input feature i: ${i \mapsto \phi_i}_i$

These attributions can be visualized across different modalities (text, images, tabular data).

### Key Concept: Input Perturbation

The central idea is to perturb input features and observe how these changes affect the output. Features that cause significant changes when perturbed are considered more important.

### Challenges in Feature Attribution:

- **How to perturb features?** Options include:
    
    - Removing features (but models aren't trained for incomplete inputs)
    - Masking features (with zeros, means, or samples from data)
    - Risk of creating out-of-distribution examples
- **Listwise masking complexity:**
    
    - Should all documents be masked simultaneously or individually?
    - How to explain information not present in the results?
    - How to measure output similarity for complex model outputs?

### Case Study: Talent Search White Box Model

To illustrate these concepts, the lecture presents a white box model for talent search with these components:

- Features: skills, experience, grade, university, and job requirements
- The model calculates scores based primarily on skills and experience
- A condition checks if job requirements are met; if not, the score is multiplied by 0.1
- In a biased version, if the candidate is from a specific university, the score is multiplied by 0.7

This simple model provides a clear framework to explore explainability techniques.

## 5. Shapley Values and SHAP

### Shapley Values

The lecture introduces Shapley values from cooperative game theory using a taxi fare example:

- Different combinations of passengers (Alice, Bob, Charlie) result in different costs
- Shapley values determine the fair contribution of each passenger by examining their marginal contributions across all possible coalitions

### SHAP (SHapley Additive exPlanations)

SHAP applies the Shapley value concept to explain model predictions:

- Features are treated as "players" in a game
- The "prize" is the predicted ranking score
- "Coalitions" are simulated by masking features not in the coalition

The SHAP value for a feature i is calculated as: $$\phi_i(x) = \sum_{S\subset{1,...n}\setminus i} w_S \cdot E_{b\sim B}[\tilde{R}(\text{mask}_{S\cup{i},b}(x)) - \tilde{R}(\text{mask}_{S,b}(x))]$$

Where:

- $w_S$ is a weight based on coalition size
- $B$ represents background data used for masking
- $\text{mask}_{S,b}$ masks out features not in set $S$ with samples from background data $B$
- $\tilde{R}$ is the model's prediction function

### Listwise Feature Attribution - RankingSHAP

For ranking systems, standard SHAP needs to be extended. RankingSHAP adapts SHAP for listwise ranking models:

4. **Value definition**: Defines a listwise explanation objective that reduces the model prediction to a single value, such as rank similarity: $g_q(\tilde{\pi}) = \tau(\pi_q, \tilde{\pi})$
    
5. **Masking approach**: Applies the same mask to each document feature vector: $\text{mask}_{t,b}(D_q) = \prod_{|D_q|} \text{mask}_{t,b}(x_{q,i})$
    

The case study shows how RankingSHAP provides more contrastive explanations compared to pointwise SHAP, especially for negatively biased queries.

## 6. Understanding Model Internals (Part 2)

The second part of the lecture focuses on analyzing the internal components of models to understand how they process information.

### Why Analyzing Model Internals Is Important:

- Input-output relationships only capture surface-level information
- Models develop complex internal representations
- Information is highly contextualized in attention-based models
- Models might detect complex patterns like sarcasm that aren't visible through simple input-output analysis

### Key Concepts in Model Internal Analysis:

- **Internal representations**: The embedding space, latent space, and activation space of the model
- **Subgraph (Circuit)**: A subset of neurons, layers, and connections within a neural network
- **Features**: Properties or characteristics of the input data that may be used in the task

### Methods for Analyzing Model Internals:

#### 1. Probing:

- Freeze the pretrained model
- Attach simple classifiers to each layer
- Train these classifiers on a specialized dataset
- Limitations: Provides correlation-based insights and requires additional training

#### 2. Activation Patching:

- Identifies important neurons or layers by modifying their activations
- Observes how these modifications affect the output

#### 3. Path Patching:

- Identifies connections between neurons and layers (circuits)
- Requires two inference runs (clean and corrupted)
- Replaces activations from the clean run with those from the corrupted run for specific modules
- Measures the difference in performance to identify crucial pathways

The module being patched could be an entire layer or subcomponents like the residual stream, MLP layer, or attention heads/layers.

### Case Study: Axiomatic Causal Interventions in Neural Retrieval Models

The lecture examines a study that investigates whether neural retrieval models implicitly encode IR axioms such as TFC1:

TFC1 axiom: For a single-term query, if two documents of equal length contain different counts of the query term, the document with more occurrences should be ranked higher.

The study applies path patching to investigate how term frequency affects relevance in neural retrieval models, using:

- MS-MARCO Passage Ranking dataset
- TAS-B bi-encoder model

#### Key Findings:

6. Query terms already present in the document are most important for performance
7. Information aggregates in the CLS token in the later layers
8. Term positions matter - terms at the beginning of a document have higher impact
9. Specific attention heads focus on relevant documents, but not on irrelevant ones

## 7. Challenges and Future Work

### The "Disagreement Problem" in Explainable AI:

- Multiple explanation methods often produce conflicting explanations
- A study found that 88% of data scientists use multiple explanation methods, and 84% encounter disagreements between methods
- Low agreement between different attribution methods on the same data

### Challenges Specific to IR:

- Complex model inputs and outputs make it difficult to define what constitutes an explanation
- No ground truth for evaluations
- The field strongly emphasizes evaluation metrics

### Future Work:

- Developing better evaluation methods for explanations in IR
- Extending explainability to new areas like RAG and Generative IR
- Explaining information that is not present in results
- Breaking down IR tasks into subtasks for mechanistic interpretability
- Understanding how information is encoded in models to improve IR tasks

## 8. Summary

- Listwise ranking models require listwise explanations due to complex inputs and outputs
- Model internals provide insights into feature encoding
- Lexical tf-idf features can be found in dense-retrieval encoders
- The position of query terms affects retrieval performance
- More research is needed to advance explainability in IR

The field of explainable IR (XIR) faces challenges but offers promising opportunities for improving model understanding, debugging, and bias mitigation.
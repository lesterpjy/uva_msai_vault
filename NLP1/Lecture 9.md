## Interpretability of NLP Models
Interpretability often generates explanations of model behavior.
Explanations can be:
- Local: about one specific input
- Global: about the model’s behavior across all inputs
Explanations should be faithful, i.e. explanations should reflect the underlying model mechanism behind the behavior they explain.


**Interpretability Hierarchy**
![[interpretibility_hier.png]]

## Subject-Verb Agreement Case Study

**Behavioral interpretability**
1. Create dataset, ex. sentences like "the game that the guards hate [MASK] bad"
2. Define metric to measure performance, p(agree) - p(disagree), ex. p(is) - p(are)

Pros
- Easy to adapt to any task
- No need access to model internals
Cons
- Don't know how
- Can't be sure we covered all possibilities in dataset

**Representational Analysis**
Probing
1. Craft a dataset consisting of tokens / activations and corresponding labels
2. Train (for each layer of the model) a probe, mapping from activations to labels
3. If the probe gets high accuracy, success

Pros
- Lets us test hypotheses re: model internals
- Pretty simple to implement
- Doesn’t require very specialized data
Cons
- High probing accuracy $\neq$ mechanistic relevance 
	- Probing is right: BERT uses subject number for SVA in a linguistically weird way
	- Probing is wrong: BERT doesn’t use subject number information as it suggests

**Causal Interventions**
Activation Patching
![[activation_patching.png | 300]]
- If model uses a representation when making its decision, replacing the representation will change its decision
- Pro: causal evidence about whether our model uses a given activation
- Con: very restrictive setup
Probing Interventions
Probe with binary linear classifier have linear decision boundaries -> change representations wrt the boundary to see if probe captures relevant info


## Attributions
An x attribution means that you want to find the x that is important for your model’s behavior.

- Layer attribution: what layers are most important to my model’s behavior on a given task?
	- Causal framing: If ablated/patched, what layer would cause largest change in model ability
- Input attribution: what input tokens are important to my model’s behavior on this input?
	- Causal framing: What tokens would cause the largest change in my model’s output if I masked them / zeroed them out / replaced them?
- Data attribution: what training datapoints are most important to my model’s behavior on a given input?
	- Causal framing: What training datapoints would most change my model’s behavior on a given input if removed from the training dataset?

Pros:
- Eas(ier) for users to understand
- Flexible: compatible with various tasks / inputs
- Layer and input attribution are easy to implement
Cons:
- Relies a lot on post-hoc interpretations: Can you really infer a model’s mechanisms from the tokens/datapoints it relies on?
- Results can vary depending on how you ablate things
- Data attribution is quite hard
- Causal attribution is only one kind of attribution]
## Circuits

We want explanation that is faithful, total explanation, does not require strong hypothesis.

**Definition:** A **circuit** is the minimal computational subgraph of a model that is faithful to model performance on a given task. It functions like a map, highlighting the specific parts of the model that collaborate to perform a specific task.

**Key Components of Circuits**

- **Computational Subgraphs**: Circuits are grounded in the architecture of transformer language models (LMs), particularly the residual stream, which is characterized by each component's output being added to the subsequent component's input.
- **Transformer Architecture and Residual Streams:** Circuits research mainly focuses on autoregressive language models, which predict the next word in a sequence. These models use a transformer architecture with left-to-right attention. The **residual stream** is a core feature where the output of each component is added to the input of the next one.
- **Rethinking the Computational Graph:** The residual stream view helps us see the model's components as interconnected nodes within a flow of information. This highlights how each component's input is a combination of all previous outputs.
- **Finding Minimal Subgraphs:** By tracing the information flow related to a specific task, we can pinpoint the smallest set of components—the **circuit**—that contribute to that behavior.
- **Benefits of the Residual Stream View:**
    - It provides a clear picture of information flow within the model.
    - It helps find the smallest set of essential components, making circuits easier to understand.
    - It allows us to analyze different tasks and understand how different parts of the model contribute to various behaviors.

- **Tasks**: It's crucial to clearly define the task you want the circuit to execute. This includes:
    - **Inputs**: The specific text provided to the model.
    - **Expected Outputs**: The desired result the model should produce given the input.
    - **Metric**: A quantifiable measure for assessing how effectively the model performs the task.
    - **Corrupted Inputs**: Modified versions of the original input designed to elicit distinct model outputs. This aids in identifying the circuit components.
- **Faithfulness**: A circuit is valuable only if it accurately mirrors the model's actual behavior. **Faithfulness** means that removing all components _outside_ the identified circuit should have minimal impact on the model's performance on the task.

**Methods for Finding Circuits**

1. **Activation Patching**: This technique aims to determine if a specific component (e.g., an attention head or an MLP) is essential for a task. Instead of zeroing out the activation of the component in question, activation patching replaces it with the activation from the corrupted input—an input crafted to trigger a different model behavior for the assigned task.
2. **Edge Patching**: This method specifically examines the importance of the connection between two components. To patch the edge from component A to component B, you exploit the linear nature of the residual stream. You take the input going into component B, subtract the output of component A from the normal input, and then add the output of component A when it's processing the corrupted input. This replaces the information flowing from A to B with information based on the corrupted input, isolating the impact of that specific connection.
3. **Iterative Patching**: This approach systematically patches activations, starting at the output and progressing backward to the input, allowing the identification of an entire circuit.

**Validating Circuit Faithfulness**

- Once you've identified a potential circuit, it's necessary to confirm its faithfulness. This can be done by corrupting all elements of the model _except_ the proposed circuit and then observing whether the model's performance remains consistent.

**Evaluating Identified Circuits**

- **Necessariness**: Assess whether the circuit is truly indispensable for the given task by checking if the model's performance deteriorates upon removal of the circuit alone.
- **Completeness**: Determine if you have discovered all relevant components, including those that might have a negative impact on the task.
- **Minimality**: Ascertain whether all components within the circuit are truly essential by testing whether removing any of them affects performance.

**Understanding Circuit Semantics**

- **Logit Lens**: This method helps visualize model activations in the context of vocabulary, facilitating the understanding of how different parts of the circuit contribute to the final output.

**Advantages of Circuits**

- Faithfully represent model behavior, providing a reliable explanation.
- Offer a complete explanation of the model's behavior for a particular task, encompassing the entire relevant model mechanism, not just isolated components.
- Finding circuits doesn't require starting with specific hypotheses.

**Disadvantages of Circuits**

- Require meticulous task setup, and the identified circuit may only function within that specific context.
- The circuit you uncover depends on how the difference between normal and corrupted inputs is defined.
- Don't provide detailed insights into how individual features contribute to the task at hand.

**Overall**: Circuits provide a valuable tool for understanding the inner workings of language models. However, it's important to remember that research on circuits is still in its early stages, and developing techniques to fully understand how circuits operate remains an active challenge.

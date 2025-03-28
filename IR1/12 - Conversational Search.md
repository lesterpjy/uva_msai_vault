## Introduction to Conversational Search

Conversational Search (CS) represents a significant evolution in information retrieval systems, moving beyond traditional query-response paradigms toward interactive, dialogue-based information access.

### What is Conversational Search?

Conversational Search can be defined as an interactive communication process for exchanging information between participants (humans or machines) that involves a sequence of interactions aimed at satisfying information needs. While natural language is a prerequisite for conversational interactions, conversations can also exhibit other types of interactions with different characteristics and modalities.

As noted by Zamani et al. (2022), "Information seeking conversation is a conversation in which the goal of information exchange is satisfying the information needs of one or more participants."

### Motivations for Conversational Search

Traditional search engines limit user interactions to predefined actions like "requery" or "click on a document." The shift toward conversational information retrieval systems offers several advantages:

1. <span style="color:rgb(172, 96, 230)">Richer expression</span> of information needs
2. <span style="color:rgb(172, 96, 230)">Free-form natural language interactions</span>
3. Ability to provide <span style="color:rgb(172, 96, 230)">feedback in a conversational manner</span>
4. More <span style="color:rgb(172, 96, 230)">natural interactions</span> similar to human conversations

### Retrieval Augmented Generation (RAG) as Foundation

Conversational Search builds upon Retrieval Augmented Generation (RAG), which combines retrieval and generation components:

1. **Knowledge Corpus**: Contains documents indexed offline
2. **Retriever**: Processes user query to find relevant documents
3. **Generator**: Uses retrieved documents and query to generate responses

## Core Challenges and Requirements in Conversational Search

### Key Requirements for Conversational Search Systems

1. <span style="color:rgb(172, 96, 230)">Statefulness</span>: The system must maintain conversation history and context across multiple turns
2. <span style="color:rgb(172, 96, 230)">Fluency and Conciseness</span>: Present results clearly with high comprehensibility
3. <span style="color:rgb(172, 96, 230)">Mixed-Initiative Interactions</span>: Pro-actively participate in the conversation
4. <span style="color:rgb(172, 96, 230)">Personalization</span>: Use interaction history and user-specific information for context-aware responses

### Workflow of Conversational Search

![[conversational_search_workflow.png | 300]]

A typical workflow includes:

1. Processing the current question along with historical context
2. Query reformulation to create a searchable query from conversation context
3. Conversational retrieval to find relevant information
4. Response generation to provide natural language answers
5. Optional clarification questions when needed

### Core Semantic Challenges

Conversational search must address <span style="color:rgb(172, 96, 230)">complex semantic phenomena</span>:

1. <span style="color:rgb(172, 96, 230)">Ellipsis</span>: When parts of a sentence are omitted because they are understood from context Example: "Who directed The Godfather?" followed by "When was it released?"
2. <span style="color:rgb(172, 96, 230)">Anaphora</span>: References to previously mentioned entities using pronouns Example: "Tell me about Barack Obama" followed by "Where was he born?"

## Query Reformulation in Conversational Search

### Purpose of Query Reformulation

<span style="color:rgb(172, 96, 230)">Query reformulation</span> (QR) is essential in conversational search because:

1. Using the full conversation history can be unwieldy (<span style="color:rgb(172, 96, 230)">long and noisy input</span>)
2. Using only the last utterance loses crucial context (<span style="color:rgb(172, 96, 230)">ambiguous and incomplete</span>)
3. Ambiguous references and implied information need resolution
4. <span style="color:rgb(172, 96, 230)">Resolve ellipsis and anaphora</span>

The goal is to transform an ambiguous conversational utterance into a <span style="color:rgb(172, 96, 230)">self-contained query</span> that can be processed by traditional retrieval systems.

### Approaches to Query Reformulation

#### 1. Supervised Learning Approaches

- Training generative models (like T5 or Llama) with datasets containing conversation-rewrite pairs
- Examples include CANARD dataset with 40,527 (conversation, rewrite) pairs
- Training with autoregressive token prediction loss

#### 2. In-Context Learning

- Using Large Language Models (LLMs) like GPT-3.5/4 with zero-shot or few-shot prompting
- No fine-tuning required, just effective prompt engineering
- Example prompt: "I will give you a conversation between a user and a system. You should rewrite the last question of the user into a self-contained query."

#### 3. Advanced Approaches: LLM4CS

<span style="color:rgb(172, 96, 230)">LLM4CS</span> uses multiple query reformulations with improvements like:

- Vocabulary expansion
- Breaking down information needs into different aspects
- More robust
- Generating responses that add relevant terms to the reformulated query
- Aggregation of rewrites

### Advantage/ Limitations
- <span style="color:rgb(146, 208, 80)">provide explicit rewrite: explainable</span>
- <span style="color:rgb(255, 82, 82)">Poor efficiency: LLM calls + several rewrites</span>
- <span style="color:rgb(255, 82, 82)">Rewrite then retrieve: error adds up, context lost</span>
- <span style="color:rgb(255, 82, 82)">Components not aligned: rewrite not optimized for retrieval, vice versa. </span>

## Representation Learning for Conversational Search

### ConvDR: Dense Representation for Conversational Queries

<span style="color:rgb(172, 96, 230)">ConvDR</span> (Conversational Dense Retrieval) moves beyond explicit rewrites to representation learning:

1. It <span style="color:rgb(172, 96, 230)">resolves queries directly in the dense vector space</span> rather than generating text rewrites
2. Uses a <span style="color:rgb(172, 96, 230)">teacher-student network</span> architecture:
    - <span style="color:rgb(172, 96, 230)">Teacher (ANCE ranker): Produces "ideal" embeddings</span> based on human rewrites
    - <span style="color:rgb(172, 96, 230)">Student: Generates embeddings directly</span> from conversation history

The training involves two key loss functions:

- <span style="color:rgb(172, 96, 230)">Knowledge Distillation Loss</span> ($L_{KD}$): Makes the student model learn to produce embeddings similar to the teacher
- <span style="color:rgb(172, 96, 230)">Ranking Loss</span> ($L_{Rank}$): Fine-tunes the model with contrastive learning using relevant documents

### DiSCo: Distillation with Contrastive Learning

<span style="color:rgb(172, 96, 230)">DiSCo</span> (Distillation with Contrastive Objectives) <span style="color:rgb(172, 96, 230)">unifies the distillation and ranking objectives</span>:

1. Uses document representations as anchors in the learning process
2. Focuses on maintaining consistent similarity scores between queries and documents
3. Creates a <span style="color:rgb(172, 96, 230)">contrastive distillation loss that optimizes both distillation and ranking simultaneously</span>

The approach uses hyperplanes where similarity with documents is constant, providing a relaxation of the training objective.

Advantages of DiSCo:

- For in-domain: more efficient and more effective
- Less effective out-of-domaim

QR through representation learning is more efficient than LLM calls
Efficiency through efficient LLMs or in-context learning query reformulation.

## Mixed Initiatives in Conversational Search

### Clarification Questions

Mixed initiative interactions involve the system taking initiative to ask clarifying questions when:

1. The original <span style="color:rgb(172, 96, 230)">request is incomplete</span>
2. It can be <span style="color:rgb(172, 96, 230)">interpreted in various ways</span>
3. <span style="color:rgb(172, 96, 230)">Ask clarification instead of returning bad results</span>

### Evaluation of Clarification Questions

Several metrics are used for evaluating clarification questions:

1. <span style="color:rgb(172, 96, 230)">Clarification need</span> prediction: How well the system determines <span style="color:rgb(172, 96, 230)">when to ask</span> (Precision, Recall, F1)
2. <span style="color:rgb(172, 96, 230)">Question relevance</span>: <span style="color:rgb(172, 96, 230)">How many </span>relevant questions are retrieved (Recall@k) what to ask?
3. <span style="color:rgb(172, 96, 230)">Document relevance</span>: <span style="color:rgb(172, 96, 230)">How much the selected question improves retrieval</span> (MRR, Precision@k, NDCG@k)

An important finding regarding negations:

- <span style="color:rgb(172, 96, 230)">Positive answers to clarification questions typically improve retrieval</span>
- Negative answers often cause performance decreases in <span style="color:rgb(172, 96, 230)">lexical-based retrieval</span> systems (e.g., BM25) because they <span style="color:rgb(172, 96, 230)">don't handle negations</span> well

### Human Evaluation

- Human annotators to judge how good a conversation is.
- Pairwise comparison: two competing systems get evaluated.
- Single- and mixed-initiative conversations.
- Correlation with automatic evaluation.
- Multiple aspects:
	- Engagement
	- Knowledgeable
	- Interest
	- Human-like
	- Clarifying questions

### Challenges

- Limited to a <span style="color:rgb(172, 96, 230)">pre-defined set of questions</span>.
- Document <span style="color:rgb(172, 96, 230)">relevance does not necessarily reflect a question’s quality</span>.
- Human in the loop is <span style="color:rgb(172, 96, 230)">expensive</span>; impossible to scale.
- Human annotation is also expensive; impossible to scale.

## User Simulation for Conversational Search

### Architecture of User Simulators

User simulators for conversational search typically include:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Natural language understanding (NLU) component
- Dialogue management with state tracking and policy
- Natural language generation (NLG)
- User modeling component

Different approaches to user simulation include:

- **CoSearcher**: Binary classification for yes/no answers
- **USi**: Supervised GPT-2 for text generation
- **ConvSim**: In-context learning with GPT-3.5 for free text generation

### ConvSim: A Modern Approach to User Simulation

ConvSim uses few-shot prompting with GPT-3.5 to simulate multi-turn conversations with:

- Information need understanding
- Answer generation for clarifying questions
- Explicit feedback capabilities
- Conversation termination logic

Benefits of ConvSim:

- Improves retrieval performance by providing <span style="color:rgb(172, 96, 230)">explicit feedback</span>
- Performs better than USi and sometimes even human interactions on metrics of naturalness and usefulness
- <span style="color:rgb(172, 96, 230)">More flexible</span> in generating diverse and natural responses

### Challenges in User Simulation

- Finding the <span style="color:rgb(172, 96, 230)">right balance of cooperativeness</span> in simulators
- Developing <span style="color:rgb(172, 96, 230)">proactive</span> user simulators (vs. purely reactive ones)
- <span style="color:rgb(172, 96, 230)">Determining</span> whether the <span style="color:rgb(172, 96, 230)">goal</span> is data creation, augmentation, or system evaluation
- Ensuring simulator behavior <span style="color:rgb(172, 96, 230)">aligns with real user behavior</span>

## Higher-Level Challenges and Future Directions

1. **LLM Integration**: As LLMs improve in handling long inputs, will query reformulation still be necessary?
2. **Efficiency Considerations**: How to balance effectiveness with computational costs
3. **Evaluation Metrics**: Developing better ways to evaluate personalization, mixed initiatives, and user engagement
4. **User Simulation Goals**: Clarifying whether simulations should primarily create data or evaluate systems
5. **Handling Negations**: Improving retrieval systems' understanding of negative responses

## Summary

Conversational Search represents an evolving paradigm in information retrieval that:

- Makes search more <span style="color:rgb(172, 96, 230)">interactive and natural</span>
- Addresses challenges in <span style="color:rgb(172, 96, 230)">query understanding</span> across <span style="color:rgb(172, 96, 230)">multiple turns</span>
- Employs techniques like <span style="color:rgb(172, 96, 230)">query reformulation</span> and <span style="color:rgb(172, 96, 230)">dense representation learning</span>
- Involves<span style="color:rgb(172, 96, 230)"> mixed-initiative interactions</span> through <span style="color:rgb(172, 96, 230)">clarification questions</span>
- <span style="color:rgb(172, 96, 230)">Can benefit from user simulation</span> for development and evaluation

The field continues to progress with advances in LLMs, representation learning, and evaluation methodologies, moving toward more natural and effective information-seeking conversations.
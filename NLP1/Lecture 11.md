
## Dialogue Modelling

### **Part 1: What is Dialogue Modeling?**

- **Definition:** Dialogue modeling aims to create computer systems that can engage in conversations with humans. This involves understanding the human's input, keeping track of the conversation history, making decisions about what to say next, and generating a response that conveys the system's intended meaning.
- **Challenges:** Building effective dialogue systems is difficult because it requires the system to possess numerous complex abilities, including:
    - Common sense reasoning.
    - Theory of mind (understanding the beliefs and intentions of others).
    - Planning.
- **Dialogue Types:** The sources distinguish between two main types of dialogue:
    - **Social Chit-Chat Dialogue:** Casual conversation with no specific goal or task.
    - **Task-Oriented Dialogue:** Conversation focused on achieving a particular goal, such as booking a flight or setting an alarm.

### Part 2: NLP Methods for Text-Based Dialogue

#### 1. Modular Statistical Approaches

- **Architecture:** These systems break down the dialogue process into separate modules, each trained independently.
- **Modules:**
    - **Natural Language Understanding (NLU):** Analyzes the user's input to extract meaning.
        - **Intent Prediction:** Determines the user's goal or intention.
            - **Challenges:** Mapping words to intent is not always straightforward, and the system may need to make inferences.
            - **Techniques:** Word embeddings, POS tagging, syntactic parsing, compositional semantics can be used for intent prediction.
            - **Data:** Annotated dialogue datasets with utterances labeled with meaning representations are needed for training.
        - **Slot Filling:** Extracts specific pieces of information (slots) related to the user's intent.
            - **Domain Classification:** Identifies the topic or area of the conversation (e.g., air travel, alarm clock).
                - Not necessary for single-domain systems.
            - **Frames:** Representations used to capture the domain, intent, and slots/values of the conversation.
    - **Dialogue Management (DM):** Keeps track of the conversation history and decides the next system action.
        - **Policy:** Predicts the system intent with the highest probability based on the dialogue context.
        - **Confirmation and Rejection:** Uses NLU confidence scores to decide whether to confirm or reject the user's input.
            - **Explicit Confirmation:** Asks the user a direct question to confirm understanding.
            - **Implicit Confirmation:** Demonstrates understanding by repeating back information or incorporating it into the next question.
        - **Learning and Generalization:** Confidence scores can also be used to identify unknown slots and adapt to new situations.
    - **Natural Language Generation (NLG):** Generates the system's response in natural language.
        - **Training:** Typically trained on an annotated dialogue corpus with representation/sentence pairs.
        - **Sequence-to-Sequence Prediction:** Input is a linearized meaning representation; output is the word string.
        - **Delexicalization:** Replacing specific words with generic placeholders to increase training data generality.
        - **Encoder-Decoder Models:** Used for mapping from frames to delexicalized sentences.

#### 2. End-to-End Encoder-Decoder Models

- **Architecture:** Inspired by machine translation, these models use RNNs for encoding the dialogue history and decoding the response.
- **Advantages:** No need for annotations or predefined meaning representations.
- **Disadvantages:** Data-hungry, lack of real understanding, tendency to generate generic responses, simplistic encoding of dialogue history.

#### 3. Generative (Decoder-Only) Large Language Models (LLMs)

- **Architecture:** Based on the Transformer architecture, allowing for better handling of dialogue history.
- **Examples:** ChatGPT, Llama.
- **Fine-tuning:** Dialogue-related instruction tuning and reinforcement learning with human feedback (RLHF) are used to turn base LLMs into dialogue agents.
- **Advantages:** Fluent text generation, more robustness, ability to leverage knowledge from pre-training.
- **Challenges:** Factual and inference errors, hallucinations, safety concerns, social bias.

### Part 3: Face-to-Face Dialogue

- **Multimodality:** Face-to-face dialogue involves various signals beyond speech, such as gestures, gaze, and facial expressions.
- **Research:**
    - Focuses on developing meaningful representations of co-speech gestures, considering their variability and relationship with speech.
    - Explores how to ground gesture representations with co-occurring speech.
    - Investigates the similarity of gestures with the same referent across different speakers and dialogue contexts.
    - Uses probing analysis to decode interpretable features from learned gesture representations.

This cheatsheet provides a structured overview of the key concepts covered in the slides. However, remember to review the original source material and your course notes for a comprehensive understanding of the topic.
## LLM Safety and Alignment

- **Importance of Safety:** Llama 2 was positioned to be the largest open-sourced language model, making its safety considerations critical. The developers aimed to open-source both the pre-trained model and an instruction-tuned chat version.
- **Cross-Functional Definition of Safety:** Defining safety for Llama 2 was a collaborative effort involving legal, policy, civil rights, ethics, and other teams.
- **Three Safety Risk Categories:**
    - **Illicit and criminal activities:** terrorism, theft, human trafficking.
    - **Hateful and harmful activities:** defamation, self-harm, eating disorders, discrimination.
    - **Unqualified advice:** medical, financial, legal advice.
- **Expected Model Behavior:**
    1. Address immediate safety concerns if present.
    2. Explain potential risks to the user in response to the prompt.
    3. Provide additional relevant information if possible.
- **Behaviors to Avoid:**
    - Promoting or enabling criminal activities.
    - Promoting or enabling dangerous behaviors.
    - Offensive or abusive language towards anyone.
    - Sexually explicit content.
- **Two Stages of Safety Implementation:**
    - **Safety in Pre-training:** Focused on eliminating fundamental biases and addressing privacy/copyright concerns.
    - **Safety in Fine-tuning:** Aimed at guiding the model towards desired behavioral patterns.
- **Safety in Pre-training:**
    - **Privacy and Copyright:** Removed personally identifiable information, excluded data sources with a high risk of personal information or copyright issues.
    - **Bias Mitigation:** Analyzed the occurrences of pronouns and identity terms related to gender, race, religion, nationality, and sexual orientation in the training data.
- **What Was NOT Done in Pre-training:**
    - Toxic sample filtering.
    - Artificial balancing of training data for demographic markers.
        - **Reasoning:** Filtering and balancing could lead to demographic erasure and hinder the model's ability to identify toxicity in pre-training.
- **Three-Pronged Approach to Safety in Fine-tuning:**
    - Supervised safety fine-tuning.
    - Safety RLHF (Reinforcement Learning from Human Feedback).
    - Safety Context Distillation.
- **Safety in Fine-tuning:**
    - **Safety SFT (Supervised Fine-Tuning):** Human annotators created prompts and responses demonstrating safe and helpful behavior. The model was fine-tuned using these examples.
    - **Safety RLHF:** Used RLHF to align the model with desired behaviors by training a reward model that scores responses based on human preferences.
        - **Steps:**
            1. Collected samples annotated for human preference.
            2. Trained a reward model to score model responses.
            3. Used the reward model to guide model parameter updates.
- **Safety RLHF - Preference Data:** Annotators selected the best response from multiple model-generated options based on safety guidelines, marking others as "rejected".
- **Safety RLHF - Reward Model Training:** Used a copy of the latest model checkpoint, replacing the decoder head with a sigmoid and employing binary ranking loss as the training objective.
	- Training objective: Binary ranking loss 
	- $\mathcal{L}_{\text{ranking}} = -\log(\sigma(r_{\theta}(x, y_{c})- r_{\theta}(x,y_{r})))$
	- $x$ is the prompt, $y_{c}$ is the response chosen by the annotators and $y_{r}$ is the response rejected by the annotators.
- **Safety RLHF - Guiding the Model:**
    - **Proximal Policy Optimization (PPO):** Updated the language model using the reward model as a proxy for human preference.
    - **Rejection-Sampling:** Generated multiple responses and used the reward model to select the best ones for further fine-tuning.
![[ppo.png | 500]]

- **Safety Context Distillation:** Fine-tuned the model with prefixed templates to encourage desired behavior even without the templates during actual use.
- **Evaluating Safety:**
    - Used benchmarks like ToxiGen, TruthfulQA, and BOLD to evaluate toxicity, hallucinations, and bias.
    - **Red-Teaming:** A proactive approach involving over 350 experts from various fields who tested the model's responses across a range of attack vectors, including:
        - Psychological manipulation
        - Logic manipulation
        - Syntactic manipulation
        - Semantic manipulation
        - Perspective manipulation
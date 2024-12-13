## Large Language Models

Representation learning
- Paradigm shift: learning intermediate meaning representations in the process of end-to-end training for a particular task.
	- Instead of training task specific model
	- train general purpose NN sentence encode
	- Apply across diverse NLP tasks
- Improve performance: rich semantic representations for downstream tasks
- Data efficiency: provide representation for tasks that lack training data
### ELMo: Embedding from Language Models
![[elmo.png | 350]]
- Pretrain a biLSTM model in the language modeling task
	- Encoder is 2 layer BiLSTM: model context in both directions, produce contextualised word representations
	- Language modeling objective
	- Jointly maximize log likelihood of forward and backward directions
- Use them as input to a task-specific model
	- ELMo word representations: weighted sum of hidden representations at all layers
	- Weights are learned given task
- Contributions
	- Contextualised word representations provide a level of disambiguation
	- Deep representations allow to capture linguistic information at various levels (syntax – lower layers; semantics – higher layers)
	- (Large) performance improvements in many NLP tasks
	- Paradigm shift towards sentence encoder pretraining

### BERT
![[bert.png | 300]]
- Transformer architecture
- Bidirectional context representation
- Two pretraining tasks: masked language modeling (MLM), next sentence prediction (NSP).
- Pretrain the encoder and fine-tune for a specific task.
- Stacked transformer blocks (MHA followed by FFNN)
- Introduce [CLS] and [SEP] special tokens.
![[bert_input.png | 400]]
- Standard conditional LM only model context in one direction: BERT performs bidirectional encoding by masking 15% of the input tokens. (cloze task)
- Randomly sampled sentence pairs with 50% pairs that one follow the other, predict whether the second sentence follow the first or not. (useful for tasks like QA)
- Pretraining loss: sum of mean MLM likelihood and mean NSP likelihood.
- Data: BooksCorpus  and English Wiki
- Contributions
	- SOTA in range of NLP tasks at the time
	- show importance of bidirectional preptraining
	- Reduced need for task specific architectures
	- most widely used model
	- Traditional linguistic hierarchy emerges within layers of BERT: lower layers – syntax; higher layers – semantics and discourse.

### Generative Language Models

- Left-to-right language model
- Generative model, able to generate text (unlike BERT)
- Transformer
- Interesting intuition: multitask learning from natural language instructions
- RLHF: Humans provide feedback on model outputs, and the model is then adjusted to better reflect human preferences

#### Reinforcement learning from human feedback (RLHF)
Key components:
1. Pretrained language model
2. gathering data and training a reward model
3. fine-tuning the LM with reinforcement learning

![[rlhf.png | 400]]
**Training a reward model**
- Collect a set of prompts
- Generate continuations for these prompts using multiple LMs
- Human rank the continuations, obtain a scalar score for each
- Score represents human preference
- Train a reward model to predict this score

**Fine-tune LM with RL**
Fine-tune the LM to better match human preferences
At each iteration:
- Given the prompt x, the LM generates continuation y
- Concatenate x and y, and pass as input to the reward model
- Reward model outputs a reward score r
- Fine-tune the LM to maximize the reward score for the current batch of data
- Regularizations to ensure the per token probability distributions don’t change too much (from original LM)

### Challenges

Can LLMs generalise systematically?
```
robin is a bird; robins are flowers
```
Factual errors and inference errors
Hallucinations: making up content
Memorisation vs. generalisation?
Data contamination: performance evaluations in many tasks are misleading
Many issues related to safety and bias
Interpretability
Better learning algorithms, e.g. continual learning
Low-resource languages
Few-shot learning and generalisation
Common sense reasoning
Ethics and alignment
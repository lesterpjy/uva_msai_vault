### **Sequence2sequence Models**

- Sequence2sequence models consist of two parts:
    1. **Encoder:** Takes a variable-length sequence of elements as the input and transforms it into a fixed-size context representation.
    2. **Decoder:** Maps the encoded state of fixed size to a variable-length sequence of elements.
- This model architecture was used to solve many problems, especially in **language processing**.

### **Sequence2sequence Example**

- The encoder reads a sentence in one language and converts it into a context vector.
- The decoder outputs a translation based on this context vector.

### **Sequence2sequence Example: Encoder**

- An encoder encodes the input sentence, a sequence of vectors $x = x_1, x_2,...,x_{T_x}$, into a context vector $c$.
- A common approach is to use an **RNN/LSTM**, such that: $h_t = f(x_t, h_{t-1})$ and $c=q({h_1,...,h_{T_x}})$.
    - $h_t \in R^h$ is a hidden state at time-step $t$.
    - $c$ is the context vector generated from the sequence of hidden states.
    - $f$ and $q$ are nonlinear functions.

### **Sequence2sequence Example: Decoder**

- The decoder is trained to predict the next word $y_t$, given the context vector $c$ and all the previously predicted words ${y_1,...,y_{t-1}}$.
- It defines a probability over the translation $y$ by decomposing the joint probability into the conditionals:
    $p(y) = \prod_{t=1}^{T_y}p(y_t | {y_1,...,y_{t-1}},c)$ where $y = (y_1,...,y_{T_y})$.
- With an RNN decoder, each conditional probability is modeled as: $p(y_t | {y_1,...,y_{t-1}},c)=g(y_{t-1},s_t,c)$, where
    - $g$ is a nonlinear (multi-layered) function.
    - $s_t$ is the hidden state of the RNN.

### **Limitation of seq2seq Models**
- The model needs to compress all necessary information from a source sequence into the fixed-length context vector $c$.
    - The context vector $c$ is seen as a bottleneck.
- When dealing with long sequences, it is challenging for the model to compress all information into a fixed-length context - due to **vanishing gradients**.
    - The context is a vector of floats and its size is the number of hidden units in the encoder RNN.

### **Overview of Attention**
- **Problem Addressed**: Traditional encoder-decoder models bottleneck the input by compressing it into a single fixed-length vector, which can limit performance, especially for long or complex sequences.
- **Solution**: Attention mitigates this issue by dynamically focusing on relevant parts of the input during decoding rather than relying on a fixed-length representation.
### **Core Ideas**
1. **Dynamic Input Representation**:
    - The input is encoded into a sequence of vectors, rather than a single fixed vector.
    - While decoding, the model adaptively selects a subset of these vectors as context, tailoring the focus to the current decoding step.
    - This approach avoids "squashing" all input information into one representation.
2. **Focus in Decoding**:
    - At each step of decoding, the model determines which parts of the input are most relevant to the current output, inspired by tasks like neural machine translation.

### **Formalization of Attention**

1. **Decoder Conditional Probability**:
    - The probability of generating a target word $y_t$ is conditioned on:
        - The previous target word $y_{t-1}$,
        - The decoder's previous state $s_{t-1}$,
        - A **context vector** $c_i$ specific to the current target word.
    - Formula:$$p(y_{t}∣y1,…,yt−1,x)=g(yt−1,st,ci)$$
2. **Context Vector $c_i$**:
    - Computed as a **weighted sum** of input annotations $h_j$: ci=∑j=1Txaijhjc_i = \sum_{j=1}^{T_x} a_{ij} h_jci​=j=1∑Tx​​aij​hj​
        - $h_j$: Annotations representing encoded input, focusing on regions near the $j$-th input word.
3. **Attention Weights $a_{ij}$**:
    - Define how much importance each input annotation $h_j$ has for the current context. aij=exp⁡(eij)∑k=1Txexp⁡(eik)a_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}aij​=∑k=1Tx​​exp(eik​)exp(eij​)​
        - $e_{ij}$: An **alignment score** that measures the relevance between the $j$-th input position and the $i$-th output position.
        - $e_{ij}$ is parameterized by a feedforward neural network, jointly trained with the entire model.
4. **Attention in Action**:
    - The alignment scores ($e_{ij}$) and weights ($a_{ij}$) ensure the decoder "pays attention" to relevant parts of the source input at each step.
    - This attention mechanism dynamically guides the decoder to focus on different input parts, enhancing its ability to generate contextually appropriate outputs.

### **Self-attention**
- **Self-attention** (or intra-attention) relates parts of a sequence with each other.
- The result is a representation of the whole sequence.
- In general terms, self-attention can be seen as an operation on sets of elements.

### **The Transformer**

- The transformer is an encoder-decoder model based on (self-)attention mechanisms.
    
    - It does not use any recurrence or convolutions.
    - The introduction of the transformer is referred to as NLP's ImageNet moment.
- The transformer completely changed the deep learning model landscape!
- It achieved **state-of-the-art performance in NLP tasks** and more recently, in other domains.
- The transformer is the **key concept behind LLMs**.

### **Important Transformer Concepts:**

- Queries, keys, values
- Scaled dot-product attention
- Multi-head (self-)attention

### **Transformers Step-by-Step**


### **Queries, Keys, and Values**

- The Transformer paper redefines the attention mechanism by providing a generic definition based on queries, keys, and values.
- **Intuition:** Use the query of the target and the key of the input to calculate a matching score. These matching scores act as the weights of the value vectors.

### **Attention: Dot Product with Scaling**

- The attention mechanism in transformers is called "**scaled dot-product attention**".
- The dot product computes similarity between queries and keys.
- For large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.
- Scaling by $\frac{1}{\sqrt{d_k}}$ counteracts this effect.
    - $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

### **Multi-headed Attention**

- It is beneficial to linearly project the queries $Q$, keys $K$, and values $V$ $h$ times with different, linear projections to $d_k$, $d_k$, and $d_v$ dimensions.
- The attention function is performed in parallel on each of these projected versions of $Q$, $K$, and $V$.
- These are concatenated and again projected, resulting in the final values. This is similar to the multiple filters in each convolutional layer.
    - $MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O$ where $W^O_i \in R^{hd_v \times d_{model}}$.

### **Multi-headed Self-attention**

For multi-head self-attention: the queries, keys, and values are equal to the input representation or from the previous (encoding/decoding) layer.

### **Transformer Encoder**
- The encoder consists of $N=6$ identical layers.
    
- Each encoder layer has 2 sub-layers:
    
    - Multi-head attention
    - Fully connected feed-forward network
- Each sub-layer has a residual connection around it, followed by layer normalization.
    

### **Transformer Decoder**

- The decoder also consists of $N=6$ identical layers.
- A decoder layer is identical to the encoder layer.
- It has an additional third sub-layer that performs multi-head attention over the output of the encoder.
- The masked self-attention sub-layer in the decoder prevents positions from attending to subsequent positions.
    - The predictions for position $i$ can depend only on the known outputs at positions $<i$.

### **Position Encoding**

- Attention is a permutation-invariant operation. A pure attention module will return the same output regardless of the order of its inputs.
- **Solution**: Positional encodings are added to the input to make use of the order of the sequence.

### **Positional Encoding in the Transformer**

- **Intuition:** Positional encodings follow a specific pattern that the model learns to determine the position of each word and the distance between words in the sequence.
- Positional encodings encode spatial, temporal, and modality identity, and they can be learned or fixed.
- The original Transformer uses sine and cosine functions of different frequencies:
    - $PE(pos,2i)=sin(pos/10000^{2i/d_{model}})$
    - $PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})$


### **Transformer Pros & Cons**

- **Pros:**
    - The transformer operates on data in parallel, which accelerates the learning process compared to RNNs that operate sequentially.
    - The transformer can deal with long-term dependencies in sequences.
- **Cons:**
    - The transformer scales quadratically with the number of inputs.
    - Transformers are memory-intense and require lots of data and long training.

### **Summary**

- Encoder-decoder is a useful architecture for many deep learning problems.
- The traditional encoder-decoder for NMT has a bottleneck issue with the context vector.
- The attention mechanism overcomes this problem by learning to select important features.
- The transformer is the first model that relies entirely on attention.

### **Recommended Papers that Started it All**

- Neural Machine Translation by Learning to Align and Translate, Bahdanu et al. ICLR (2015)
- Long Short-Term Memory-Networks for Machine Reading, Cheng et al. (2016)
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, Xu et al. (2016)
- Attention Is All You Need, Vaswani et al. (2017)

### **BERT**

- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers
- **Idea**: Pre-train bidirectional representations from unlabeled text, by jointly conditioning on both left and right context.
- A pre-trained BERT model can be fine-tuned with just one additional output layer to create SOTA models for NLP tasks.

### **BERT Input Representation**

- The input representation is a single sentence OR a pair of sentences in one sequence.
- The complete input is the sum of the token embeddings, the segmentation embeddings, and the position embeddings.
- The first token of every sequence is always a special classification token ([CLS]).
    - The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
- Sentence pairs are packed together into a single sequence, separated with a special token ([SEP]).
- Also, a learned embedding is added to every token indicating whether it belongs to sentence A or sentence B.

### **BERT Pre-training**

- **Task #1: Masked Language Modeling (MLM)**
    - Mask a percentage of the input tokens at random with a special token [MASK], and then predict those masked tokens.
- **Task #2: Next Sentence Prediction (NSP)**
    - Binary prediction whether the next sentences is a correct one.
    - When choosing the sentences A and B, 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence from the corpus.
- Datasets: BooksCorpus (800M words) and Wikipedia (2,500M words).

### **BERT Fine-tuning**

- Fine-tuning is straightforward since the self-attention mechanism allows BERT to model many downstream tasks.
- It is relatively inexpensive compared to pre-training.

### **A Family of BERT Models**

- RoBERTa: A Robustly Optimized BERT Pretraining Approach
    - More data, longer training, larger batches
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
- DeBERTa
- DistilBERT
- CamamBERT
- RoBERT ClinicalBERT

### **GPT**

- **G**enerative **P**retraining by **T**ransformers == GPT1
- A pre-trained unidirectional Transformer decoder
- **Idea**: To train a generative language model using "unlabeled" data and then fine-tune it on specific downstream tasks.

### **Unsupervised Pre-training**

- Given an unsupervised corpus of tokens, use a standard language modeling objective (to predict the next word in a sequence) to pretrain the model.

### **GPT 1 to 3**

- **GPT1:** Proves that language modeling serves as an effective pre-training objective which helps the model to generalize well.
- **GPT2:** Uses a larger dataset for training and adds additional parameters to build a stronger language model.
- **GPT3:** Even larger than GPT2; can automatically generate high-quality paragraphs.
    - Performs well on tasks on which it was never explicitly trained on, like writing SQL queries and code given a natural language description of the task.
- **Table of GPT Model Sizes**:

### **GPT Enables In-context Learning**

- GPT models can perform a new task without any parameter updates, simply by providing it with a few examples of the task as part of the input.

### **GPT versus BERT**

- The key difference between BERT and GPT is that GPT is a unidirectional Transformer **decoder**, whereas BERT is a bidirectional Transformer **encoder**.
    
- GPT outputs one token at a time, just like traditional language models.
    - After each token is produced, that token is added to the sequence of inputs.
    - That new sequence becomes the input to the model in its next step.
    - This is called **auto-regression**.
- In losing auto-regression, BERT gained the ability to incorporate the context on both sides of a word.

### **Reinforcement Learning from Human Feedback**

- ChatGPT is trained using reinforcement learning from human feedback (RLHF), where human AI trainers provide conversations in which they play both parts - the user and the AI assistant. [Source: OpenAI's ChatGPT blog post: [https://openai.com/blog/chatgpt/](https://openai.com/blog/chatgpt/)]

### **Why is ChatGPT so Convincing?**

- It is a result of exposure to extreme scale.
- It’s network consists of billions of parameters.
- It is a product of careful and elaborate hyperparameter tuning.
- Human feedback RL creates output formats that humans like.
- In short, ChatGPT has condensed the internet and gives it to us in a pleasing narrvative style.

### **Vision Transformers**

- Attention for text is intuitive, but it has also proven to be important for vision.
- Transformers assume the input is a sequence of tokens.
- **What is a token in the context of an image?**

### **Vision Transformer (ViT)**

Input image patches
![[vit_patches.png | 400]]

### **Vision Transformer (ViT)**

- Like BERT's [CLS] token, a learnable embedding is prepended to the sequence of embedded patches, and the classification is done on this token (with an MLP).
- "Position encodings" are added to the patch embeddings to retain positional information (attention by itself doesn't have any notion of ordering/space).
    - These vectors are also simply learned.

### **Attention versus Convolution**

- Convolution is local and shared.
- Attention is global.
- Hence, both bring different views to visual representation learning.
- Transformers impose less structure, which is useful in large-scale settings.

### **Attention as a Superset of Convolutions**

- Further reading: Deformable convolutions.

### **Training a ViT is More Difficult**

- The original paper required pre-training on ImageNet-22k (14M images) to achieve good performance.
- The DeiT (data-efficient image transformers) paper showed training with ImageNet-1k was possible if more augmentations and regularizations are used.

### **Swin Transformer: Return of Locality**

- **Idea**: Mostly looking at the "local" neighborhood can save some computation (remember attention is O(n^2)) or gain some accuracy by modeling this.
- Strong performance but slow models.

### **Convolutions and Attention are Complementary**

- Convolutions and attention can be combined to create more powerful models.

### **Summary**

- Sequential modeling
- Attention and self-attention
- Transformers
- Language and vision transformers





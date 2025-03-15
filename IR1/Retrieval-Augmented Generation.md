
## Introduction and Motivation

Retrieval-Augmented Generation (RAG) combines information retrieval systems with language generation models to enhance the capabilities of language models (LMs) by giving them access to external knowledge. This approach addresses several key limitations of traditional language models:

### Why RAG is Necessary

1. **Knowledge limitations**: Language models cannot memorize all knowledge, especially long-tail (less popular) information. Studies show that while LLMs learn popular information easily, they struggle with rare or specialized knowledge. RAG helps by retrieving this information when needed.
2. **Outdated knowledge**: LMs' knowledge becomes quickly outdated after training. For example, when asked "What's the affiliation of professor Eunsol Choi?", a model might incorrectly answer based on outdated knowledge, whereas a RAG system can retrieve current information.
3. **Verification challenges**: LLMs often generate plausible-sounding but incorrect information (hallucinations). RAG systems provide verifiable sources for generated content.
4. **Privacy and intellectual property concerns**: RAG allows models to access external knowledge without incorporating everything into model parameters.

## Definition and Core Concepts

RAG doesn't have a single, universally agreed-upon definition, but generally refers to systems that enhance language model capabilities by retrieving and incorporating external information during text generation.

### Two Primary Directions
1. **RAG for language modeling**: Enhancing next token prediction with retrieved information (e.g., KNN-LM)
2. **RAG for knowledge-intensive NLP tasks**: Using retrieval to solve tasks requiring domain knowledge (QA, fact verification, dialogue systems)
### Mathematical Formulation

![[rag_formulation.png]]

The general formulation of RAG can be expressed as:
$$y_i = LM(x, Z \leftarrow R(q; D))$$
Where:

- $x$ is the input
- $q$ is the query (often equal to $x$)
- $D$ is the knowledge corpus
- $R$ is the retriever function
- $Z = R(q; D)$ represents the retrieved chunks
- $LM$ is the language model
- $y$ is the output

### Core Challenges in RAG
1. **What to retrieve?*
    - ~~Tokens~~
    - Chunks/documents
    - ~~Entities~~
2. **How to incorporate retrieved information?**
    - Input level (e.g., prompts)
    - Intermediate level (e.g., hidden states)
    - ~~Output level (during decoding)~~
3. **When to retrieve?**
    - Once per input
    - Multiple times during generation
4. **How to train or adapt modules?**
    - Joint training
    - Independent training
    - Sequential training
5. **How to reduce computational costs?**
    - Memory optimization
    - Inference time reduction
    - Parameter efficiency

## Basic Architectures

### Retrieve-then-Read Approach

This was one of the earlier RAG architectures used primarily for Open-domain Question Answering (ODQA).

![[realm.png]]

**REALM (Retrieval-Augmented Language Model)** is a significant example of this approach:
1. **Retrieval phase**: $Z = BERT_{retriever}(q, D)$
    - Uses a bi-encoder to retrieve relevant passages from a knowledge corpus (see Dense Retrieval)
    - Implements passage chunk retrieval using an InfoNCE-like loss
    - Encodes query and retrieves most relevant documents
2. **Reading phase**: $y = BERT_{selector}(x; Z)$
    - A reader model (typically BERT-based) processes the retrieved texts
    - For QA tasks, performs span prediction to identify the answer
    - Maximizes likelihood of correct answer spans
    - ![[realm_span_pred.png]]
3. **Training approach**: REALM can be trained jointly end-to-end
    - Computes marginal likelihood: $P(y|x) = \sum_{z\in Z} P_θ(y|x,z)P_η(z|x)$
    - With assumption that non-top-k chunks have zero probability

REALM showed impressive results, outperforming much larger T5 models with significantly fewer parameters, demonstrating that effective retrieval provides relevant reading material that leads to more accurate answers.

![[realm_challenges.png | 400]]

### Retrieve-then-Generate Approach

This approach extends the retrieve-then-read strategy by replacing the reader with a generative model.
$\rightarrow$
$$
P_θ(y_{i}|x,z,y_{i}) \ \ \text{vs.}\ \ P_θ(y|x,z)
$$

**RAG (Lewis et al., 2020)** is a key example:

1. **Retrieval component**: Uses DPR (Dense Passage Retriever) backbone
    - Bi-encoder
    - Fine-tunes query encoder
    - keep document encoder frozen
2. **Generation component**: Employs BART for text generation
    - Fine-tunes the generator on task-specific data
3. **Marginalization strategies**:
    - **RAG-sequence**: Ensembles generations from k different chunks
        - Generates an entire sequence independently for each retrieved document.
        - retrieval happens at the sequence level, meaning the entire response is conditioned on one retrieved document at a time.
        - $$\begin{equation}\begin{aligned}p_{RAG-Sequence}(y|x) &\approx \sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x)p_\theta(y|x,z)\\&=\sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x) \prod_{i}^N p_\theta(y_{i}|x,z,y_{1:i-1})\end{aligned}\end{equation}$$
    - **RAG-token**: Implements retrieval-augmented token generation
        - Each token is computed as a weighted sum of probabilities over retrieved documents.
        - Every token in the generated sequence can consider different retrieved documents.
        - $$p_{RAG-Token}(y|x) \approx \prod_{i}^N \sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x)p_\theta(y_i|x,z,y_{1:i-1})$$
Experiments showed that better retrieval consistently leads to better QA performance, with RAG-Sequence generally outperforming RAG-Token.

![[rag_challenges.png | 400]]

**Fusion-in-Decoder (FiD)** further improved the retrieve-then-generate approach:
1. **Encoder**: Similar to RAG, encodes all retrieved chunks
2. **Decoder**: Unlike RAG, uses a specialized fusion mechanism
    - Concatenates all encoded chunks before decoding
    - Performs cross-attention over all chunks simultaneously
    - Effectively disambiguates between multiple candidates

This approach allows FiD to leverage many more passages (up to 100) than standard RAG systems. In practice, FiD achieved state-of-the-art results on multiple QA benchmarks with models smaller than 1B parameters, outperforming much larger models (11B T5, 100B+ GPT models).

![[fid_challenges.png | 400]]

## Advanced Architectures

As language models evolved into Large Language Models (LLMs), RAG architectures also advanced.

### Preliminary: Understanding LLMs

Modern LLMs (like GPT models) have several key characteristics:

- Trained on broad data with next-token prediction
- Scale to billions of parameters
- Learn multi-task capabilities from diverse data
- Exhibit few-shot and zero-shot generalization
- Instruction-tuned for better task understanding

### Adapting Retrieval to LLMs

**Atlas** (Izacard et al., 2022) addresses how to make retrieval systems work effectively with pre-trained LLMs:
1. **Key insight**: "If the language model finds a document useful when generating output, the retriever should rank that document higher."
2. **Training approach**: Rather than relying on document annotations, Atlas estimates document relevance based on LM usage patterns.
3. **Relevance estimation methods**:
    
    - **Relevance estimation**: similar to InfoNCE but not $q_{rel}$
        - $$p_{\eta}(z_{i}|q)=\frac{\exp(s(z_{i},q))}{\sum_{k=1}^K \exp(s(z_{k},q))}$$
    - **Attention Distillation (ADist)**: Uses cross-attention scores in FiD's decoder
        - $$\alpha_{z_i} = \frac{CrossAttn([q_1,q_2,...q_n], z_i)}{\sum_{k=1}^{K} CrossAttn([q_1,q_2,...q_n], Z)}$$
    - **Perplexity Distillation (PDist)**: likelihood of answer from FiD likelihood
        - $$PPL_{zi} = \frac{exp(log P_\theta(y|z_i))}{\sum_{k=1}^{K} exp(log P_\theta(y|z_k))}$$
4. **Results**: Atlas showed especially strong performance in few-shot settings, with query-side fine-tuning providing an efficient adaptation mechanism.

![[atlas_challenges.png | 400]]

### Adapting LLMs to Retrieval

**Self-RAG** (Asai et al., 2023) focuses on making LLMs retrieval-aware:
1. **Inference process**: Uses special tokens for RAG actions:
    - Retrieve: Determines when to retrieve documents
    - Judgment: Evaluates document relevance
    - Verification: Assesses factual correctness
2. **Training approach**: Fine-tunes LLMs to make retrieval decisions
    - Retriever remains frozen
    - LLM decides when to retrieve with a "[retrieve]=yes" token
    - Uses synthetic instruction-output pairs
3. **Strengths**: Allows dynamic, contextual retrieval decisions and self-critique
4. **Limitations**: Expensive training with GPT-4 synthetic data, limited scaling, reduced flexibility

![[self_rag_challenges.png | 400]]

## Higher-Level Challenges

Several ongoing research challenges affect RAG systems:
1. **Effectiveness issues**:
    - Can LMs effectively use all retrieved contexts?
    - "Lost-in-the-middle" problem: Information in the middle of retrieved documents is often ignored
    - "The power of noise" paradox: Some studies show random documents can sometimes improve performance
2. **Efficiency concerns**:
    - Knowledge conflicts: How to handle contradictory retrieved information
    - Memory augmentation: Efficiently using retrieval as context extension

## Conclusion and Future Directions

RAG systems offer several key advantages:
1. **Enhanced applications**: Better verifiability, handling of time-dependent questions, reduced hallucinations
2. **Information access**: Provides access to information beyond model parameters (e.g., REALM outperforming much larger T5 models)
3. **Parameter efficiency**: Achieves strong performance with smaller models compared to 100B+ parameter LLMs
4. **Adaptability**: Systems like Atlas and Self-RAG demonstrate flexible integration with different LLMs

Ongoing challenges include evaluation metrics, identifying retrieval bottlenecks, and optimizing the interaction between retrieval and generation components.

The field continues to evolve with new benchmarks (TREC RAG, TREC RAGTIME) and implementations (ragnaroks, bergen) supporting further research and development.
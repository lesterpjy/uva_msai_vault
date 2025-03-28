## Introduction and Motivation

Retrieval-Augmented Generation (RAG) combines information retrieval systems with language generation models to enhance the capabilities of language models (LMs) by giving them access to external knowledge. This approach addresses several <span style="color:rgb(172, 96, 230)">key limitations of traditional language models</span>:

### Why RAG is Necessary

1. <span style="color:rgb(172, 96, 230)">Knowledge limitations</span>: Language models cannot memorize all knowledge, especially <span style="color:rgb(172, 96, 230)">long-tail information</span>. Studies show that while LLMs learn popular information easily, they struggle with rare or specialized knowledge. RAG helps by retrieving this information when needed.
2. <span style="color:rgb(172, 96, 230)">Outdated knowledge</span>: LMs' knowledge becomes quickly outdated after training. For example, when asked "What's the affiliation of professor Eunsol Choi?", a model might incorrectly answer based on outdated knowledge, whereas a RAG system can retrieve current information.
3. <span style="color:rgb(172, 96, 230)">Verification challenges</span>: LLMs often generate plausible-sounding but incorrect information (<span style="color:rgb(172, 96, 230)">hallucinations</span>). RAG systems provide verifiable sources for generated content.
4. <span style="color:rgb(172, 96, 230)">Privacy and intellectual property concerns</span>: RAG allows models to access external knowledge without incorporating everything into model parameters.

## Definition and Core Concepts

RAG doesn't have a single, universally agreed-upon definition, but generally refers to systems that enhance language model capabilities by retrieving and incorporating external information during text generation.

### Two Primary Directions

1. <span style="color:rgb(172, 96, 230)">RAG for language modeling</span>: Enhancing next token prediction with retrieved information (e.g., KNN-LM)
2. <span style="color:rgb(172, 96, 230)">RAG for knowledge-intensive NLP tasks</span>: Using retrieval to solve tasks requiring domain knowledge (QA, fact verification, dialogue systems)
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

![[rag-core-challenges.png | 500]]

1. <span style="color:rgb(172, 96, 230)">What to retrieve?</span>
    - ~~Tokens~~
    - Chunks/documents
    - ~~Entities~~
2. <span style="color:rgb(172, 96, 230)">How to incorporate</span> retrieved information?
    - Input level (e.g., prompts)
    - Intermediate level (e.g., hidden states)
    - ~~Output level (during decoding)~~
3. <span style="color:rgb(172, 96, 230)">When to retrieve?</span>
    - Once per input
    - Multiple times during generation
4. <span style="color:rgb(172, 96, 230)">How to train or adapt modules?</span>
    - Joint training
    - Independent training
    - Sequential training
5. <span style="color:rgb(172, 96, 230)">How to reduce costs?</span>
    - Memory 
    - inference time
    - compute

## Basic Architectures

### Retrieve-then-Read Approach

<span style="color:rgb(172, 96, 230)">Retrieve-the-read</span>: one of the earlier RAG architectures used primarily for Open-domain Question Answering (ODQA).

![[realm.png]]

<span style="color:rgb(172, 96, 230)">REALM (Retrieval-Augmented Language Model)</span> is a significant example of this approach:

1. **Retrieval phase**: $Z = BERT_{retriever}(q, D)$
    - Uses a bi-encoder to retrieve relevant passages from a knowledge corpus (see Dense Retrieval)
    - Implements passage chunk retrieval using an InfoNCE-like loss
    - Encodes query and retrieves most relevant documents
    - <span style="color:rgb(172, 96, 230)">ANN + InfoNCE loss</span>
2. **Reading phase**: $y = BERT_{selector}(x; Z)$
    - A reader model (typically BERT-based) processes the retrieved texts
    - For QA tasks, performs <span style="color:rgb(172, 96, 230)">span prediction to identify the answer</span>
    - <span style="color:rgb(172, 96, 230)">Maximizes likelihood of correct answer spans</span>
    - ![[realm_span_pred.png]]
3. **Training approach**: REALM can be <span style="color:rgb(172, 96, 230)">trained jointly end-to-end</span>
    - Computes <span style="color:rgb(172, 96, 230)">marginal likelihood</span>: $P(y|x) = \sum_{z\in Z} P_θ(y|x,z)P_η(z|x)$
    - <span style="color:rgb(172, 96, 230)">Assumption: non-top-k chunks have zero probability</span>

REALM showed impressive results, outperforming much larger T5 models with significantly fewer parameters, demonstrating that effective retrieval provides relevant reading material that leads to more accurate answers.

![[realm_challenges.png | 400]]

### Retrieve-then-Generate Approach

This approach extends the retrieve-then-read strategy by <span style="color:rgb(172, 96, 230)">replacing the reader with a generative model</span>.
$\rightarrow$
$$
P_θ(y_{i}|x,z,y_{<i}) \ \ \text{vs.}\ \ P_θ(y|x,z)
$$

**RAG (Lewis et al., 2020)** is a key example:

1. **Retrieval component**: Uses DPR (Dense Passage Retriever) backbone
    - <span style="color:rgb(172, 96, 230)">Bi-encoder</span>
    - <span style="color:rgb(172, 96, 230)">Fine-tunes query encoder</span>
    - <span style="color:rgb(172, 96, 230)">Frozen document encoder</span>
2. **Generation component**: Employs <span style="color:rgb(172, 96, 230)">BART for text generation</span>
    - Fine-tunes the generator on task-specific data
3. **Marginalization strategies**:
    - **RAG-sequence**: <span style="color:rgb(172, 96, 230)">Ensembles k generations from k chunks</span>
        - Generates an entire sequence independently for each retrieved document.
        - retrieval happens at the sequence level, meaning the <span style="color:rgb(172, 96, 230)">entire response is conditioned on one retrieved document at a time</span>.
        - $$\begin{equation}\begin{aligned}p_{RAG-Sequence}(y|x) &\approx \sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x)p_\theta(y|x,z)\\&=\sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x) \prod_{i}^N p_\theta(y_{i}|x,z,y_{1:i-1})\end{aligned}\end{equation}$$
    - **RAG-token**: <span style="color:rgb(172, 96, 230)">Implements retrieval-augmented token generation</span>
        - <span style="color:rgb(172, 96, 230)">Each token is computed as a weighted sum of probabilities over retrieved documents</span>.
        - Every token in the generated sequence can consider different retrieved documents.
        - $$p_{RAG-Token}(y|x) \approx \prod_{i}^N \sum_{z \in top-k(p(\cdot|x))} p_\eta(z|x)p_\theta(y_i|x,z,y_{1:i-1})$$
Experiments showed that better retrieval consistently leads to better QA performance, with RAG-Sequence generally outperforming RAG-Token.

![[rag_challenges.png | 400]]

<span style="color:rgb(172, 96, 230)">Fusion-in-Decoder (FiD)</span> further improved the retrieve-then-generate approach:
- How to have better performance when considering more passages?
- <span style="color:rgb(172, 96, 230)">retrieval: BM25 or DPR</span>
- <span style="color:rgb(172, 96, 230)">generation: FiD-T5</span>

FiD-T5
1. **Encoder**: <span style="color:rgb(172, 96, 230)">encodes all chunks</span> (same as RAG)
2. **Decoder**: Fuse chucks with cross attention
    - <span style="color:rgb(172, 96, 230)">Concatenate all encoded chunks -> cross-attention over all chunks</span>
    - Effectively <span style="color:rgb(172, 96, 230)">disambiguates between candidates</span>

This approach allows FiD to leverage many more passages (up to 100) than standard RAG systems. In practice, FiD achieved state-of-the-art results on multiple QA benchmarks with models smaller than 1B parameters, outperforming much larger models (11B T5, 100B+ GPT models).

![[fid_challenges.png | 400]]

## Advanced Architectures

As language models evolved into Large Language Models (LLMs), RAG architectures also advanced.

### Preliminary: Understanding LLMs

Modern LLMs (like GPT models) have several key characteristics:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Trained on broad data with next-token prediction
- Scale to billions of parameters
- Learn multi-task capabilities from diverse data
- Exhibit few-shot and zero-shot generalization
- Instruction-tuned for better task understanding

### Adapting Retrieval to LLMs

<span style="color:rgb(172, 96, 230)">Atlas</span> (Izacard et al., 2022) addresses how to <span style="color:rgb(172, 96, 230)">make retrieval systems work effectively with pre-trained LLMs</span>:

1. **Key insight**: "<span style="color:rgb(172, 96, 230)">If the language model finds a document useful when generating output, the retriever should rank that document higher.</span>" $\rightarrow$ <span style="color:rgb(172, 96, 230)">train on query-output pairs without doc annotations</span>
2. **Training approach**: Rather than relying on document annotations, Atlas <span style="color:rgb(172, 96, 230)">estimates document relevance based on LM usage patterns</span>.
3. **Relevance estimation methods**:
    
    - **Relevance estimation**: similar to InfoNCE but no $q_{rel}$
        - $$p_{\eta}(z_{i}|q)=\frac{\exp(s(z_{i},q))}{\sum_{k=1}^K \exp(s(z_{k},q))}$$
    - **Attention Distillation (ADist)**: Uses cross-attention scores in FiD's decoder
        - $$\alpha_{z_i} = \frac{CrossAttn([q_1,q_2,...q_n], z_i)}{\sum_{k=1}^{K} CrossAttn([q_1,q_2,...q_n], Z)}$$
    - **Perplexity Distillation (PDist)**: likelihood of answer from FiD likelihood
        - $$PPL_{zi} = \frac{exp(log P_\theta(y|z_i))}{\sum_{k=1}^{K} exp(log P_\theta(y|z_k))}$$
4. **Results**: Atlas showed especially strong performance in few-shot settings, with <span style="color:rgb(172, 96, 230)">query-side fine-tuning providing an efficient adaptation</span> mechanism.

![[atlas_challenges.png | 400]]

### Adapting LLMs to Retrieval

<span style="color:rgb(172, 96, 230)">Self-RAG</span> (Asai et al., 2023) focuses on making LLMs retrieval-aware:

1. <span style="color:rgb(172, 96, 230)">Inference process</span>: Uses special tokens for RAG actions
    - <span style="color:rgb(172, 96, 230)">Retrieve</span>: Determines when to retrieve documents
    - <span style="color:rgb(172, 96, 230)">Judgment</span>: Evaluates document relevance
    - <span style="color:rgb(172, 96, 230)">Verification</span>: Assesses factual correctness
2. **Training approach**:<span style="color:rgb(172, 96, 230)"> Fine-tunes LLMs to make retrieval decisions</span>
    - Retriever remains frozen
    - LLM decides when to retrieve with a "[retrieve]=yes" token
    - Uses synthetic instruction-output pairs
3. **Strengths**: Allows <span style="color:rgb(172, 96, 230)">dynamic, contextual retrieval decisions and self-critique</span>
4. **Limitations**: <span style="color:rgb(172, 96, 230)">Expensive</span> training with GPT-4 synthetic data, <span style="color:rgb(172, 96, 230)">limited scaling</span>, <span style="color:rgb(172, 96, 230)">lower flexibility</span>

![[self_rag_challenges.png | 400]]

## Higher-Level Challenges

Several ongoing research challenges affect RAG systems:

1. <span style="color:rgb(172, 96, 230)">Effectiveness</span>:
    - Can LMs leverage retrieved contexts?
    - "<span style="color:rgb(172, 96, 230)">Lost-in-the-middle</span>" problem: Information in the middle of retrieved documents is often ignored
    - "<span style="color:rgb(172, 96, 230)">power of noise</span>" paradox: Some studies show random documents can sometimes improve performance
2. <span style="color:rgb(172, 96, 230)">Efficiency</span>:
    - <span style="color:rgb(172, 96, 230)">Knowledge conflicts</span>: How to handle contradictory retrieved information
    - <span style="color:rgb(172, 96, 230)">Memory augmentation</span>: Efficiently using retrieval as context extension

## Conclusion and Future Directions

RAG systems offer several key advantages:

1. <span style="color:rgb(172, 96, 230)">Enhanced applications</span>: Better verifiability, handling of time-dependent questions, reduced hallucinations
2. <span style="color:rgb(172, 96, 230)">Information access</span>: Provides access to information <span style="color:rgb(172, 96, 230)">beyond model parameters</span> (e.g., REALM outperforming much larger T5 models)
3. <span style="color:rgb(172, 96, 230)">Parameter efficiency</span>: Achieves strong performance with smaller models compared to 100B+ parameter LLMs
4. <span style="color:rgb(172, 96, 230)">Adaptability</span>: Systems like Atlas and Self-RAG demonstrate flexible integration with different LLMs

Ongoing <span style="color:rgb(172, 96, 230)">challenges include evaluation metrics, identifying retrieval bottlenecks</span>, and optimizing the interaction between retrieval and generation components.

The field continues to evolve with new benchmarks (TREC RAG, TREC RAGTIME) and implementations (ragnaroks, bergen) supporting further research and development.
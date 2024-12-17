
## Language Generation and Summarization


###  Sequence Modelling

- **Sequence modelling** is about predicting sequences of text, like POS tags, named entities, translations, or summaries.
- Neural sequence prediction models are used in many NLP tasks: part-of-speech tagging, named-entity recognition, machine translation, text summarization, entity retrieval, information extraction.
- Most solutions to these tasks use a **neural sequence prediction model** as their foundation.

### Sequence-to-sequence

- **Sequence-to-sequence** models try to learn the (often complex) relationship between an input sequence $x$ and an output sequence $y$.
- The relationship between the input and output sequences can be modelled as directional ($x \to y$) and non-deterministic.
- **Probabilistic modelling**: we treat $y$ as an observation of the random variable $Y$ and $x$ as an observation of the random variable $X$.
    - The probability of observing $y$ given $x$ is given by the function:

$$P(Y=y|X=x) = f(y|x; \theta)$$

where $\theta$ are the parameters.

- **Designing the probability mass function**: This involves specifying the parametric family and choosing parameter values.

### Parameterization

- **Conditional probability distributions (cpds) for structures**: Given an input $x$, we want to parameterize a distribution over possible outputs $y$.
- **Challenges**: Both the input and output spaces are typically very large.
- **Structured X, unstructured Y**: If the output space $Y$ is small, we can store the probabilities for each output for every input $x$. However, this is impractical for large input spaces $X$.
- **Log-linear (or logistic) cpds**: To handle large $X$, we can map $x$ to a fixed number of features $h(x)$, map those features to scores for each output, and then constrain the outputs to a probability simplex. This maps any $x$ to a Categorical pmf using a limited number of parameters.
![[log-linear_model.png | 350]]
- **Encoding functions**: These map an arbitrary input $x$ into a fixed-dimensional space. Before 2010, these were manually created, but now they are learned by neural networks.

### A Neural Text Classifier
![[k-way_text_classifier.png | 300]]
- A $K$-way text classifier assigns a probability to each of the $K$ classes for an input text $x$.
- **Encoder-decoder architecture**: An example of such a classifier is shown below, which uses an LSTM to encode the input sequence and a linear layer to predict the output class probabilities.
    - The parameters $\theta$ include the word embedding matrix, LSTM parameters, and parameters for the final linear transformation.

### Conditional Probability Distributions

- **Structured Y**: To handle large output spaces, we decompose the output sequence into smaller parts. For example, a sequence of POS tags is decomposed into individual tag predictions for each word. Similarly, a translation can be decomposed into individual target word predictions.
- **Autoregressive model**: In general, for an input sequence $x = \langle x_1,...,x_I \rangle$ and an output sequence $y = \langle y_1,...,y_J \rangle$, the probability of the output sequence given the input sequence is:
    
    $$P(Y=y|X=x) = \prod_{j=1}^J P(Y_j = y_j | X=x, Y_{<j} = y_{<j}).$$
    
    Each output $y_j$ is conditioned on the input $x$ and the previous outputs $y_{<j}$.
    

### One C-way classifier, J steps

- A general model for sequence prediction can be seen as repeatedly applying a shared text classifier.
    - At each step $j$, the classifier maps the input sequence $x$ and the previously generated outputs $y_{<j}$ to a probability distribution over the possible outputs for the current step.
- **POS tagging example**: In POS tagging, the same neural network $g(\cdot; \theta)$ is used at each step, mapping the input sentence and the previously predicted tags to a probability distribution over all possible tags.

### A Neural Tagger

- **Statistical model for neural POS tagging**:

$$ Y_j | X = x, Y_{<j}= y_{<j} \sim Cat(g(x, y_{<j};\theta)). $$
where $g(x, y_{<j};\theta))$ computes the probability distribution over the $K$ tags for the word at position $j$, given the input sequence $x$ and the previously predicted tags $y_{<j}$.

- **Encoder-decoder architecture for POS tagging**: This architecture uses a BiLSTM to encode the input sequence and an LSTM to decode the output tag sequence.

### Translation Example

- **Neural machine translation**: Similar to POS tagging, the same neural network $g(\cdot; \theta)$ is used repeatedly, mapping the input sentence and the previously generated target words to a probability distribution over all possible target words.
- In machine translation, the output length is not necessarily equal to the input length. The model continues generating words until a special end-of-sequence symbol is generated.

### A Neural Translation Model

- **Statistical model for neural machine translation**:

$$ Y_j | X=x, Y_{<j} = y_{<j} \sim Cat(g(x, y_{<j}; \theta)). $$

where $g(x, y_{<j}; \theta))$ computes the probability distribution over all target words for position $j$ in the translation, given the input sentence $x$ and the previously generated target words $y_{<j}$.

### Text Summarization

- **Text summarization** aims to generate a shorter version of a document (or collection of documents) containing the most important information.
    - **Extractive summarization**: This approach selects and extracts the most important sentences from the input document(s).
    - **Abstractive summarization**: This approach generates a summary by interpreting the content of the document(s) and producing original text.
- **Neural abstractive summarization**: Similar to machine translation, a conditional probability distribution is parameterized over possible tokens for each position in the summary, given the input document(s) and the previously generated tokens.

### An Improved Neural Abstractive Summarization Model

- Model architectures can be designed to encourage the model to produce specific types of summaries. For example, a model can be designed to prefer extractive summarization in certain situations.

### Parameter Estimation

- **Data**: A collection of input-output sequence pairs $(x, y)$, where both $x$ and $y$ are sequences of discrete symbols.
- **Statistical task**: Given an input sequence $x$, predict the conditional probability distribution over all possible output sequences.
- **NLP task**: Map an input sequence $x$ to an output sequence $y$, often by choosing the output sequence with the highest probability under the model.

### Formalization of Statistical Task

- **Statistical model**: The function $g$ maps an input sequence $x$ to a chain rule factorization of the conditional probability distribution $Y | X = x$.
- **Statistical objective**: Maximize the likelihood of the model given a dataset of observations $D$. This is typically done by minimizing the negative log-likelihood:

$$L(\theta | D) = \sum_{(x, y) \in D} \log P(Y = y | X = x) = \sum_{(x,y) \in D} \sum_{j=1}^{|y|} \log g_{y_j}(x, y_{<j};\theta).$$

where $\theta$ represents all trainable model parameters.

- **Algorithm**: Find the parameters $\theta$ that minimize the negative log-likelihood.

### Parameter Estimation (Continued)

- **Algorithm**: There is no closed-form solution to find the optimal parameters. Gradient-based optimization methods like stochastic gradient descent (SGD) can be used to find an approximate solution.
- **Handling massive datasets**: Computing the gradient over the entire dataset can be computationally expensive for very large datasets. Stochastic optimization can be used to estimate the gradient using smaller, random subsets of the data (mini-batches).

### A Soup of Names

- **Cross-entropy loss**: This is equivalent to the negative log-likelihood, and is a commonly used loss function for training sequence prediction models.
- **Categorical cross-entropy loss & Softmax loss**: These terms are often used interchangeably with cross-entropy loss. However, using the term "softmax loss" can be misleading, as the softmax function itself is not a loss function.

### Predictions

- **Making decisions**: Given an input $x$, our model outputs a probability distribution $P(Y|X=x)$ over all possible output sequences. We need a way to map this distribution to a single output sequence.
- **Most probable output**: A common approach is to choose the output sequence $c$ with the highest probability under the model:

$$ y^* = \operatorname{argmax}_{c \in Y} f(c|x;\theta).$$
However, finding the most probable output sequence is often computationally infeasible. Approximations like greedy decoding and beam search are commonly used.


### Most ‘useful’ in expectation

- **Expected utility**: In decision theory, a rational decision-maker chooses an action that maximizes their expected utility.
    - We can define a utility function $u(y,c;x)$ that quantifies the utility of predicting an output sequence $c$ when the true output sequence is $y$, given the input sequence $x$.
    - The expected utility can be approximated using Monte Carlo sampling.
- **Candidate enumeration techniques**: Various methods can be used to generate candidate output sequences for evaluation, including greedy decoding, beam search, ancestral sampling, top-p sampling, top-k sampling, and sampling without replacement.

### Evaluation

- **Statistical evaluation**: This assesses how well the model fits the training data. Perplexity and statistics of model-generated samples are commonly used.
- **Task-driven evaluation**: This measures how well the model performs on a specific NLP task.
    - Metrics like exact match, precision, recall, and F1 score are used for tasks with short output sequences, like question answering and entity linking.
    - String similarity metrics like BLEU, ROUGE, METEOR, and BEER are used for longer output sequences, like machine translation and summarization.
    - Semantic similarity metrics like COMET and BLEURT are also used for evaluating the meaning of generated text.

### Design Choices

- **Architectures**: Various neural network architectures can be used as encoders and decoders, including CNNs, GCNs, and Transformers.
    - **Transformers** are currently the most popular choice for many NLP tasks.
- **Transformers**: These models use self-attention and feed-forward networks to process input and output sequences.
    - In the encoder, each output position attends to all input positions.
    - In the decoder, each output position attends only to previous output positions to maintain autoregressiveness.

### Transformer

- **Transformers vs. RNNs**: Transformers are stateless, while RNNs are stateful. This impacts parallelization during training. Transformers can compute all outputs in parallel, while RNNs require sequential processing due to their recursive nature. At test time, both architectures require sequential decoding.

### Alternative Factorization

- **Alternative factorizations**: The chain rule factorization is not the only way to decompose the probability of an output sequence. Alternatives include:
    - Latent variable models (LVMs): These models use latent variables to capture dependencies between outputs that are not captured by the chain rule.
    - Non-autoregressive models: These models generate all outputs in parallel, without conditioning on previous outputs.
    - Conditional random fields (CRFs): These models define a probability distribution over output sequences based on a global scoring function.
    - Energy-based models (EBMs): These models define a probability distribution based on an energy function that assigns a score to each input-output pair.

### Moving away from chain rule?

- While alternative factorization methods can be beneficial, they often come with challenges:
    - Strong conditional independence assumptions might be unrealistic.
    - Computing marginal probabilities and normalizing constants can be intractable, making learning and prediction more difficult.
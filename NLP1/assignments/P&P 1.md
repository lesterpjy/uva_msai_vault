#### 1. Recognising language models
**Models** 
<u>NB</u>: Using the text and the sentiment annotation we trained a Naive Bayes classifier that views text as a bag of words. Note that even though the NBC is normally used to classify an existing piece of text, we can always sample text from it, after all it's a generative model as well. Before training this model, we preprocessed the dataset to remove stop words (i.e., words that are disproportionately too frequent).

<u>NGram LMs</u>:  Using only the text (that is, discarding sentiment, POS, and syntactic information), we trained two NGram language models, one using unigrams and another using trigrams.

<u>HMM</u>: Using the text and POS, we trained an hidden (first-order) Markov model (HMM).

<u>PCFG</u>: Using the text, the POS tag sequences and the syntactic trees, we trained a probabilistic context-free grammar (PCFG).

For all of our models, we used tabular Categorical CPDs with parameters chosen via maximum likelihood estimation (MLE).

**Training data**
The training data is made of movie reviews that have been annotated with positive/negative sentiment labels, with universal POS tags and binary-branching constituency syntax. There's a balanced number of positive/negative reviews, they're all written in English and have on average 22 words each.  

We generated sentences from each of those models, below you will find one sentence from each model. The sentences are presented in arbitrary order.  

1. moreover the cast and the producers begins in the past where it could have been so much brighter
2. intuition who direction cast great but what spaceship sharply theme buys 
3. the plot and the cast is okay but not too deep in the story and the acting and the ending .
4. a the too 's the be to awry ) they and 's dreamworks admit what . other friend have
5. a entire performances is not many write under cast where he might actors

For each sentence above, argue for which model likely generated it. Your arguments should highlight patterns you observe in the text in relation to specific technical aspects of the model you associate it with (that is, if you say model A because of pattern X, you need to explain in what way pattern X is expected for a model of type A).  If there are more than one such patterns, you should mention them, in order to obtain full points. No points will be awarded for answers without explanations.

Sentence 1 model: PCFG
- Complex syntactic structure such as "the cast and the producers" and "begins in the past" suggest that the sentence is generated with PCFGs as PCFGs can generate nested and embedded clauses through modeling hierarchical and recursive syntactic structures with context-free grammar rules augmented with probabilities.
- Long range dependency and hierarchical relationships: "where" with a relative clause "it could have been so much brighter"
- Probabilistic rule selection: incorrect selection of rule for collective noun, "the cast and the producers begins", but should be "begin". Such error suggests that PCFGs can select probable but incorrect rules due to learned probabilities.

Sentence 2 Model: Naive Bayes
- Collection of words without order or syntactic relationships: the independence assumption in NB lead to a bag-of-words treatment, and thus generated sentence are words related to movie reviews but lacks grammatical structure.
- Lack of stop words: stop word like "the", "is", "and", "of" that have high occurance in English are often removed from the data before training the NB model. This helps the model focus on more discriminative words, but during generation these common words are not sampled from the model.

Sentence 3 Model: Trigram LM
- Coherent semantics within trigram sequences: sequences like "the plot and", "plot and the", "and the cast", etc. The trigram model sample each word based on the previous two words, capturing trigram sequences that are locally coherent.
- Repetition of structures like "the plot and", "the story and", "the acting and" shows that these are common trigram structures contained in the training data.

Sentence 4 Model: Unigram LM
- Lack of grammatical structure: unigram LM ignores word order and syntax, so the generated sentence is just a sequence of words sampled independently from the model.
- High frequency of common stopwords: the higher frequency of words like "a", "the", "and", "be" compared to other sentences suggests that the model is a unigram LM as it would be generating words solely based on their probabilities, which is higher in a unigram LM for these common stopwords.
- Random inclusion of punctuations: punctuations are treated as tokens during the unigram LM training, and thus sampled based on their probabilities to be placed arbitrarily in the sentence.

Sentence 5 Model: Hiden Markov Model
- Syntactically acceptable structures based on POS tags: since HMM generate POS tags based on transition probabilities and then emits words for each POS tag based on emission probabilities, the generated sentences can be syntactically structured but semantically incoherent. Transitions like VB -> IN -> NN are syntactically valid transitions that can correspond to phrases like "draw on paper", but the emitted phrase in the generated sentence is "write under cast", which makes little semantic sense. This suggests that the sentence is generated with HMM.

#### 2. Log-linear LM
Disappointed with challenges in estimating NGram LMs via MLE, in particular with particularly long N-grams, an NLP1 student decided to try a different way of parameterising a Markov model. 

The student figured that the key to an NGram LM is to design a mechanism capable of mapping a history containing N-1 words to a probability vector over the known words, hence prescribing a Categorical distribution for the next word W given a history H=h. 

In a tabular representation, this mechanism is essentially a table lookup: we store the probabilities (or the counts used to compute probabilities) in a table, with rows for each known history and columns for each known word. 

The student proposes to use a log-linear model instead:
- Represent the history h as a vector ϕ(h) of size V (for vocabulary size), where each coordinate counts how many times the corresponding word occurs inside of the history.
- Initialise parameter matrix W of size V×V and a parameter vector b of size V.
- Whenever we need to prescribe the distribution of the next word W given some history H=h, we use: softmax(W ϕ(h)+b) as the probabilities for the V possibilities for W.
- Use a training corpus to obtain maximum likelihood estimates for W, b via gradient-based optimisation. 
The student trained two variants of this model:
1. The first setting is a bigram LM, that is, any one h has a single word in it. 
2. The second setting is a 10-gram LM, that is, any one h has 9 words in it.


##### 2a. The student was disappointed to see that model (1) performed just like a classic tabular bigram LM.  Based on the design choices of this log-linear LM, should the student be disappointed? Justify your answer in either case.

With only one word in the history, the vector $\phi(h)$ corresponds to an one-hot-encoding of the vocabulary with a value of 1 at the index corresponding to the word in $h$, and zero else where. Based on this design choice, the probability calculation $\text{softmax}(W \phi(h)+b)$ in the log-linear model equates to selecting the column from $W$ corresponding to word $h$, adding $b$ and taking the softmax to obtain the probability of each word in the vocabulary. Learning the parameters $W$ and $b$ via MLE via a gradient based optimization essentially equates to learning the probability specific to each word pair $(w,h)$ or in other words, for each bigram. The probabilities estimated via this approach equates to estimating the probabilities directly from count with $P(W=w|H=h)=\frac{\text{count}(h,w)}{\text{count}(h)}$, which is also estimated via MLE. Therefore, the student should not be disappointed with model (1), because it should estimate similar probabilities for each word pair, and thus performs just like a tabular bigram LM.

##### 2b. The student was amazed by their ability to work with 10-grams while storing a relatively small model. Explain why model (2) above is much more compact than a classic tabular 10-gram LM.

For a classic tabular 10-gram LM, all counts corresponding to each potential pairs of word and history of 9 words must be stored to estimate the probabilities. This leads to a very sparse storage of the counts, as not every word would have occurred with every possible combination of histories of 9 words. For a vocabulary of size V, the number of potential histories of 9 words, in addition to the word itself, would give V^10 number of possible combinations. All of these possible combinations must be enumerated and their counts stored. Since it is likely that many of these combinations do not occur, this leads to a sparsity of the stored counts. For a log-linear model, the history $h$ is represented as a vector $\phi(h)$ of size $V$, where each index correspond to the counts of occurrences of each word in the history, therefore many different 9 word sequences can map to the same vector. With a fixed parameter size of $W$ and $b$, the total number of parameter stored depend only on the vocabulary size $V$ instead of the number of unique 10-word sequence like in the 10-gram LM.

##### 2c. The student had seen samples from NGram LMs in a textbook, and samples tended to get better whenever more history was employed. The student then obtained samples from a trained model (2), and, for the student's surprise, the samples looked terrible. Analyse the design choices the student made and explain to them why their observation is not at all surprising.

As word order information in the history is lost in the construction of vector $\phi(h)$, since only the counts of word occurrences are being considered, it is not surprising that the samples generated are terrible. The trained log-linear model (2) although considers a 9-word history, due to the loss of word order information in $\phi(h)$, the model learns word co-occurrences without sequential dependency. This leads to a dilution of the context information in the history: since different sequences would map to the same $\phi(h)$ (as mentioned in the previous question), the model averages over many possible context and lead to poor performance for sampling.

#### 3. POS and HMM Tagger
##### 3a. Many jokes depend on ambiguity: for the following example, specify the type of ambiguity involved and explain it briefly (in 1 or 2 sentences). A: Your dog's chasing a man on a bicycle. B: Don't be silly, my dog can't ride a bicycle.

The ambiguity is a syntactical ambiguity, where the sentence could be parsed into two different syntactic trees. In these interpretations, it is unclear whether the dog is chasing the man who is riding the bicycle, or that the dog is riding a bicycle to chase the man.

##### 3b. One common source of errors of HMM POS taggers is that nouns occurring immediately before other nouns (e.g., catamaran trailer) are often tagged as adjectives. Conversely, the same HMM POS tagger might often incorrectly tag prenominal adjectives as nouns (e.g., trial offer). Give one valid reason for this effect.

HMM POS tagger relies on transition probabilities between part of speech and emission probabilities of words given a tag. Since the transition probability from adjective to noun are typically high in English (as adjectives often precedes noun), words that are less commonly observed as prenominal adjectives, like "catamaran", can be mis-tagged as adjectives due to this high adjective-noun transition probability. For words like "trail" that can be ambiguously considered as either a noun or an adjective, mistagging is more likely. In this case as maybe the word "trial" is more commonly used as nouns in the training data, it can have higher emission probability from a noun tag, leading to the mistagging. Additionally, the emission probability of HMM POS taggers is not context dependent and only dependent of the POS tag the word is emitted from, as such, the context of noun-noun phrases is lost in the model.

#### 4. Pizza with what?

Here's a short PCFG for this exercise (in brackets we have the probability of the right-hand side given the left-hand side of the rule), the PCFG is in Chomsky normal form (CNF) for your convenience:  
  
- (1.0) S → NP VP
- (0.3) NP → NP PP
- (0.3) NP → I
- (0.18) NP → pizza  
- (0.22) NP → fork
- (0.6) N → pizza
- (0.4) N → fork
- (1.0) PP → P N
- (1.0) P → with
- (0.18) VP → V NP
- (0.25) VP → V PP
- (0.22) VP → VP NP
- (0.35) VP → VP PP
- (1.0) V → eat

##### 4a. Use the inside algorithm to evaluate the marginal probability of the sentence "_I eat pizza with fork_" and report its numerical value (along with all steps of your answer, and do invest in making them clear). As this is a toy example, it is possible, in principle, to solve this via exhaustive enumeration in our own lifetime. While you can do that in order to verify the correctness of your answer, for this exercise we expect you to use the inside algorithm.



##### 4b. Use dynamic programming to find the most probable parse for the sentence "_I eat pizza with fork_". Report all steps of your answer, and do invest in making them clear.  As this is a toy example, it is possible, in principle, to solve this via exhaustive enumeration in our own lifetime. While you can do that in order to verify the correctness of your answer, for this exercise we expect you to use dynamic programming. 


##### 4c. Starting from the nonterminal symbol S, draw one complete sample from the PCFG above. Explain the steps in your procedure. Guide your sampler with these randomly generated numbers: [0.37, 0.95, 0.73, 0.6, 0.16, 0.16, 0.06, 0.87, 0.6, 0.71, 0.02, 0.97, 0.83, 0.21, 0.18, 0.18, 0.3, 0.52, 0.43, 0.29, 0.61, 0.14, 0.29, 0.37, 0.46, 0.79, 0.2, 0.51, 0.59, 0.05, 0.61, 0.17, 0.07, 0.95, 0.97, 0.81, 0.3, 0.1, 0.68, 0.44, 0.12, 0.5, 0.03, 0.91, 0.26, 0.66, 0.31, 0.52, 0.55, 0.18].  Treat this as a stack, pop numbers one at a time, from left to right. FYI we obtained them in Python via:  np.round(np.random.RandomState(42).uniform(size=50), decimals=2).






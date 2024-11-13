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

#### 2. Log-linear LM
Disappointed with challenges in estimating NGram LMs via MLE, in particular with particularly long N-grams, an NLP1 student decided to try a different way of parameterising a Markov model. 

The student figured that the key to an NGram LM is to design a mechanism capable of mapping a history containing N-1 words to a probability vector over the known words, hence prescribing a Categorical distribution for the next word W given a history H=h. 

In a tabular representation, this mechanism is essentially a table lookup: we store the probabilities (or the counts used to compute probabilities) in a table, with rows for each known history and columns for each known word. 

The student proposes to use a log-linear model instead:
- Represent the history h as a vector ϕ(h) of size V (for vocabulary size), where each coordinate counts how many times the corresponding word occurs inside of the history.
- Initialise parameter matrix W of size V×V and a parameter vector bb of size V.
- Whenever we need to prescribe the distribution of the next word W given some history H=h, we use: softmax(W ϕ(h)+b) as the probabilities for the V possibilities for W.
- Use a training corpus to obtain maximum likelihood estimates for W, b via gradient-based optimisation. 
The student trained two variants of this model:
1. The first setting is a bigram LM, that is, any one h has a single word in it. 
2. The second setting is a 10-gram LM, that is, any one h has 9 words in it.


##### 2a. The student was disappointed to see that model (1) performed just like a classic tabular bigram LM.  Based on the design choices of this log-linear LM, should the student be disappointed? Justify your answer in either case.

With only one word in the history, the vector $\phi(h)$ corresponds to an one-hot-encoding of the vocabulary with a value of 1 at the index corresponding to the word in $h$, and zero else where. 

##### 2b. The student was amazed by their ability to work with 10-grams while storing a relatively small model. Explain why model (2) above is much more compact than a classic tabular 10-gram LM.


##### 2c. The student had seen samples from NGram LMs in a textbook, and samples tended to get better whenever more history was employed. The student then obtained samples from a trained model (2), and, for the student's surprise, the samples looked terrible. Analyse the design choices the student made and explain to them why their observation is not at all surprising.


#### 3. POS and HMM Tagger


#### 4. Pizza with what?

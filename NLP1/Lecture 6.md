## Compositional semantics

- Principle of Compositionality: meaning of each whole phrase derivable from meaning of its parts.
- Sentence structure conveys some meaning.
- Deep grammars: model semantics alongside syntax, one semantic composition rule per syntax rule.

Issues with semantic composition
- similar syntactic structure with different meanings: pleonastic pronouns (it snows)
- different syntactic structure similar meaning
- not all phrases interpreted compositionally: idioms
- additional meaning arise through composition: logical metonymy
- Meaning transfers and additional connotations can arise through composition: metaphor

**Compositional Distributional Semantics**
- Extending distributional semantics to phrases/sentences
- Unsupervised model $\Rightarrow$ general-purpose representations
- Model composition in vector space. However, if we would model every sentence as independent, we would get an infinite dimensional space
- Vector mixture model
- Lexical function model

## Compositional Semantics with Neural Networks

1. How do we learn a (task-specific) representation of a sentence with a neural network?
2. How do we make a prediction for a given task from that representation?

- Supervised learning framework $\Rightarrow$ compositional representations are fine-tuned for specific application/task
- Word representations are taken as input and processed within the network
- Example tasks include sentiment classification, paraphrasing, machine translation, ...
- Using recurrent and/or recursive networks (LSTMs, Tree-LSTMs, ...)

### LSTM

![[lstm.png | 450]]

Child-Sum Tree-LSTM
○ sum over the hidden representations of all children of a node (no children order)
○ can be used for a variable number of children
○ shares parameters between children
○ suitable for dependency trees
N-ary Tree-LSTM
○ discriminates between children node positions (weighted sum)
○ fixed maximum branching factor: can be used with N children at most
○ different parameters for each child
○ suitable for constituency trees
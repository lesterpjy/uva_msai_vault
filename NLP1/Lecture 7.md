## Discourse Processing

### Discourse structure

Discourse coherence assumptions can affect interpretation:
```
John likes Bill. He gave him a nice Christmas present.
```
If EXPLANATION - ‘he’ is probably Bill.
If JUSTIFICATION (supplying evidence for another sentence), ‘he’ is John

Factors influencing discourse interpretation
1. Cue phrases (e.g. because, and)
2. Punctuation (also prosody) and text structure.
```
Max fell (John pushed him) and Kim laughed.
Max fell, John pushed him and Kim laughed.
```
3. Real world content:
```
Max fell. John pushed him as he lay on the ground.
```
4. Tense and aspect.
```
Max fell. John had pushed him.
Max was falling. John pushed him
```
- Analyzing a text for rhetorical relations mostly gives a binary structure: the main sentence is called **nucleus**, and subsidiary phrase (explanation, justification, ...) is called **satellite**
- In a narration (cue phrase and) both sentences have equal weight instead of nucleus vs satellite.
### Learning document representations

Sentence representations
1. use hL — the final hidden state of the LSTM
2. use an average of LSTM hidden states at all time steps (mean-pooling)
3. use max-pooling — take the maximum value in each vector component of all hidden states
4. use an attention mechanism, i.e. a weighted sum of the hidden states at all hidden states. 

Document Representations
1. Feed the whole document to an LSTM word by word. Possibly use word-level attention to learn what are the useful words.
2. Build a hierarchical model
	1. first compute sentence representations
	2. combine sentence representations into a document representation
	3. using another LSTM and / or attention over sentences
	4. train with a document level objective

### Referring expressions and coreference
To fully process a discourse, co-references/referring expressions like pronouns need to be resolved

We can define the following entities for a referring expression:
- referent - a real world entity to which is referred
- referring expression - part of speech that refers to an entity
- antecedent - the text initially evoking a referent (where referent is named)
- anaphora - the phenomenon of referring to an antecedent
- cataphora - pronouns that appear before the referent (rare)


### Algorithm for coreference resolution

- For anaphora resolution, we mostly apply a supervised training algorithm
- The instances in the corpus are possible pairs of pronoun and antecedent (possible antecedent include all noun phrases in the current and last 5 sentences)
- The classification is binary (true if pronoun refers to this specific antecedent, otherwise false)
- Training data is annotated by humans
- Beware that there are also pronouns in the text that might have no referent at all (pleonastic pronouns)
- Distinguishing between hard and soft constraints that must be fulfilled between pronouns and antecedent.
- Hard constraints
	- pronoun agreement (singular, plural, etc.)
	- Reflexive pronouns must be coreferential with a preceding argument of the same verb, non-reflexive pronouns cannot be
	- Pleonastic pronouns are semantically empty, and don’t refer (it is snowing)
- Soft constraints/Salience
	- recency - more recent antecedents are preferred
	- grammatical role - subjects might be referred to more often than objects. Also, it is preferred that entity and pronoun has same role in sentence (subject, object, ...)
	- repeated mention - entities that have been mentioned more often are preferred
	- parallelism: entities sharing same role as the pronoun is same sort of sentences are preferred
	- coherence effect: pronoun resolution might depend on discourse relation/ semantic within the sentences

- Based on the hard and soft constraints, we can define features for every pronoun-antecedent pair

![[coreference_resolution_features.png | 300]]

- Simple classification model takes these features as input and classifies the link as valid or not
- Simplest evaluation matrix is link accuracy (number of correct links). However, it does not take into account pleonastic pronouns, repeated mention, or use information from previous links, so multiple metrics exist.
## Brief History of IR

Information Retrieval (IR) has evolved from traditional library systems to modern search engines. The historical progression shows how information organization has changed to accommodate growing volumes of knowledge:

### Libraries as Knowledge Repositories

Libraries have served as civilization's knowledge repositories for millennia, with notable examples showing the growth of information volume:

- Library of Alexandria (280 BC): 700,000 scrolls
- Vatican Library (1500): 3,600 codices
- Herzog-August-Bibliothek (1661): 116,000 books
- British Museum (1845): 240,000 books
- Library of Congress (1990): 100,000,000 documents

Early librarians organized information using subject catalogs with various sorting methods:

- By author
- By title
- By subject

This organizational challenge intensified as collections grew, leading to debates about the best cataloging systems.

### The Memex Concept

A pivotal moment in IR history came with Vannevar Bush's 1945 article "As We May Think" in The Atlantic Monthly. Bush proposed the "memex," a theoretical device for individuals to store all their books, records, and communications. Bush described it as "an enlarged intimate supplement to memory" that could be consulted "with exceeding speed and flexibility." This concept anticipated many features of modern information systems.

### Evolution of IR Models

Several key models emerged as computing technology advanced:

#### Vector Space Model (G. Salton, 1989)

- Represents queries and documents as high-dimensional vectors in word space
- Associates weights (e.g., frequency) with each word
- Models relevance as cosine similarity between vectors

#### Probabilistic Relevance Model (Robertson & Spärck Jones, 1972)

- Views documents and queries as probability distributions over a word space
- Models relevance as similarity between distributions
- Karen Spärck Jones made foundational contributions that established the basis for search engines

#### Non-textual Signals (PageRank, 1999)

- Exploit link structure of the web to determine page authority
- Leverage usage data to determine what other users find relevant
- Page, Brin, Motwani, and Winograd's work on "bringing order to the web"

#### Semantic Search

- Focuses on entities and their relationships
- Recognizes that we're surrounded by entities connected by relations
- Enables more conceptual rather than just keyword-based searching

#### Digital Assistants

- Implements dialogue systems for natural interaction
- Provides a more natural way of accessing and storing knowledge
- Enables personalization of search results

### The Future of IR (as predicted in 2019)

- Making systems more intelligent through comprehension and combination of information
- Improving machine reading and text understanding
- Integrating statistics with semantics
- Understanding and anticipating user intention
- Using queries, context, and user preferences for better results

## How Search Works

### Big Picture: Offline vs Online Processing

Search systems operate in two main phases:

<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
**Offline Processing:**
- Crawling: Discovering and fetching documents
- Quality/Freshness assessment: Determining document value and recency
- Pre-processing and Indexing: Preparing documents for efficient retrieval
**Online Processing:**
- Query understanding: Interpreting what the user wants
- Context analysis: Incorporating user context and preferences
- Logging: Recording user interactions for system improvement
- Ranking: Ordering results by relevance

### Text Representation and Processing

#### Text Representation Approaches

Text can be represented in various ways:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- **Syntactic structure**: How words relate grammatically
- **Semantic structure**: The meaning relationships between terms
- **Bag of words model**: Historically, text was modeled without considering word order (e.g., "dog bites man" = "man bites dog")
- **Modern approaches**: Preserve order information in some way

Text can be broken down into different units:
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
- Stems (root forms of words)
- Terms (meaningful units)
- Phrases (word combinations)
- Entities (named objects, people, places)
- N-grams (sequences of n consecutive words)
- Visual words (for image retrieval)

#### Text Processing Pipeline
<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. **Tokenization**: Dividing text into separate words/tokens
    - Surprisingly complex in English and even more challenging in other languages
    - <span style="color:rgb(172, 96, 230)">Challenges</span> include handling:
        - Multi-word combinations (e.g., "ben e king", "world war II")
        - Hyphenations (e.g., "e-bay", "winston-salem")
        - Special characters in tags, URLs, and code
        - Capitalization (e.g., "Bush" vs "bush", "Apple" vs "apple")
        - Apostrophes (e.g., "o'donnell", "can't", "80's")
        - Numbers (e.g., "quicktime 6.5 pro")
        - Periods (e.g., "Ph.D.", "cs.umass.edu")
2. **Case normalization**: Converting terms to lowercase
    
3. **Stopword removal**:
    - Frequency-based: Remove words with frequency higher than a threshold
    - Dictionary-based: Remove words from a predefined list
    - Example: In "to be or not to be," all words would typically be stopwords
4. **Stemming**: Reducing words to their root forms
    - Morphological variations include:
        - Inflectional (plurals, tenses)
        - Derivational (verbs to nouns, etc.)
    - Example: "fishing", "fished", "fisher" → "fish"
    - Types:
        - Dictionary-based: Uses lists of related words
        - Algorithmic: Uses rules to determine related words
    - Porter Stemmer:
        - Classic algorithm from the 1970s
        - Applies a series of rules to remove suffixes
        - Produces stems, not necessarily valid words
        - Has known errors (false positives like "organization"→"organ" and false negatives like "european"→"europe")
5. **Phrase detection**:
    - Noun phrase detection using part-of-speech tagging (sequence of nouns/ adjectives followed by nouns)
    - N-gram identification (bigrams, trigrams)
    - Query-time phrase detection using index with word positions

### Boolean Search and Indexing

#### Boolean Search Model

![[bool_search_model.png | 450]]

- Documents represented as sets of words
- <span style="color:rgb(172, 96, 230)">Queries expressed as Boolean expressions (AND, OR, NOT with brackets)</span>
- Example: "[[Rio & Brazil] | [Hilo & Hawaii]] & hotel & !Hilton]"
- Output is an unranked set of matching documents

**<span style="color:rgb(172, 96, 230)">Pros</span>:**
- Easy to understand for simple queries
- Clean formalism with rigid logic (AND means all, OR means any)

**<span style="color:rgb(172, 96, 230)">Cons</span>:**
- Difficult to express complex user requests
- Hard to control the number of retrieved documents
- No inherent way to rank results (all matches satisfy the query equally)
- Cannot build matrix on bigger collections.

#### Inverted Index

The <span style="color:rgb(172, 96, 230)">inverted index</span> is a crucial data structure for efficient retrieval:

- For each term, stores a list of documents containing that term
- Each document is identified by a numeric ID
- Structure:

![[inverted_index.png | 400]]

**Processing AND queries:**

- Locate each term in the dictionary
- Retrieve their postings (document lists)
- "Merge" the postings by taking their <span style="color:rgb(172, 96, 230)">intersection</span> (Merge takes $O(x+y)$ for posting sizes of $x$ and $y$)
- For positional queries (phrases like "to be or not to be"):
	- $<term:docs>$ no longer sufficient.
	- Use <span style="color:rgb(172, 96, 230)">positional indexes</span>:
		- ![[positional_indexes.png | 300]]
		- ![[positional_index_example.png | 400]]
	- Use merge algorithm recursively at the document level

**Constructing the inverted index:**

![[inverted_index_construction.png | 400]]

<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. In-memory approach:
    - Simple but limited by available memory
    - Not easily parallelizable
2. Merge-based approach:
    - Build partial indexes until memory runs out
    - Write partial indexes to disk
    - Merge them into a complete index
    - Maintain alphabetical order for efficient lookup
3. Distributed indexing:
    - Uses frameworks like MapReduce
    - Mappers process individual documents
    - Reducers combine and sort term occurrences

## Key Challenges in IR

Throughout the slides, several core challenges are highlighted:

<span style="color:rgb(172, 96, 230)">CHEATSHEET</span>
1. **Scalability and efficiency (C1)**: Handling massive document collections and query volumes
    - Text processing reduces dimensionality
    - Inverted indexes enable efficient retrieval
    - Distributed processing handles scale
2. **Lexical mismatch (C3)**: Bridging the gap between query and document vocabulary
    - Stemming helps connect word variations
    - Stopword removal focuses on meaningful terms
    - Phrase detection captures multi-word concepts

## Takeaways

### Text Processing

- Critical for both scalability and addressing lexical mismatch
- Involves complex tradeoffs between effectiveness and efficiency
- Must be carefully tuned for the specific retrieval task
### Inverted Index

- Fundamental data structure enabling efficient retrieval
- Stores all information needed for document retrieval
- Construction is computationally expensive but essential for search performance

Modern search systems build upon these foundations, incorporating additional signals (like link structure and user behavior) to improve relevance and meet evolving user expectations.
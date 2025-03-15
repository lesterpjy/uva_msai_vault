## 1. Historical Development of Information Retrieval

### Evolution of Information Storage

- Libraries have served as knowledge repositories throughout human civilization
- Notable examples show exponential growth in document collections:
    - Library of Alexandria (280 BC): 700,000 scrolls
    - Vatican Library (1500): 3,600 codices
    - Library of Congress (1990): 100,000,000 documents
- This growth created the need for effective organization systems

### Early Library Organization

- Traditional libraries used subject catalogues with cards sorted by:
    - Author
    - Title
    - Subject
- Librarians debated the optimal subject cataloguing methodology
- As information volume increased, managing these systems became increasingly difficult

### Early Concepts of Modern Information Retrieval

- Vannevar Bush's "Memex" (1945) - conceptual precursor to modern information systems
    - Described as "a device in which an individual stores all his books, records, and communications"
    - Envisioned as a "mechanized private file and library" with "exceeding speed and flexibility"
    - Represented one of the first concepts of personal information management systems

## 2. Core IR Models and Their Evolution

### Vector Space Model (G. Salton, 1989)

- Represents documents and queries as high-dimensional vectors in a word vector space
- Each word in the vocabulary becomes a dimension in the vector space
- Words are associated with weights (e.g., term frequency)
- Relevance is modeled as cosine similarity between query and document vectors
- Allows ranking of documents based on similarity scores

### Probabilistic Relevance Model (Robertson & Spärck Jones, 1972)

- Views documents and queries as probability distributions over an underlying word space
- Models relevance as similarity between these distributions
- Incorporates statistical properties of term occurrence
- Pioneered by Karen Spärck Jones who established fundamental concepts for search engines

### Non-textual Signals

- PageRank (Page, Brin, et al., 1999) - exploits web link structure
    - Relevance affected by page authority
    - Pages with many incoming links from authoritative sources rank higher
- Usage data as a relevance signal
    - User behavior (clicks, dwell time) indicates document relevance
    - "Relevance is what other users believe"

### Semantic Search

- Focuses on entities and their relationships rather than just keywords
- Incorporates semantic understanding beyond simple term matching
- Aims to understand the meaning behind queries

### Digital Assistants

- Dialogue-based information access systems
- Provide natural interaction for accessing/storing knowledge
- Incorporate personalization based on user history and preferences

## 3. How Search Works: The Big Picture

### Search System Architecture

- Divided into offline and online components

#### Offline Components:

- **Crawling**: Discovering and fetching content
- **Quality/Freshness Assessment**: Evaluating document quality and recency
- **Pre-processing & Indexing**: Preparing documents for efficient retrieval

#### Online Components:

- **Query Understanding**: Interpreting user intent
- **Ranking Algorithm**: Combining signals to determine document order
- **Logging**: Recording user interactions for system improvement
- **Context & Personalization**: Adapting results to user context

## 4. Text Representation and Processing

### Text Representation Forms

- **Bag of Words**: Traditional model that ignores word order
    - "dog bites man" = "man bites dog" in this representation
    - Simple but loses semantic relationships
- **Modern Approaches**: Preserve order information in various ways
- **Representation Units**:
    - Words, stems, terms, phrases, entities
    - N-grams (unigrams, bigrams, etc.)
    - Visual words (for image retrieval): pixels, patches, etc.

### Text Processing Pipeline
1. **Tokenization**: Forming words from character sequences
    - Surprisingly complex process with many edge cases
    - Issues include: word boundaries, hyphenation, apostrophes, special characters, numbers
2. **Case Normalization**: Converting terms to lowercase
    - May cause issues with proper nouns (e.g., "Apple" vs "apple")
3. **Stopword Removal**: Eliminating common words
    - Frequency-based approach: remove words above frequency threshold
    - Dictionary-based approach: remove words from a predefined list
    - Improves efficiency by reducing index size
4. **Stemming**: Reducing words to their stems
    - Helps address lexical mismatch problem
    - Types:
        - Dictionary-based: uses lists of related words
        - Algorithmic: uses rules to determine related words (e.g., Porter Stemmer)
    - Example: "running," "runs," "runner" → "run"
    - Can produce errors: both false positives and false negatives
    - Generally provides modest but significant effectiveness improvement
5. **Phrase Detection**:
    - Methods include:
        - Using part-of-speech taggers to identify noun phrases
        - Finding frequent n-grams
        - Query-time phrase detection with position-aware indexes

## 5. Boolean Search and Indexing

### Boolean Search Model

- Documents represented as sets of words
- Queries are Boolean expressions with AND, OR, NOT operators
- Returns exact matches (no ranking)

#### Pros:

- Easy to understand for simple queries
- Clean formalism with precise semantics
- Very rigid: AND means all; OR means any

#### Cons:

- Difficult to express complex information needs
- Hard to control result set size
- No ranking of results (all matches satisfy query equally)

### Inverted Index

- Fundamental data structure for efficient retrieval
- For each term, store a list of documents containing it
- Postings: lists of document IDs associated with each term
- Efficient for retrieval but computationally expensive to build

#### Query Processing with Inverted Index:

- For AND queries: intersect document sets
- For OR queries: union document sets
- For NOT queries: complement document sets
- Process uses merge algorithms on sorted lists for efficiency
- Time complexity: O(x+y) for merging lists of length x and y

#### Positional Indexes:

- Enhanced inverted index that stores word positions
- Enables phrase queries like "stanford university"
- Format: `<term, doc_count; doc1: pos1, pos2...; doc2: pos1, pos2...>`
- Larger but more powerful than basic inverted indexes

### Inverted Index Construction

- **In-memory indexing**: Simple but limited by memory capacity
- **Merge-based indexing**: Build partial indexes and merge them
    - Build index until memory is full
    - Write partial index to disk
    - Merge partial indexes
- **Distributed indexing**: Using frameworks like MapReduce
    - Mappers process documents and emit (term, docID) pairs
    - Reducers aggregate postings for each term
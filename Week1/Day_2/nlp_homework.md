# Natural Language Processing Homework Assignment

## Overview

This homework assignment is designed to introduce you to fundamental Natural Language Processing (NLP) concepts and techniques. You will work with text data and implement various NLP tasks using Python libraries such as NLTK, spaCy, and scikit-learn.

## Dataset Description

You will be working with a collection of movie reviews. This dataset contains:

- 1000 movie reviews (500 positive, 500 negative)
- Each review is labeled as positive (1) or negative (0)
- Reviews are of varying lengths and complexity

## Tasks

### Task 1: Text Preprocessing (Beginner)

1. Load the dataset and examine its structure
2. Implement the following preprocessing steps:
   - Convert text to lowercase
   - Remove punctuation
   - Remove numbers
   - Remove stopwords
   - Remove extra whitespace
3. Tokenize the preprocessed text
4. Implement stemming and lemmatization on the tokens
5. Compare and discuss the results of stemming vs. lemmatization with 3 examples

### Task 2: Text Exploration and Visualization (Beginner-Intermediate)

1. Calculate basic text statistics:
   - Average review length (in words)
   - Distribution of review lengths
   - Vocabulary size
2. Identify the most common words in positive and negative reviews
3. Create a word cloud for positive and negative reviews
4. Generate and visualize n-gram frequencies (for n=2 and n=3)
5. Calculate and visualize TF-IDF scores for the top 20 terms

### Task 3: Named Entity Recognition (NER) Exploration (Intermediate)

1. Install and set up NLTK for entity recognition
2. Process a subset of movie reviews (at least 50) with NER and extract all entities
3. Categorize and count entities by type (PERSON, ORGANIZATION, LOCATION, etc.) found in positive vs. negative reviews
4. Create visualizations showing:
   - Distribution of entity types across the dataset
   - Frequency of top 10 entities for each entity type
   - Comparison of entity patterns between positive and negative reviews
5. Implement a custom entity recognition approach for movie-specific entities:
   - Directors (e.g., "Christopher Nolan", "Quentin Tarantino")
   - Actors/Actresses (e.g., "Tom Hanks", "Meryl Streep")
   - Movie titles (hint: look for capitalized phrases and patterns)
   - Awards (e.g., "Oscar", "Golden Globe")
6. Evaluate your custom NER approach on a small manually labeled test set
7. Create a function that takes a review text and returns a highlighted version showing all identified entities with their categories (using different colors or formatting)

## Dataset Generation

Use the following code to generate a synthetic movie review dataset:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create positive and negative vocabulary
positive_words = [
    'excellent', 'amazing', 'great', 'good', 'fantastic', 'wonderful', 'brilliant',
    'perfect', 'outstanding', 'superb', 'masterpiece', 'stunning', 'impressive',
    'enjoyable', 'entertaining', 'captivating', 'engaging', 'powerful', 'moving',
    'beautiful', 'compelling', 'memorable', 'remarkable', 'spectacular', 'phenomenal'
]

negative_words = [
    'terrible', 'awful', 'bad', 'poor', 'disappointing', 'boring', 'dull',
    'mediocre', 'waste', 'horrible', 'worst', 'stupid', 'annoying', 'predictable',
    'unbearable', 'ridiculous', 'failure', 'disaster', 'nonsense', 'mess',
    'underwhelming', 'forgettable', 'confusing', 'pointless', 'painful'
]

# Create lists of movie-related named entities for NER task
director_names = [
    'Steven Spielberg', 'Christopher Nolan', 'Martin Scorsese', 'Quentin Tarantino',
    'James Cameron', 'Kathryn Bigelow', 'Alfred Hitchcock', 'Ridley Scott',
    'Greta Gerwig', 'Sofia Coppola', 'Denis Villeneuve', 'Francis Ford Coppola',
    'David Fincher', 'Spike Lee', 'Wes Anderson', 'Ava DuVernay'
]

actor_names = [
    'Tom Hanks', 'Meryl Streep', 'Leonardo DiCaprio', 'Jennifer Lawrence',
    'Denzel Washington', 'Viola Davis', 'Brad Pitt', 'Cate Blanchett',
    'Robert De Niro', 'Kate Winslet', 'Morgan Freeman', 'Scarlett Johansson',
    'Daniel Day-Lewis', 'Emma Stone', 'Samuel L. Jackson', 'Natalie Portman'
]

movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight',
    'Schindler\'s List', 'Forrest Gump', 'Inception', 'The Matrix',
    'Titanic', 'Avatar', 'Parasite', 'Casablanca',
    'Goodfellas', 'The Silence of the Lambs', 'Jurassic Park', 'Star Wars'
]

award_names = [
    'Oscar', 'Academy Award', 'Golden Globe', 'BAFTA',
    'Palme d\'Or', 'Emmy', 'Screen Actors Guild Award', 'Tony Award',
    'Critics\' Choice', 'Independent Spirit Award', 'Cesar Award', 'Goya Award'
]

# Fetch some real texts to build more realistic reviews
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data[:5000]  # Get some real text

# Function to generate a synthetic review
def generate_review(sentiment, length_range=(50, 500)):
    # Select base text
    base_text = random.choice(texts)
    words = base_text.split()

    # Select random length within range
    target_length = random.randint(*length_range)
    if len(words) > target_length:
        words = words[:target_length]

    # Add sentiment words
    word_list = positive_words if sentiment == 1 else negative_words
    num_sentiment_words = random.randint(3, 10)

    for _ in range(num_sentiment_words):
        insert_pos = random.randint(0, len(words) - 1)
        sentiment_word = random.choice(word_list)
        words.insert(insert_pos, sentiment_word)

    # Add movie-related terms sometimes
    movie_terms = ['movie', 'film', 'cinema', 'director', 'actor', 'actress',
                   'script', 'screenplay', 'scene', 'plot', 'character', 'performance']

    for _ in range(random.randint(1, 5)):
        insert_pos = random.randint(0, len(words) - 1)
        movie_term = random.choice(movie_terms)
        words.insert(insert_pos, movie_term)

    # Add named entities to some reviews (for NER task)
    if random.random() < 0.7:  # 70% chance to add named entities
        # Add 1-3 director names
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.6:
                insert_pos = random.randint(0, len(words) - 1)
                director = random.choice(director_names)
                words.insert(insert_pos, director)

        # Add 1-3 actor names
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.7:
                insert_pos = random.randint(0, len(words) - 1)
                actor = random.choice(actor_names)
                words.insert(insert_pos, actor)

        # Add 0-2 movie titles
        for _ in range(random.randint(0, 2)):
            if random.random() < 0.5:
                insert_pos = random.randint(0, len(words) - 1)
                title = random.choice(movie_titles)
                words.insert(insert_pos, title)

        # Add 0-1 award mentions
        if random.random() < 0.3:
            insert_pos = random.randint(0, len(words) - 1)
            award = random.choice(award_names)
            words.insert(insert_pos, award)

    # Join and return
    review = ' '.join(words)

    # Clean up a bit
    review = review.replace('\n', ' ').replace('  ', ' ')

    return review

# Generate 1000 reviews (500 positive, 500 negative)
reviews = []
labels = []

for _ in range(500):
    # Generate positive reviews
    reviews.append(generate_review(1))
    labels.append(1)

    # Generate negative reviews
    reviews.append(generate_review(0))
    labels.append(0)

# Create DataFrame
reviews_df = pd.DataFrame({
    'review': reviews,
    'sentiment': labels
})

# Shuffle the DataFrame
reviews_df = reviews_df.sample(frac=1).reset_index(drop=True)

# Display sample
print(reviews_df.head())

# Save to CSV if needed
# reviews_df.to_csv('movie_reviews.csv', index=False)
```

## Hints

### Task 1 Hints:

- Use NLTK or spaCy for preprocessing tasks
- For stopwords removal: `from nltk.corpus import stopwords`
- For stemming: `from nltk.stem import PorterStemmer` or `SnowballStemmer`
- For lemmatization: `from nltk.stem import WordNetLemmatizer` or use spaCy's lemmatizer

### Task 2 Hints:

- Use Python's collections: `from collections import Counter`
- For word clouds: `from wordcloud import WordCloud`
- For visualization: `matplotlib` and `seaborn`
- For n-grams: `from nltk import ngrams`
- For TF-IDF visualization: Use scikit-learn's `TfidfVectorizer`

### Task 3 Hints:

- For NER setup with NLTK:
  ```python
  import nltk
  from nltk import ne_chunk
  from nltk.tag import pos_tag
  from nltk.tokenize import word_tokenize
  ```
- To extract named entities with NLTK:
  ```python
  def extract_entities(text):
      tokens = word_tokenize(text)
      tagged = pos_tag(tokens)
      chunks = ne_chunk(tagged)
      entities = []
      for chunk in chunks:
          if hasattr(chunk, 'label'):
              entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
      return entities
  ```
- For custom NER, consider regex patterns or context-based rules with dictionaries
- To evaluate NER performance, use precision, recall, and F1-score on entity detection
- For text highlighting in Jupyter notebooks, you can use HTML formatting:

  ```python
  from IPython.display import HTML, display

  def highlight_entities(text, entities):
      # Code that takes text and entity locations and returns highlighted HTML
      # (detailed implementation in solution)
  ```

## Submission Requirements

Submit a Jupyter notebook containing:

1. All code with clear comments
2. Visualizations with interpretations
3. Discussion of results and findings
4. Answers to all questions posed in the tasks

## Sample Solution for Task 3: Named Entity Recognition

```python
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from nltk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from IPython.display import display, HTML

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load the dataset (this would be your movie reviews dataset)
reviews_df = pd.read_csv('movie_reviews.csv')

# 1. Setup NLTK for entity recognition
def extract_entities_nltk(text):
    """Extract named entities using NLTK's ne_chunk."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)

    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join(c[0] for c in chunk)
            entities.append((chunk.label(), entity_text))

    return entities

# 2. Process a subset of reviews
sample_size = 50
review_sample = reviews_df.sample(sample_size, random_state=42)

# Store entities for analysis
all_entities = []
pos_entities = []
neg_entities = []

for idx, row in review_sample.iterrows():
    review_text = row['review']
    sentiment = row['sentiment']

    # Extract entities
    entities = extract_entities_nltk(review_text)

    # Store entities with review ID and sentiment
    for entity_type, entity_text in entities:
        entity_record = {
            'review_id': idx,
            'sentiment': sentiment,
            'entity_type': entity_type,
            'entity_text': entity_text
        }
        all_entities.append(entity_record)

        if sentiment == 1:
            pos_entities.append(entity_record)
        else:
            neg_entities.append(entity_record)

# Convert to DataFrame for easier analysis
entities_df = pd.DataFrame(all_entities)
print(f"Total entities found: {len(entities_df)}")

# 3. Categorize and count entities by type
entity_type_counts = entities_df['entity_type'].value_counts()
print("\nEntity types distribution:")
print(entity_type_counts)

# Compare entity types in positive vs negative reviews
pos_entity_types = [e['entity_type'] for e in pos_entities]
neg_entity_types = [e['entity_type'] for e in neg_entities]

pos_type_counts = Counter(pos_entity_types)
neg_type_counts = Counter(neg_entity_types)

# 4. Create visualizations
# Entity type distribution plot
plt.figure(figsize=(10, 6))
entity_type_counts.plot(kind='bar')
plt.title('Distribution of Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Top 10 entities for each type
plt.figure(figsize=(12, 8))
for i, entity_type in enumerate(entity_type_counts.index[:4]):  # Show top 4 entity types
    entities_of_type = entities_df[entities_df['entity_type'] == entity_type]
    top_entities = entities_of_type['entity_text'].value_counts().head(10)

    plt.subplot(2, 2, i+1)
    top_entities.plot(kind='barh')
    plt.title(f'Top 10 {entity_type} Entities')
    plt.tight_layout()

plt.show()

# Compare entity patterns between positive and negative reviews
comparison_data = pd.DataFrame({
    'Positive': pd.Series(pos_type_counts),
    'Negative': pd.Series(neg_type_counts)
}).fillna(0)

plt.figure(figsize=(10, 6))
comparison_data.plot(kind='bar')
plt.title('Entity Types in Positive vs Negative Reviews')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Implement custom entity recognition for movie-specific entities

# Load our lists of known entities
director_names = [
    'Steven Spielberg', 'Christopher Nolan', 'Martin Scorsese', 'Quentin Tarantino',
    'James Cameron', 'Kathryn Bigelow', 'Alfred Hitchcock', 'Ridley Scott',
    'Greta Gerwig', 'Sofia Coppola', 'Denis Villeneuve', 'Francis Ford Coppola',
    'David Fincher', 'Spike Lee', 'Wes Anderson', 'Ava DuVernay'
]

actor_names = [
    'Tom Hanks', 'Meryl Streep', 'Leonardo DiCaprio', 'Jennifer Lawrence',
    'Denzel Washington', 'Viola Davis', 'Brad Pitt', 'Cate Blanchett',
    'Robert De Niro', 'Kate Winslet', 'Morgan Freeman', 'Scarlett Johansson',
    'Daniel Day-Lewis', 'Emma Stone', 'Samuel L. Jackson', 'Natalie Portman'
]

movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight',
    'Schindler\'s List', 'Forrest Gump', 'Inception', 'The Matrix',
    'Titanic', 'Avatar', 'Parasite', 'Casablanca',
    'Goodfellas', 'The Silence of the Lambs', 'Jurassic Park', 'Star Wars'
]

award_names = [
    'Oscar', 'Academy Award', 'Golden Globe', 'BAFTA',
    'Palme d\'Or', 'Emmy', 'Screen Actors Guild Award', 'Tony Award',
    'Critics\' Choice', 'Independent Spirit Award', 'Cesar Award', 'Goya Award'
]

def custom_movie_ner(text):
    """Custom NER for movie-specific entities."""
    entities = []

    # Check for directors
    for director in director_names:
        if director.lower() in text.lower():
            # Find exact position with case preserved
            start = text.lower().find(director.lower())
            actual_text = text[start:start+len(director)]
            entities.append(('DIRECTOR', actual_text))

    # Check for actors
    for actor in actor_names:
        if actor.lower() in text.lower():
            start = text.lower().find(actor.lower())
            actual_text = text[start:start+len(actor)]
            entities.append(('ACTOR', actual_text))

    # Check for movie titles
    for title in movie_titles:
        if title.lower() in text.lower():
            start = text.lower().find(title.lower())
            actual_text = text[start:start+len(title)]
            entities.append(('MOVIE', actual_text))

    # Check for awards
    for award in award_names:
        if award.lower() in text.lower():
            start = text.lower().find(award.lower())
            actual_text = text[start:start+len(award)]
            entities.append(('AWARD', actual_text))

    # Additional pattern matching for potential movie titles
    # Look for patterns like capitalized words in quotes
    movie_pattern = r'"([A-Z][^"]+)"'
    movie_matches = re.findall(movie_pattern, text)
    for match in movie_matches:
        if match not in [m[1] for m in entities if m[0] == 'MOVIE']:
            entities.append(('POTENTIAL_MOVIE', match))

    return entities

# Apply custom NER to the sample
custom_entities = []

for idx, row in review_sample.iterrows():
    review_text = row['review']
    sentiment = row['sentiment']

    # Extract entities
    movie_entities = custom_movie_ner(review_text)

    # Store entities with review ID and sentiment
    for entity_type, entity_text in movie_entities:
        entity_record = {
            'review_id': idx,
            'sentiment': sentiment,
            'entity_type': entity_type,
            'entity_text': entity_text
        }
        custom_entities.append(entity_record)

# Convert to DataFrame
custom_entities_df = pd.DataFrame(custom_entities)
print(f"\nCustom movie entities found: {len(custom_entities_df)}")

# Count by entity type
custom_type_counts = custom_entities_df['entity_type'].value_counts()
print("\nCustom entity types distribution:")
print(custom_type_counts)

# Visualize custom entity types
plt.figure(figsize=(10, 6))
custom_type_counts.plot(kind='bar')
plt.title('Distribution of Custom Movie Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6. Evaluate custom NER on a small test set

# Let's create a small manually labeled test set
test_reviews = [
    {"text": "Steven Spielberg directed 'Jurassic Park' which won an Oscar for special effects.",
     "expected": [('DIRECTOR', 'Steven Spielberg'), ('MOVIE', 'Jurassic Park'), ('AWARD', 'Oscar')]},
    {"text": "I thought The Dark Knight was brilliant with amazing performances by Christian Bale.",
     "expected": [('MOVIE', 'The Dark Knight')]},
    {"text": "Quentin Tarantino's Pulp Fiction is a cult classic starring Samuel L. Jackson.",
     "expected": [('DIRECTOR', 'Quentin Tarantino'), ('MOVIE', 'Pulp Fiction'), ('ACTOR', 'Samuel L. Jackson')]},
    {"text": "I didn't enjoy Avatar despite its Golden Globe nominations.",
     "expected": [('MOVIE', 'Avatar'), ('AWARD', 'Golden Globe')]},
    {"text": "Martin Scorsese finally won an Academy Award for The Departed.",
     "expected": [('DIRECTOR', 'Martin Scorsese'), ('AWARD', 'Academy Award')]}
]

# Function to evaluate NER performance
def evaluate_ner(test_data, ner_function):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for example in test_data:
        text = example["text"]
        expected = set([(t, e) for t, e in example["expected"]])

        # Get predictions
        predicted = set([(t, e) for t, e in ner_function(text)])

        # Count TP, FP, FN
        true_positives += len(expected.intersection(predicted))
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Evaluate our custom NER
evaluation = evaluate_ner(test_reviews, custom_movie_ner)
print("\nCustom NER Evaluation:")
print(f"Precision: {evaluation['precision']:.2f}")
print(f"Recall: {evaluation['recall']:.2f}")
print(f"F1 Score: {evaluation['f1']:.2f}")

# 7. Create a function to highlight entities in text

def highlight_entities(text, entities):
    """
    Highlight entities in text with different colors based on entity type.
    Returns HTML for display in Jupyter notebook.
    """
    # Sort entities by their position in the text to handle overlapping entities correctly
    positioned_entities = []
    for entity_type, entity_text in entities:
        start = text.lower().find(entity_text.lower())
        while start != -1:
            # Verify the full word matches
            end = start + len(entity_text)
            before = '' if start == 0 else text[start-1]
            after = '' if end >= len(text) else text[end]
            if (start == 0 or not before.isalnum()) and (end >= len(text) or not after.isalnum()):
                positioned_entities.append((start, end, entity_type, text[start:end]))
                break
            start = text.lower().find(entity_text.lower(), start + 1)

    # Sort by start position, with longer entities first in case of ties
    positioned_entities.sort(key=lambda x: (x[0], -len(x[3])))

    # Define colors for different entity types
    color_map = {
        'PERSON': '#ffadad',  # light red
        'ORGANIZATION': '#ffd6a5',  # light orange
        'LOCATION': '#caffbf',  # light green
        'DIRECTOR': '#9bf6ff',  # light cyan
        'ACTOR': '#bdb2ff',  # light purple
        'MOVIE': '#ffc6ff',  # light pink
        'AWARD': '#fdffb6',  # light yellow
        'POTENTIAL_MOVIE': '#fffffc',  # off-white
        'GPE': '#caffbf',  # light green (same as LOCATION)
        'FACILITY': '#a0c4ff',  # light blue
        'DATE': '#e2e2e2'  # light gray
    }

    # Build HTML with highlighting
    html_parts = []
    last_end = 0

    for start, end, entity_type, entity_text in positioned_entities:
        if start > last_end:
            html_parts.append(text[last_end:start])

        color = color_map.get(entity_type, '#e2e2e2')  # default to light gray
        html_parts.append(f'<span style="background-color: {color};" title="{entity_type}">{entity_text}</span>')

        last_end = end

    if last_end < len(text):
        html_parts.append(text[last_end:])

    return HTML(''.join(html_parts))

# Demonstrate the highlighting function with a sample text
sample_text = "Steven Spielberg's Jurassic Park won an Oscar for its groundbreaking special effects. Tom Hanks and Leonardo DiCaprio are two of my favorite actors."
entities = custom_movie_ner(sample_text) + extract_entities_nltk(sample_text)
display(highlight_entities(sample_text, entities))

# Show highlighting for a few reviews from our dataset
for i, (idx, row) in enumerate(review_sample.head(5).iterrows()):
    review_text = row['review']
    print(f"\nReview {i+1} (Sentiment: {'Positive' if row['sentiment'] == 1 else 'Negative'}):")

    # Extract both standard and custom entities
    all_entities = extract_entities_nltk(review_text) + custom_movie_ner(review_text)

    # Display highlighted text
    display(highlight_entities(review_text, all_entities))
```

This solution demonstrates:

1. Setting up NLTK for entity recognition
2. Extracting and analyzing entities from movie reviews
3. Creating custom entity recognition for movie-specific entities
4. Evaluating NER performance
5. Highlighting entities in text with different colors

The code includes detailed comments and produces visualizations to help understand the entity patterns in the dataset.

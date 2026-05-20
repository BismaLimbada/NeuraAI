import json
import numpy as np
from preprocess import tokenize, stem, bag_of_words

# 1. Load the structured mental health intents dataset
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []  # Holds tuples of (tokenized_pattern_sentence, matching_tag_label)

# 2. Loop through each intent and sentence pattern in our JSON data
for intent in intents['intents']:
    tag = intent['tag']
    # Catalog unique tags
    if tag not in tags:
        tags.append(tag)
        
    for pattern in intent['patterns']:
        # Tokenize the sentence string into individual words
        w = tokenize(pattern)
        # Add the words to our master vocabulary tracking list
        all_words.extend(w)
        # Pair the tokenized words with their corresponding category tag
        xy.append((w, tag))

# 3. Clean the vocabulary list by stemming words and removing common punctuation symbols
ignore_words = ['?', '!', '.', ',', ';']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicate words and sort them alphabetically
all_words = sorted(list(set(all_words)))
tags = sorted(list(set(tags)))

print("--- Data Extraction Metrics ---")
print(f"Total pattern-tag pairs (xy size): {len(xy)}")
print(f"Unique mental health category tags cataloged: {len(tags)}")
print(f"Total unique stemmed words found in vocabulary: {len(all_words)}")
print("-------------------------------\n")

# 4. Create the final numerical training arrays for the AI model
X_train = []  # Input features: Numerical Bag of Words vectors
y_train = []  # Target labels: Numeric index positions corresponding to tags

for (pattern_sentence, tag) in xy:
    # Transform the tokenized sentence into a 1s and 0s vector array
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # Convert the string label (e.g., 'greeting') into its index integer position
    label = tags.index(tag)
    y_train.append(label)

# Convert Python lists to high-performance NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

print("Phase 1 Complete! Data matrices successfully generated.")
print(f"Input Matrix Shape (X_train): {X_train.shape} -> ({len(xy)} sentences, {len(all_words)} words each)")
print(f"Output Target Shape (y_train): {y_train.shape} -> ({len(xy)} corresponding category index assignments)")
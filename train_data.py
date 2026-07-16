import json
import numpy as np
from preprocess import embed_sentences

# 1. Load the structured mental health intents dataset
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []  # Holds tuples of (pattern_sentence, matching_tag_label)

# 2. Loop through each intent and sentence pattern in our JSON data
for intent in intents['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)

    for pattern in intent['patterns']:
        xy.append((pattern, tag))

tags = sorted(list(set(tags)))

print("--- Data Extraction Metrics ---")
print(f"Total pattern-tag pairs (xy size): {len(xy)}")
print(f"Unique mental health category tags cataloged: {len(tags)}")
print("-------------------------------\n")

# 3. Embed every pattern sentence with the Universal Sentence Encoder.
# This replaces the old vocabulary/bag-of-words step entirely - there's no
# more "all_words" list, since the model no longer depends on exact word
# overlap. Semantically similar sentences land close together in embedding
# space even if they don't share a single word.
print("Embedding all pattern sentences with Universal Sentence Encoder...")
pattern_sentences = [p for (p, _tag) in xy]
X_train = embed_sentences(pattern_sentences)

y_train = np.array([tags.index(tag) for (_p, tag) in xy])

print("Phase 1 Complete! Data matrices successfully generated.")
print(f"Input Matrix Shape (X_train): {X_train.shape} -> ({len(xy)} sentences, {X_train.shape[1]}-dim USE embeddings)")
print(f"Output Target Shape (y_train): {y_train.shape} -> ({len(xy)} corresponding category index assignments)")

import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# --- Sentence embedding (replaces bag-of-words) ---------------------------
# Universal Sentence Encoder maps a whole sentence to a 512-dim vector that
# captures meaning, not just exact word overlap. That's what lets the model
# recognize paraphrases it never saw during training (e.g. "my mind won't
# quit racing" vs. "I overthink everything") instead of requiring the user's
# words to literally match a pattern in intents.json.
_embedder = None
EMBEDDING_DIM = 512


def load_embedder():
    """
    Lazily loads the Universal Sentence Encoder from TensorFlow Hub.
    Loaded once per process and cached, since it's a large model
    (downloads on first use, ~1GB, then cached locally by tf-hub).
    """
    global _embedder
    if _embedder is None:
        import tensorflow_hub as hub
        print("Loading Universal Sentence Encoder (first run may take a while to download)...")
        _embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return _embedder


def embed_sentence(sentence):
    """
    Converts a raw sentence string into a 512-dim USE embedding vector.
    This is what both training (train_data.py) and inference (app.py)
    use as the model's input, replacing the old bag_of_words() vector.
    """
    embedder = load_embedder()
    vector = embedder([sentence]).numpy()[0]
    return vector.astype(np.float32)


def embed_sentences(sentences):
    """Batch version of embed_sentence - much faster for training on many patterns at once."""
    embedder = load_embedder()
    vectors = embedder(sentences).numpy()
    return vectors.astype(np.float32)


# --- Lightweight text utilities (still used for short-word routing in app.py) ---

def tokenize(sentence):
    """
    Splits text string into structural word array,
    forcing lowercase and ignoring raw punctuation noise.
    """
    sentence = sentence.lower()
    sentence = sentence.replace("won't", "will not")
    sentence = sentence.replace("can't", "cannot")
    sentence = sentence.replace("don't", "do not")
    sentence = sentence.replace("i'm", "i am")

    tokens = nltk.word_tokenize(sentence)
    cleaned_tokens = [w for w in tokens if w.isalnum()]
    return cleaned_tokens


def stem(word):
    """Returns the base lower root variant of a string."""
    return stemmer.stem(word.lower().strip())

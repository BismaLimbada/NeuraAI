import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Splits text string into structural word array, 
    forcing lowercase and ignoring raw punctuation noise.
    """
    # Clean contractions manually so they don't fracture token indices
    sentence = sentence.lower()
    sentence = sentence.replace("won't", "will not")
    sentence = sentence.replace("can't", "cannot")
    sentence = sentence.replace("don't", "do not")
    sentence = sentence.replace("i'm", "i am")
    
    tokens = nltk.word_tokenize(sentence)
    # Filter out lone hanging symbols like ?, !, ., ,
    cleaned_tokens = [w for w in tokens if w.isalnum()]
    return cleaned_tokens

def stem(word):
    """Returns the base lower root variant of a string."""
    return stemmer.stem(word.lower().strip())

def bag_of_words(tokenized_sentence, all_words):
    """
    Constructs an identical length multi-hot mathematical 
    binary vector array representing pattern densities.
    """
    stemmed_tokens = [stem(w) for w in tokenized_sentence]
    vector = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in stemmed_tokens:
            vector[idx] = 1.0
            
    return vector
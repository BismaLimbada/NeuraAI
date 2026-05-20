import json
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Download the specific NLTK tokenizer and punctuation packages required for Python 3.13+
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split a sentence string into an array of individual words/tokens.
    Example: "Hello world!" -> ["Hello", "world", "!"]
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Find the root/base form of a word and convert it to lowercase.
    Example: "Anxious" -> "anxi", "Hurting" -> "hurt"
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Return a numerical bag of words array:
    1.0 for each known vocabulary word that exists in the sentence, 0.0 otherwise.
    
    Example:
    tokenized_sentence = ["hello", "how", "are", "you"]
    all_words          = ["hi", "hello", "i", "you", "bye", "thank"]
    bag                = [  0.,    1.  ,  0.,   1. ,   0. ,    0.   ]
    """
    # 1. Stem each word in the user's input sentence to match root forms
    stemmed_sentence = [stem(w) for w in tokenized_sentence]
    
    # 2. Initialize an array of zeros with the exact length of the master vocabulary list
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # 3. Loop through the master word list. If a word matches the user's input, set its index to 1.0
    for idx, w in enumerate(all_words):
        if w in stemmed_sentence: 
            bag[idx] = 1.0
            
    return bag

# Local execution block to test your functions
if __name__ == "__main__":
    print("--- Running Local Verification Tests ---")
    
    # Test 1: Basic Tokenization & Stemming
    test_sentence = "I am feeling incredibly overwhelmed and anxious today."
    print(f"Original: {test_sentence}")
    
    tokenized_words = tokenize(test_sentence)
    print(f"Tokenized: {tokenized_words}")
    
    # Strip out standard punctuation marks for testing clean words
    stemmed_words = [stem(w) for w in tokenized_words if w not in ['?', '!', '.', ',']]
    print(f"Stemmed: {stemmed_words}\n")
    
    # Test 2: Bag of Words Vectorizer Verification
    mock_vocabulary = ["hi", "hello", "i", "you", "by", "thank", "anxi"]
    mock_user_input = ["hello", "how", "are", "you"]
    
    # This should flag "hello" (index 1) and "you" (index 3) inside the vocabulary matrix
    vector_result = bag_of_words(mock_user_input, mock_vocabulary)
    print(f"Test Vocabulary: {mock_vocabulary}")
    print(f"Test User Input: {mock_user_input}")
    print(f"Bag of Words Vector Result: {vector_result}")
    print("----------------------------------------")
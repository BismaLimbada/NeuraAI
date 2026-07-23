# 🌸 Neura AI

Neura AI is an empathetic, context-aware chatbot framework built to support mental health awareness. It combines a semantic sentence-embedding classification engine with a Model-Based Reflex Agent architecture, implemented with TensorFlow, NLTK, and Streamlit, to maintain conversational state, track emotional progression, and manage multi-turn dialogue safely.

**Status:** Live Deployment Active

### Quick Links
* **Live Web Application:** [neuraai.streamlit.app](https://neuraai.streamlit.app/)
* **Development Repository:** [GitHub Root](https://github.com/BismaLimbada/ChatBot)

---

## Key Features

* **Semantic Intent Matching** — Uses Universal Sentence Encoder (USE) sentence embeddings rather than exact word overlap, allowing the agent to recognize paraphrased input it never saw during training.
* **Deterministic Crisis & Self-Harm Safety Net** — A keyword-based check for suicidal ideation and self-harm disclosures runs first, on every turn, ahead of all other logic. A match clears any pending conversational context, activates a crisis lockout, and returns mental health helpline and emergency contact information.
* **Contextual Yes/No Resolution** — When a followup is active (e.g. offering a coping exercise), a dedicated keyword and semantic-anchor check determines agreement, disagreement, or topic change, independent of the general intent classifier.
* **Model-Based Reflex Agent State** — Tracks conversational context and emotional signals dynamically across the session using persistent state buffers.
* **Greeting Duplication Prevention** — Detects repeat greetings mid-conversation and responds conversationally instead of reintroducing itself.
* **Smart Fallback Management** — Gracefully handles unrecognized input or low-confidence matches with a structured fallback response.

---

## Pipeline Architecture

The core processing pipeline evaluates each user turn in the following order:

1. **Crisis & Self-Harm Detection** — A deterministic keyword check scans for suicidal ideation or self-harm language. A match immediately clears any active context, enables crisis lockout mode, and returns helpline information. This step runs before context resolution so a crisis disclosure cannot be absorbed by a pending followup question.
2. **Crisis Lockout & Correction Guards** — If crisis mode is already active, the lockout response is shown. Otherwise, an NLTK-based negation check detects whether the user is correcting a misclassified context.
3. **Context Resolution** — If a followup context is open, the reply is classified as agreement, disagreement, or topic change using a keyword table and semantic anchor comparison, rather than the full intent classifier.
4. **Semantic Intent Matching** — Input is embedded using the Universal Sentence Encoder and compared via cosine similarity against precomputed pattern embeddings from `intents.json`. The closest match above a 0.42 confidence threshold determines the response.
5. **State Management** — Based on the matched intent and confidence, the agent updates emotion-tracking counters, sets new followup context, or returns to standard conversation.

---

## Technology Stack

* **Python** — Core application logic
* **TensorFlow / TensorFlow Hub** — Universal Sentence Encoder for semantic embeddings
* **NLTK** — Tokenization, stemming, and rule-based negation detection
* **Streamlit** — Web interface and session state management
* **NumPy** — Vector operations for similarity scoring

---

## Project Structure

```text
├── data/
│   └── intents.json           # Tags, patterns, responses, and followup prompts
├── .gitignore
├── README.md
├── app.py                     # Streamlit interface, semantic matching, and state management
├── data_mappings.json          # Registered tag list
├── preprocess.py               # Sentence embedding, tokenization, and stemming utilities
├── requirements.txt
├── train_data.py               # Embeds intents.json patterns for training
└── train_model.py              # Neural network training routine
```

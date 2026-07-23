# Neura AI — Mental Health Awareness ChatBot

## Objective

Neura AI is a context-aware conversational chatbot designed to support mental health awareness. Traditional chatbot interfaces treat each message as an independent, stateless snippet, which creates a disjointed experience when a user discusses an ongoing concern (e.g. panic attacks, burnout, overthinking) across several turns. The objective of this project is to build a chatbot that:

* Classifies user messages into mental-health-relevant intents using an NLP pipeline built with Python, NLTK, and TensorFlow.
* Maintains conversational memory across turns using a Model-Based Reflex Agent architecture, so followup questions (e.g. "would you like to try a grounding exercise?") are correctly resolved on the next turn.
* Detects and safely responds to crisis and self-harm disclosures ahead of all other processing.
* Tracks emotional signals across a conversation and reflects them back to the user through a simple visual interface.

---

## Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BismaLimbada/ChatBot
   cd ChatBot
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the required NLTK data** (this also happens automatically on first run of `app.py`):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

5. **Confirm the following project files are present** in the project root before running:
   * `data/intents.json` — intent patterns, responses, and followup prompts
   * `data_mappings.json` — registered tag list used by the trained model
   * `mental_health_model.keras` — trained classification model

---

## Required Libraries

| Library | Purpose |
|---|---|
| `streamlit` | Web interface and session state management |
| `tensorflow` | Neural network training and inference |
| `tensorflow-hub` | Loads the Universal Sentence Encoder for sentence embeddings |
| `numpy` | Vector and similarity operations |
| `nltk` | Tokenization, stemming, and rule-based negation detection |

Exact version constraints are listed in `requirements.txt`.

---

## Instructions on How to Run the Project

**Option 1 — Run locally:**
```bash
streamlit run app.py
```
Streamlit will start a local development server and print a URL (typically `http://localhost:8501`). Open this URL in a browser to interact with the chatbot.

**Option 2 — Use the live deployment:**
The application is already deployed and publicly accessible at:
[https://neuraai.streamlit.app/](https://neuraai.streamlit.app/)

**To retrain the classification model from scratch (optional):**
```bash
python train_data.py     # embeds intents.json patterns using the Universal Sentence Encoder
python train_model.py    # trains and saves mental_health_model.keras + data_mappings.json
```

---

## Expected Output

On launch, the application displays a chat interface titled **Neura AI** with a pastel-themed message window and an input box at the bottom. A sidebar shows the agent's internal state, including four emotional tracking scales (anxious, depressed, exhausted, frustrated), the currently active conversational context token, and a running turn counter.

Typical interaction flow:

1. The user sends a message describing how they feel (e.g. *"I can't stop overthinking everything"*).
2. The bot classifies the message and responds with an empathetic, relevant reply, in some cases offering a followup (e.g. a grounding or breathing exercise).
3. If the user replies to that followup, the bot correctly interprets agreement, disagreement, or a change of topic — even if the reply isn't an exact "yes" or "no" — and continues appropriately.
4. The sidebar's emotional scales update in real time to reflect the conversation.
5. If a message contains language indicating suicidal ideation or self-harm, the bot immediately responds with mental health helpline and emergency contact information and pauses ordinary conversation until the session is reset.
6. If a message doesn't match any known intent with sufficient confidence, the bot responds with a supportive fallback message rather than an irrelevant or broken reply.

---

## Key Features

* **Semantic Intent Matching** — Classifies input using a trained neural network over Universal Sentence Encoder (USE) sentence embeddings, allowing recognition of paraphrased input never seen during training.
* **Deterministic Crisis & Self-Harm Safety Net** — A keyword-based check for suicidal ideation and self-harm disclosures runs first, on every turn, ahead of all other logic.
* **Contextual Yes/No Resolution** — Determines agreement, disagreement, or topic change during an active followup using a dedicated keyword and semantic-anchor check.
* **Model-Based Reflex Agent State** — Tracks conversational context and emotional signals dynamically using persistent session state.
* **Greeting Duplication Prevention** — Detects repeat greetings mid-conversation and responds conversationally instead of reintroducing itself.
* **Smart Fallback Management** — Gracefully handles unrecognized input or low-confidence matches.

---

## Pipeline Architecture

The core processing pipeline evaluates each user turn in the following order:

1. **Crisis & Self-Harm Detection** — A deterministic keyword check scans for suicidal ideation or self-harm language. A match immediately clears any active context, enables crisis lockout mode, and returns helpline information.
2. **Crisis Lockout & Correction Guards** — If crisis mode is already active, the lockout response is shown. Otherwise, an NLTK-based negation check detects whether the user is correcting a misclassified context.
3. **Context Resolution** — If a followup context is open, the reply is classified as agreement, disagreement, or topic change using a keyword table and semantic anchor comparison.
4. **Semantic Intent Matching** — Input is embedded using the Universal Sentence Encoder and classified by the trained neural network (`mental_health_model.keras`) against the registered tag set in `data_mappings.json`.
5. **State Management** — Based on the matched intent and confidence, the agent updates emotion-tracking counters, sets new followup context, or returns to standard conversation.

---

## Project Structure

```text
├── data/
│   └── intents.json           # Tags, patterns, responses, and followup prompts
├── .gitignore
├── README.md
├── app.py                     # Streamlit interface, classification, and state management
├── chat_logic.py               # Crisis detection, yes/no resolver, and negation logic
├── data_mappings.json          # Registered tag list
├── preprocess.py               # Sentence embedding, tokenization, and stemming utilities
├── mental_health_model.keras   # Trained classification model
├── requirements.txt
├── tests/
│   └── test_bot.py             # Unit tests for deterministic logic paths
├── train_data.py               # Embeds intents.json patterns for training
└── train_model.py              # Neural network training routine
```

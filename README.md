# 🌸 Neura AI

Neura AI is an empathetic, context-aware chatbot framework built to support mental health awareness. Powered by a Model-Based Reflex Agent architecture implemented via TensorFlow, NLTK, and Streamlit, the system actively monitors and logs conversational state dynamics, tracked emotional progressions, and strict context-driven verification loops over multi-turn exchanges.

> **Status:** Live Deployment Active

An empathetic, context-aware Model-Based Reflex Agent framework designed for mental health awareness, emotional scaling, and supportive dialogue.

### 🔗 Quick Links
* **Live Web Application:** [neuraai.streamlit.app](https://neuraai.streamlit.app/)
* **Development Repository:** [GitHub Root](https://github.com/BismaLimbada/ChatBot)

---

## Key Features

* **Model-Based Reflex Agent State:** Tracks conversational contexts dynamically across user exchanges using localized historical context vectors.
* **Semantic Intent Matching:** Uses Universal Sentence Encoder (USE) sentence embeddings instead of exact word overlap, so the agent recognizes paraphrases it never saw during training (e.g. "my mind won't quit racing" vs. "I overthink everything").
* **Dedicated Context Yes/No Resolver:** When a followup is active (e.g. offering a grounding exercise), a focused keyword + semantic-anchor check decides agreement / disagreement / topic-change — rather than re-running the full intent classifier, which used to lose track of the active followup on any reply that wasn't an exact "yes"/"no".
* **Greeting Duplication Prevention:** Tracks state variables under-the-hood to identify initial greetings, ensuring the agent provides supportive, conversational replies instead of generic introductory messages if a greeting tag is triggered mid-dialogue.
* **Context State Machine Loops:** Employs explicit guard structures to seamlessly process confirmations (e.g., grounding rules, sleep advice, or burnout recovery tasks) before moving to generic fallback or deep semantic matching.
* **Smart Fallback Management:** Gracefully flags unrecognized inputs or text scoring below the similarity confidence limit, keeping conversations grounded and safe.

---

## Pipeline Architecture

The core interactive processing script operates sequentially:

1. **Phase 1: Crisis & Correction Guards** – Before anything else, the pipeline checks whether crisis-lockout mode is active, and whether the user is correcting a misclassified context (e.g. "no I didn't mean that") via an NLTK negation/keyword check.
2. **Phase 2: Context Memory Validation** – If `st.session_state.active_context` is open, the input is routed to a dedicated yes/no/topic-change resolver *before* the general intent matcher runs. This resolver first checks an expanded keyword table (substring match, English + Roman Urdu), then falls back to cosine similarity against a small set of agreement/disagreement anchor phrases — deliberately kept separate from the full 28-intent matcher so short replies aren't diluted by unrelated categories. Only if neither signal is clear does the context release and fall through to standard classification.
3. **Phase 3: Sentence Embedding & Semantic Matching** – Raw input text is embedded in full (not tokenized into a bag-of-words) using Google's Universal Sentence Encoder via TensorFlow Hub, producing a 512-dimension vector. This is compared via cosine similarity against precomputed embeddings of every pattern in `intents.json`; the closest match above a 0.42 confidence floor determines the predicted tag. NLTK tokenization/stemming is retained only for the lightweight short-word router and the negation-correction layer above — it no longer feeds the main classifier.
4. **Phase 4: Multi-Turn State Management** – Based on matching confidence thresholds, it triggers empathetic intent templates, updates runtime emotion-tracking counters, or sets new tracking milestones inside `st.session_state.active_context`.

> **Note on `train_model.py` / `mental_health_model.keras`:** the training script compiles a feedforward Keras classifier (Dense 128 → Dropout → Dense 64 → Dropout → Softmax) over the same USE embeddings. `app.py` currently performs inference via direct nearest-neighbor cosine similarity against pattern embeddings rather than loading this trained model — the two are not yet wired together. Worth resolving before final submission so the report and the running app describe the same classification path.

---

## Project Structure

```text
├── data/
│   └── intents.json           # Structural dataset containing tags, patterns, responses, and followup prompts
├── .gitignore                 # Specifies intentionally untracked build files to ignore
├── README.md                  # Project documentation overview
├── app.py                     # Main Streamlit user interface, semantic matching, context resolver, & state tracking
├── data_mappings.json          # Compiled tag list only (no vocabulary needed — inference re-embeds raw sentences)
├── preprocess.py               # USE sentence embedding, plus lightweight tokenization/stemming for routing & negation checks
├── requirements.txt            # Project environment dependencies
├── train_data.py               # Embeds intents.json patterns with USE for training
└── train_model.py              # Feedforward neural network compilation and training routine script
```
import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, embed_sentence, load_embedder
import nltk

# --- NLTK INITIALIZATION ---
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk()

# =========================================================
# NLP ENGINE: TRAINED NEURAL NETWORK CLASSIFIER
# =========================================================
# Previously this app classified via nearest-neighbor cosine similarity
# against every raw pattern in intents.json (recomputed at every cold
# start), while a *separate* feedforward network was trained by
# train_model.py and saved to mental_health_model.keras but never actually
# loaded here. That meant the trained model and the deployed classifier
# were two different things -- this class loads and uses the trained
# model directly, so the report, the training script, and the running app
# now describe the same classification path. It also removes the need to
# re-embed every pattern in intents.json on every app restart.
class TrainedIntentClassifier:
    def __init__(self, model_path="mental_health_model.keras",
                 mappings_path="data_mappings.json",
                 intents_path="data/intents.json"):
        self.model = tf.keras.models.load_model(model_path)

        with open(mappings_path, "r") as f:
            self.tags = json.load(f)["tags"]

        with open(intents_path, "r") as f:
            self.intents_data = json.load(f)

        self.intent_by_tag = {
            intent["tag"]: intent for intent in self.intents_data["intents"]
        }

    def match(self, user_input, threshold=0.42):
        """Embeds the input and classifies it with the trained network."""
        user_vector = embed_sentence(user_input)
        probs = self.model.predict(user_vector[np.newaxis, :], verbose=0)[0]

        best_idx = int(np.argmax(probs))
        best_score = float(probs[best_idx])
        predicted_tag = self.tags[best_idx]

        if best_score >= threshold:
            return self.intent_by_tag.get(predicted_tag), best_score
        return None, best_score


from chat_logic import (
    classify_yes_no,
    detect_crisis_language,
    CRISIS_RESPONSE,
    user_is_correcting_bot,
)


# =========================================================
# STREAMLIT UI &amp; RESOURCES
# =========================================================
st.set_page_config(page_title="Neura AI", page_icon="🌸", layout="centered")

# Custom UI Styling
st.html("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { text-align: center; margin-bottom: 0px; }
        div[data-testid="stCaptionContainer"] { text-align: center; }
        [data-testid="stChatMessageAvatarAssistant"] { background-color: black !important; color: #ffb6c1 !important; }
        [data-testid="stChatMessageAvatarUser"] { background-color: black !important; color: #ffb6c1 !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { color: black !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background-color: #FFC5D3; border-radius: 15px; padding: 10px; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-radius: 15px; padding: 10px; }
        .stButton > button { background-color: #FFC5D3 !important; color: black !important; border: 3px solid #ff9fbc !important; font-weight: 600; }
        .stButton > button:hover { background-color: #FFC5D3 !important; color: black !important; }
        .stProgress > div > div > div > div { background-color: #ffb6c1 !important; }
        [data-testid="stChatInput"] { background-color: #FFC5D3 !important; border-radius: 15px !important; border: 2px solid #ff9fbc !important; padding: 8px !important; }
        [data-testid="stChatInput"] > div { background-color: #FFC5D3 !important; border-radius: 15px !important; }
        [data-testid="stChatInput"] textarea { background-color: #FFC5D3 !important; color: black !important; }
        [data-testid="stChatInput"] textarea::placeholder { color: black !important; opacity: 1 !important; }
    </style>
""")

# Main Header Design
st.markdown("""
<div style="background-color:#FFC5D3; padding:20px; border:3px solid #ff9fbc; border-radius:20px; text-align:center; margin-bottom:20px; color:black;">
    <a href="https://neuraai.streamlit.app/" target="_blank" style="text-decoration: none; color: black;">
        <h1 style="margin-bottom:5px; color:black; cursor:pointer;">Neura AI</h1>
    </a>
    <p style="font-size:16px;">Your empathetic, context-aware framework for mental health awareness & support.</p>
</div>
""", unsafe_allow_html=True)

# Optimized Resource Loading
@st.cache_resource
def load_bot_resources():
    load_embedder()  # Load Universal Sentence Encoder
    matcher = TrainedIntentClassifier(
        model_path="mental_health_model.keras",
        mappings_path="data_mappings.json",
        intents_path="data/intents.json",
    )
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return matcher, intents

matcher, intents = load_bot_resources()

# --- MODEL-BASED REFLEX AGENT: STATE ARCHITECTURE SETUP ---
if "agent_internal_state" not in st.session_state:
    st.session_state.agent_internal_state = {
        "emotion_counters": {"anxious": 0, "depressed": 0, "exhausted": 0, "frustrated": 0},
        "crisis_mode_active": False,
        "total_conversation_turns": 0,
        "has_greeted": False
    }

if "active_context" not in st.session_state:
    st.session_state.active_context = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Emotion mapping
EMOTION_TAG_ROUTING = {
    "panic_attack": "anxious", "overthinking": "anxious", "night_anxiety": "anxious",
    "social_anxiety": "anxious", "health_anxiety": "anxious", "sleep_problems": "anxious",
    "loneliness": "depressed", "general_distress": "depressed", "low_self_esteem": "depressed",
    "grief": "depressed", "self_isolation": "depressed",
    "motivation_issues": "exhausted", "burnout": "exhausted", "academic_pressure": "exhausted",
    "anger_frustration": "frustrated", "family_pressure": "frustrated", "relationship_stress": "frustrated"
}

FOLLOWUP_BY_TAG = {
    intent["tag"]: intent["followup"]
    for intent in intents["intents"]
    if "followup" in intent
}
CONTEXT_SETTERS = {tag: f"awaiting_{tag}_followup" for tag in FOLLOWUP_BY_TAG}
CONTEXT_TO_TAG = {v: k for k, v in CONTEXT_SETTERS.items()}

with st.sidebar:
    st.subheader("🤖 Agent Internal Model State")
    st.write("This panel tracks how the Model-Based Reflex Agent builds its perception of the conversation state:")

    state = st.session_state.agent_internal_state
    st.progress(min(state["emotion_counters"]["anxious"] * 25, 100), text=f"Anxiety Scale: {state['emotion_counters']['anxious']}")
    st.progress(min(state["emotion_counters"]["depressed"] * 25, 100), text=f"Depression Scale: {state['emotion_counters']['depressed']}")
    st.progress(min(state["emotion_counters"]["exhausted"] * 25, 100), text=f"Exhaustion Scale: {state['emotion_counters']['exhausted']}")
    st.progress(min(state["emotion_counters"]["frustrated"] * 25, 100), text=f"Frustration Scale: {state['emotion_counters']['frustrated']}")

    st.divider()
    st.caption(f"Active Context Token: `{st.session_state.active_context}`")
    st.caption(f"Global Dialogue Index: {state['total_conversation_turns']}")

    if st.button("Reset Session History & Internal State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_context = None
        st.session_state.agent_internal_state = {
            "emotion_counters": {"anxious": 0, "depressed": 0, "exhausted": 0, "frustrated": 0},
            "crisis_mode_active": False,
            "total_conversation_turns": 0,
            "has_greeted": False
        }
        st.rerun()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CONVERSATIONAL CONTEXT LOOP ---
if user_input := st.chat_input("Type something here..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.agent_internal_state["total_conversation_turns"] += 1

    reply = None
    predicted_tag = None
    confidence = 0.0
    cleaned_choice = user_input.strip().lower().replace(".", "").replace("!", "")

    with st.spinner("Processing..."):
        # =========================================================
        # 1. CRISIS TRIGGER GUARD
        # =========================================================
        if st.session_state.agent_internal_state["crisis_mode_active"]:
            reply = (
                "Please reach out to a trusted professional or contact "
                "the Umang Helpline at 0311-7786264 immediately. "
                "For safety purposes, active chatbot configurations are suspended."
            )

        # =========================================================
        # 1.5 DETERMINISTIC CRISIS / SELF-HARM CHECK (runs before ANYTHING else,
        #     including the active-context resolver, so a crisis disclosure can
        #     never get swallowed by a pending yes/no followup)
        # =========================================================
        if reply is None and detect_crisis_language(cleaned_choice):
            st.session_state.active_context = None
            st.session_state.agent_internal_state["crisis_mode_active"] = True
            predicted_tag = "crisis_support"
            confidence = 1.0
            reply = CRISIS_RESPONSE
            print("[DEBUG] Deterministic crisis keyword match -- crisis mode activated.")

        # =========================================================
        # 2. RULE-BASED NEGATION DETECTOR (BREAK THE LOOP)
        # =========================================================
        if reply is None and user_is_correcting_bot(user_input, st.session_state.active_context):
            st.session_state.active_context = None
            reply = "I apologize for misinterpreting your situation. Let's start fresh. Can you tell me more about what you are currently feeling?"

        # =========================================================
        # 3a. ACTIVE CONTEXT RESOLUTION (dedicated yes/no/topic-change check)
        # =========================================================
        # This runs BEFORE the general 28-way intent matcher whenever the bot
        # is mid-followup (e.g. it just offered a coping exercise). It keeps
        # "memory" of that followup across a much wider range of phrasing
        # than an exact "yes"/"no" match would allow.
        if reply is None and st.session_state.active_context is not None:
            followup_tag = CONTEXT_TO_TAG.get(st.session_state.active_context)
            followup_cfg = FOLLOWUP_BY_TAG.get(followup_tag)

            if followup_cfg:
                decision, yn_confidence = classify_yes_no(user_input, cleaned_choice)

                if decision == "yes":
                    reply = followup_cfg["yes_reply"]
                    predicted_tag = "affirmative_response"
                    confidence = yn_confidence
                    st.session_state.active_context = None
                elif decision == "no":
                    reply = followup_cfg["no_reply"]
                    predicted_tag = "negative_response"
                    confidence = yn_confidence
                    st.session_state.active_context = None
                else:
                    # Genuinely unclear / topic changed - release context and
                    # let the standard flow below classify it fresh.
                    st.session_state.active_context = None
                    print("[DEBUG] Context reply was ambiguous; topic considered changed, context cleared.")
            else:
                st.session_state.active_context = None

        # =========================================================
        # 3b. STANDARD SEMANTIC INTENT MATCHING
        # =========================================================
        if reply is None:
            # Deterministic override for common conversational responses typed
            # in isolation (i.e. NOT inside an active followup, which is
            # handled above by classify_yes_no instead).
            SHORT_WORD_ROUTER = {
                "yes": "affirmative_response", "yeah": "affirmative_response", "yup": "affirmative_response",
                "haan": "affirmative_response", "ji": "affirmative_response", "sure": "affirmative_response",
                "no": "negative_response", "nah": "negative_response", "nope": "negative_response", "nahi": "negative_response",
                "okay": "neutral_acknowledgment", "ok": "neutral_acknowledgment", "nothing": "neutral_acknowledgment"
            }

            if cleaned_choice in SHORT_WORD_ROUTER:
                predicted_tag = SHORT_WORD_ROUTER[cleaned_choice]
                confidence = 1.0
                matched_intent = next((intent for intent in intents["intents"] if intent["tag"] == predicted_tag), None)
            else:
                matched_intent, confidence = matcher.match(user_input)
                predicted_tag = matched_intent["tag"] if matched_intent else None

            print(f"[DEBUG] Input: '{user_input}' | Prior Context: {st.session_state.active_context}")
            print(f"[DEBUG] Best Semantic Match: Tag='{predicted_tag}' | Confidence Score: {confidence:.2f}")

            # Handle false positive "goodbye" triggers on inputs like "leave"
            farewell_words = {"bye", "goodbye", "leave", "leaving", "going", "exit"}
            farewell_phrases = ["allah hafiz", "khuda hafiz"]
            farewell_tokens = set(tokenize(cleaned_choice))
            is_explicit_farewell = bool(farewell_tokens & farewell_words) or any(
                phrase in cleaned_choice for phrase in farewell_phrases
            )
            if predicted_tag == "goodbye" and not is_explicit_farewell:
                confidence = 0.10

            if confidence < 0.42:
                # Fallback triggers
                fallback_pool = intents.get("fallback_responses", [
                    "I hear you, but I want to make sure I understand correctly. Could you describe how you're feeling a bit differently?"
                ])
                reply = random.choice(fallback_pool)
            else:
                if predicted_tag in ["greeting", "islamic_greeting"]:
                    if st.session_state.agent_internal_state["has_greeted"]:
                        reply = random.choice([
                            "I'm right here with you. What's on your mind?",
                            "I'm listening. Tell me more about how you've been doing.",
                            "I'm here. Whenever you're ready, feel free to share what's going on."
                        ])
                    else:
                        reply = random.choice(matched_intent['responses'])
                        st.session_state.agent_internal_state["has_greeted"] = True

                elif predicted_tag in FOLLOWUP_BY_TAG:
                    # Set context and prompt the user
                    reply = FOLLOWUP_BY_TAG[predicted_tag]["prompt"]
                    st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]
                    print(f"[DEBUG] Context set to: {st.session_state.active_context}")

                else:
                    reply = random.choice(matched_intent['responses'])

        # =========================================================
        # 4. AGENT STATE UPDATE
        # =========================================================
        explicit_depressed_keywords = ["depressed", "depression", "sad", "hopeless", "lonely", "isolated"]
        if any(word in cleaned_choice.split() for word in explicit_depressed_keywords):
            target_dimension = "depressed"
        elif predicted_tag in EMOTION_TAG_ROUTING:
            target_dimension = EMOTION_TAG_ROUTING[predicted_tag]
        else:
            target_dimension = None

        if target_dimension:
            st.session_state.agent_internal_state["emotion_counters"][target_dimension] += 1

        if predicted_tag == "crisis_support":
            st.session_state.agent_internal_state["crisis_mode_active"] = True

    # Render Assistant Output
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

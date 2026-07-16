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
# ADVANCED NLP ENGINE: SEMANTIC INTENT MATCHER (VECTOR SEARCH)
# =========================================================
class SemanticIntentMatcher:
    def __init__(self, intents_path="data/intents.json"):
        with open(intents_path, "r") as f:
            self.intents_data = json.load(f)

        self.patterns = []
        self.pattern_embeddings = []
        self.pattern_to_intent = []  # Maps each pattern directly to its intent dict

        self._precompute_pattern_embeddings()

    def _precompute_pattern_embeddings(self):
        """Embeds every single pattern in intents.json once at startup."""
        for intent in self.intents_data["intents"]:
            for pattern in intent["patterns"]:
                # Use your existing USE embedder from preprocess.py
                vector = embed_sentence(pattern)
                self.patterns.append(pattern)
                self.pattern_embeddings.append(vector)
                self.pattern_to_intent.append(intent)

        self.pattern_embeddings = np.array(self.pattern_embeddings)

    def match(self, user_input, threshold=0.42):
        """Finds the semantically closest pattern using Cosine Similarity."""
        user_vector = embed_sentence(user_input)

        # Cosine Similarity = Dot Product (since USE embeddings are L2 normalized)
        similarities = np.dot(self.pattern_embeddings, user_vector)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= threshold:
            return self.pattern_to_intent[best_idx], best_score
        return None, best_score


# =========================================================
# CONTEXTUAL YES / NO / TOPIC-CHANGE RESOLVER
# =========================================================
# Why this exists:
# When the bot is inside a followup (e.g. "Would you like to try a grounding
# exercise?"), we do NOT want to ask "which of the 28 intents is this?" — a
# short reply like "not really" or "yeah kind of" will compete against
# academic_pressure / burnout / etc. patterns and often loses, which is why
# the bot used to forget what it had just asked and fall back or reset.
#
# Instead, this resolver ONLY decides between three options: agree, disagree,
# or "the user moved on to something else" — using (1) a broad keyword table
# and (2) similarity against a small anchor set, rather than the full intent
# matcher. This keeps the model-based reflex agent's "memory" of the active
# followup intact across a much wider range of real phrasing.

AGREEMENT_KEYWORDS = [
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "fine",
    "definitely", "absolutely", "of course", "i do", "i guess so", "kind of",
    "kinda", "sort of", "a bit", "somewhat", "i think so", "probably",
    "haan", "ji", "ji haan", "bilkul", "theek hai",
]

DISAGREEMENT_KEYWORDS = [
    "no", "nah", "nope", "not really", "not at all", "i do not", "i don't",
    "never", "not exactly", "not particularly", "don't think so",
    "nahi", "bilkul nahi", "nahi yar",
]

# Small, curated anchor phrases used ONLY for the yes/no/neutral decision -
# kept deliberately separate from intents.json patterns so this stays fast
# and isn't diluted by unrelated intents.
YES_ANCHORS = [
    "yes", "yes i would like that", "sure that sounds good", "i think so",
    "okay let's try it", "kind of, yes",
]
NO_ANCHORS = [
    "no", "not really", "no I would rather not", "not right now",
    "I don't think so", "no thank you",
]


@st.cache_resource
def _load_context_anchors():
    """Precompute embeddings for the yes/no anchor phrases once per process."""
    load_embedder()
    yes_vecs = np.array([embed_sentence(p) for p in YES_ANCHORS])
    no_vecs = np.array([embed_sentence(p) for p in NO_ANCHORS])
    return yes_vecs, no_vecs


def classify_yes_no(user_input, cleaned_choice):
    """
    Decides whether a reply -- given while a followup context is active --
    means agreement, disagreement, or "topic changed / unclear".

    Returns one of: "yes", "no", "unclear", plus a confidence score.
    """
    # 1. Fast keyword pass (substring match, not exact match, so phrases like
    #    "not really" or "yeah i guess" are caught, not just single words).
    if any(kw in cleaned_choice for kw in DISAGREEMENT_KEYWORDS):
        return "no", 1.0
    if any(kw in cleaned_choice for kw in AGREEMENT_KEYWORDS):
        return "yes", 1.0

    # 2. Semantic fallback against the small anchor sets (not the full
    #    28-intent matcher) so unrelated topics don't dilute the score.
    yes_vecs, no_vecs = _load_context_anchors()
    user_vector = embed_sentence(user_input)

    yes_score = float(np.max(np.dot(yes_vecs, user_vector)))
    no_score = float(np.max(np.dot(no_vecs, user_vector)))

    CONTEXT_THRESHOLD = 0.35  # lower than the global 0.42 since this is a
                              # binary decision, not a 28-way one
    if max(yes_score, no_score) < CONTEXT_THRESHOLD:
        return "unclear", max(yes_score, no_score)

    return ("yes", yes_score) if yes_score >= no_score else ("no", no_score)


# =========================================================
# DETERMINISTIC CRISIS / SELF-HARM SAFETY NET
# =========================================================
# Why this exists:
# Previously, crisis language was only caught by the general semantic
# matcher, AFTER context resolution ran. That meant if the bot was mid-way
# through a followup (e.g. "Would you like to try a grounding exercise?")
# and the user's reply contained both a "no" and something like "I just
# want it to end", the yes/no resolver would grab the "no" and reply with
# ordinary followup text -- completely missing the crisis disclosure.
#
# This check is a deterministic keyword net (not embedding similarity) run
# as the very FIRST thing on every turn, before the context resolver, the
# negation-correction layer, or the general intent matcher. Embedding
# confidence thresholds are useful for everyday conversation, but they are
# not something to gate safety-critical detection behind -- a keyword net
# is slower to adapt to novel phrasing, but it never "almost" catches
# something this important.
CRISIS_KEYWORDS = [
    "kill myself", "killing myself", "suicide", "suicidal",
    "end my life", "ending my life", "ended my life",
    "want to die", "wanna die", "wish i was dead", "wish i were dead",
    "better off dead", "no reason to live", "not want to live",
    "don't want to live", "do not want to live",
    "can't go on", "cant go on", "cannot go on",
    "take my own life", "taking my own life",
    "hurt myself", "hurting myself", "harm myself", "harming myself",
    "self harm", "self-harm", "selfharm",
    "cutting myself", "cut myself",
    "give up on life", "giving up on life",
    "no point in living", "no point living",
    "ending it all", "end it all", "not want to be alive", "don't want to be alive",
    "want it to end", "want this to end", "i want to end it", "make it stop",
    "want my life to end", "life to end", "tired of living", "tired of being alive",
]

CRISIS_RESPONSE = (
    "I'm really glad you told me this, and I want to take it seriously. "
    "You don't have to go through this alone. Please reach out to the Umang "
    "Mental Health Helpline at 0311-7786264 -- it's free, confidential, and "
    "available 24/7. If you are in immediate danger, please call 1122 "
    "(Rescue) right now or get to someone who can be physically with you. "
    "For safety, this chat will pause other topics -- please use one of the "
    "resources above."
)


def detect_crisis_language(cleaned_choice):
    """Deterministic substring check for self-harm / suicide disclosures."""
    return any(kw in cleaned_choice for kw in CRISIS_KEYWORDS)


# =========================================================
# NLTK RULE-BASED NEGATION LAYER (ESCAPE HATCH)
# =========================================================
def user_is_correcting_bot(user_input, active_context):
    """
    Detects if the user is saying something like "I didn't talk about social anxiety"
    to break out of loop traps instantly.
    """
    if not active_context:
        return False

    tokens = [t.lower() for t in tokenize(user_input)]
    negations = {"not", "don't", "dont", "didnt", "didn't", "never", "wrong", "stop", "no", "false", "incorrect"}

    # If the user uses a negation word
    if any(neg in tokens for neg in negations):
        # Extract the current topic from context name (e.g. awaiting_social_anxiety_followup -> ["social", "anxiety"])
        context_topic = active_context.replace("awaiting_", "").replace("_followup", "")
        context_keywords = context_topic.split("_")

        # If they mention any keyword related to the current topic context, they are correcting the bot
        if any(keyword in tokens for keyword in context_keywords):
            return True

    return False


# =========================================================
# STREAMLIT UI & RESOURCES
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
    matcher = SemanticIntentMatcher("data/intents.json")
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
            is_explicit_farewell = any(word in cleaned_choice for word in ["bye", "goodbye", "leave", "going", "allah hafiz", "khuda hafiz", "exit"])
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

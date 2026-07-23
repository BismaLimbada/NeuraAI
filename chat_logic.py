"""
Deterministic conversational logic for Neura AI, kept separate from app.py
so it can be imported and unit-tested (see tests/test_bot.py) without
booting the Streamlit UI, loading the trained model, or downloading the
Universal Sentence Encoder.
"""
import numpy as np

from preprocess import tokenize, stem, embed_sentence, load_embedder

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:  # pragma: no cover - allows import outside Streamlit
    def _cache_resource(func):
        _cache = {}

        def wrapper(*args, **kwargs):
            if "result" not in _cache:
                _cache["result"] = func(*args, **kwargs)
            return _cache["result"]
        return wrapper


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


@_cache_resource
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

        # Compare stemmed forms so e.g. "attacks" still matches a context
        # built from "attack" instead of requiring an exact string match.
        stemmed_tokens = {stem(t) for t in tokens}
        stemmed_context_keywords = {stem(k) for k in context_keywords}

        # If they mention any keyword related to the current topic context, they are correcting the bot
        if stemmed_tokens & stemmed_context_keywords:
            return True

    return False

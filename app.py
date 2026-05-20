import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words

# 1. Page Configuration
st.set_page_config(page_title="MindEase AI", page_icon="🧠", layout="centered")

st.title("🧠 MindEase AI")
st.caption("Empathetic, context-aware mental health awareness support framework.")
st.divider()

# 2. Optimized Resource Loading
@st.cache_resource
def load_bot_resources():
    model = tf.keras.models.load_model('mental_health_model.keras', compile=False)
    with open("data_mappings.json", "r") as f:
        mappings = json.load(f)
    with open("data/intents.json", "r") as f:
        intents = json.load(f)
    return model, mappings["all_words"], mappings["tags"], intents

model, all_words, tags, intents = load_bot_resources()

# 3. Session Memory State Initializations
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MindEase. How are you holding up today?"}
    ]

# This tracking key acts as the conversational short-term memory memory
if "active_context" not in st.session_state:
    st.session_state.active_context = None

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. Contextual Response Dictionary (Follow-up Mapping)
# This maps what to say when a user answers "Yes" or "No" to specific context states
CONTEXT_ANSWERS = {
    "awaiting_sleep_confirmation": {
        "yes": "Thank you for confirming. When sleep becomes irregular, your biological clock gets disrupted, worsening anxiety. WAYS TO FIX IT: Try the 20-minute rule—if you can't sleep after 20 minutes, get out of bed and sit in a dim room reading a book until you feel tired. REASSURANCE: You are not lazy for feeling exhausted tomorrow; your body is simply fighting a tough cycle right now.",
        "no": "I'm glad your schedule isn't completely broken, but struggling to fall or stay asleep is still deeply taxing. WAYS TO FIX IT: Try a 'brain dump' on a physical pad of paper before bed to dump your worries out of your mind. REASSURANCE: Tonight is just one night. Your value isn't tied to how perfectly you sleep."
    },
    "awaiting_panic_confirmation": {
        "yes": "Let's work through this right now. WAYS TO FIX IT: Follow the 5-4-3-2-1 grounding rule. Name 5 things you can see, 4 things you can physically touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. REASSURANCE: This physical surge is temporary. Your body will naturally re-regulate its adrenaline shortly. You are completely safe.",
        "no": "I am so relieved to hear it hasn't escalated to that point. Let's focus on keeping things calm. Try lengthening your exhales to let your nervous system rest."
    },
    "awaiting_burnout_rest": {
        "yes": "It's wonderful that you managed to take a break, but your brain might still be dealing with cognitive overload. WAYS TO FIX IT: Practice absolute disconnection—close your laptop entirely for 30 minutes tonight and engage in a non-academic activity. REASSURANCE: Stepping away is essential maintenance, not slacking off.",
        "no": "Pushing forward without real rest is exactly what locks burnout into place. WAYS TO FIX IT: Implement a strict 25-minute study block followed by a mandatory 5-minute break (Pomodoro Technique). REASSURANCE: Your worth is not determined by your GPA or project velocity. You are allowed to rest."
    }
}

# Context Setters: Maps an intent tag to a memory context state
CONTEXT_SETTERS = {
    "sleep_problems": "awaiting_sleep_confirmation",
    "panic_attack": "awaiting_panic_confirmation",
    "burnout": "awaiting_burnout_rest",
    "academic_pressure": "awaiting_burnout_rest"
}

# 5. Core Processing Pipeline
if user_input := st.chat_input("Type something here..."):
    # Render user chat bubble
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    cleaned_input = user_input.strip().lower()
    reply = None
    
    # --- CONTEXT MEMORY CHECK LAYER ---
    # Check if the bot is currently waiting for a verification answer from a previous question
    if st.session_state.active_context in CONTEXT_ANSWERS:
        current_context = st.session_state.active_context
        
        if cleaned_input in ["yes", "yeah", "yup", "haan", "ji"]:
            reply = CONTEXT_ANSWERS[current_context]["yes"]
            st.session_state.active_context = None # Clear context state after answering
        elif cleaned_input in ["no", "nah", "nope", "nahi"]:
            reply = CONTEXT_ANSWERS[current_context]["no"]
            st.session_state.active_context = None # Clear context state after answering

    # --- STANDARD NLTK + TENSORFLOW LAYER ---
    # Runs if there is no active context tracking or if the input wasn't a simple yes/no response
    if reply is None:
        tokenized = tokenize(user_input)
        bow_vector = bag_of_words(tokenized, all_words)
        input_data = np.array([bow_vector], dtype=np.float32)
        
        # Predict using your TensorFlow neural net brain
        prediction = model(input_data, training=False).numpy()
        highest_idx = np.argmax(prediction[0])
        confidence = prediction[0][highest_idx]
        predicted_tag = tags[highest_idx]
        
        # Terminal diagnostic logger
        print(f"[DEBUG] User Input: '{user_input}' | Active Context State: {st.session_state.active_context}")
        print(f"[DEBUG] Model Classified Tag: '{predicted_tag}' | Match Confidence: {confidence:.2f}")
        
        if confidence < 0.40:
            # Random selection from your custom fallback pool
            reply = random.choice(intents.get("fallback_responses", ["I am listening. Tell me more."]))
        else:
            # Look up standard response from your custom structured dataset
            for intent in intents['intents']:
                if intent['tag'] == predicted_tag:
                    reply = random.choice(intent['responses'])
                    break
            
            # Check if this newly triggered intent requires tracking memory state for the next turn
            if predicted_tag in CONTEXT_SETTERS:
                st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]
                print(f"[DEBUG] Context Memory State Raised To: '{st.session_state.active_context}'")

    # Render assistant chat bubble
    with st.chat_message("assistant"):
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
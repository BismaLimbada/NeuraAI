import random
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from preprocess import tokenize, stem, bag_of_words

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="Neura AI", page_icon="🌸", layout="centered")

# Enforces a clean, minimalist layout and changes the assistant avatar to pastel pink
st.html("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }
        h1 { margin-bottom: 0px; }
        /* Style the native avatar container to be a beautiful pastel pink */
        [data-testid="stChatMessageAvatarAssistant"] {
            background-color: #F7D6DB !important;
            color: #000000 !important;
        }
    </style>
""")

# Main Header Design
st.title("🌸 Neura AI")
st.caption("Your empathetic, context-aware framework for mental health awareness & support.")
st.write("---")

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
        {"role": "assistant", "content": "Hello! I am Neura. How are you holding up today?"}
    ]

if "active_context" not in st.session_state:
    st.session_state.active_context = None

# Sidebar layout for project utilities (keeps main UI hyper-focused)
with st.sidebar:
    st.subheader("Reach Out")
    st.markdown("""
    If things feel overwhelming, remember that you don't have to carry it all alone. 
    
    Consider reaching out to someone you trust—whether it's a close friend, a family member, a mentor, or a professional who can offer a safe space to listen. Sharing a heavy thought is often the first step toward feeling a bit lighter.
    """)
    
    st.divider()
    
    st.subheader("Session Control")
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Neura. How are you holding up today?"}]
        st.session_state.active_context = None
        st.rerun()

# 4. Contextual Response Dictionary (Follow-up Mapping)
CONTEXT_ANSWERS = {
    "awaiting_sleep_confirmation": {
        "yes": "Thank you for confirming. When sleep becomes irregular, your biological clock gets disrupted, worsening anxiety.\n\n**💡 WAYS TO FIX IT:**\nTry the *20-minute rule*—if you can't sleep after 20 minutes, get out of bed and sit in a dim room reading a book until you feel tired.\n\n**🌿 REASSURANCE:**\nYou are not lazy for feeling exhausted tomorrow; your body is simply fighting a tough cycle right now.",
        "no": "I'm glad your schedule isn't completely broken, but struggling to fall or stay asleep is still deeply taxing.\n\n**💡 WAYS TO FIX IT:**\nTry a *'brain dump'* on a physical pad of paper before bed to clear your worries out of your mind.\n\n**🌿 REASSURANCE:**\nTonight is just one night. Your value isn't tied to how perfectly you sleep."
    },
    "awaiting_panic_confirmation": {
        "yes": "Let's work through this right now.\n\n**💡 WAYS TO FIX IT:**\nFollow the **5-4-3-2-1 grounding rule**:\n* 👀 Name **5** things you can see.\n* ✋ Name **4** things you can physically touch.\n* 👂 Name **3** things you can hear.\n* 👃 Name **2** things you can smell.\n* 👅 Name **1** thing you can taste.\n\n**🌿 REASSURANCE:**\nThis physical surge is temporary. Your body will naturally re-regulate its adrenaline shortly. You are completely safe.",
        "no": "I am so relieved to hear it hasn't escalated to that point. Let's focus on keeping things calm. Try lengthening your exhales to let your nervous system rest."
    },
    "awaiting_burnout_rest": {
        "yes": "It's wonderful that you managed to take a break, but your brain might still be dealing with cognitive overload.\n\n**💡 WAYS TO FIX IT:**\nPractice absolute disconnection—close your laptop entirely for 30 minutes tonight and engage in a non-academic activity.\n\n**🌿 REASSURANCE:**\nStepping away is essential maintenance, not slacking off.",
        "no": "Pushing forward without real rest is exactly what locks burnout into place.\n\n**💡 WAYS TO FIX IT:**\nImplement a strict 25-minute study block followed by a mandatory 5-minute break (**Pomodoro Technique**).\n\n**🌿 REASSURANCE:**\nYour worth is not determined by your GPA or project velocity. You are allowed to rest."
    }
}

CONTEXT_SETTERS = {
    "sleep_problems": "awaiting_sleep_confirmation",
    "panic_attack": "awaiting_panic_confirmation",
    "burnout": "awaiting_burnout_rest",
    "academic_pressure": "awaiting_burnout_rest"
}

# 5. Core Processing & Interactive Pipeline

# Always render ALL historic chat elements FIRST before capturing new inputs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture new input using the modern sticky container element
if user_input := st.chat_input("Type something here..."):
    
    # Render user chat bubble instantly and update history state
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    cleaned_input = user_input.strip().lower()
    reply = None
    
    # Display an empathetic animated indicator while the model computes
    with st.spinner("Processing..."):
        
        # --- CONTEXT MEMORY CHECK LAYER ---
        if st.session_state.active_context in CONTEXT_ANSWERS:
            current_context = st.session_state.active_context
            
            if cleaned_input in ["yes", "yeah", "yup", "haan", "ji"]:
                reply = CONTEXT_ANSWERS[current_context]["yes"]
                st.session_state.active_context = None 
            elif cleaned_input in ["no", "nah", "nope", "nahi"]:
                reply = CONTEXT_ANSWERS[current_context]["no"]
                st.session_state.active_context = None

        # --- STANDARD NLTK + TENSORFLOW LAYER ---
        if reply is None:
            tokenized = tokenize(user_input)
            bow_vector = bag_of_words(tokenized, all_words)
            input_data = np.array([bow_vector], dtype=np.float32)
            
            # Machine Learning Inference Block
            prediction = model(input_data, training=False).numpy()
            highest_idx = np.argmax(prediction[0])
            confidence = prediction[0][highest_idx]
            predicted_tag = tags[highest_idx]
            
            # Diagnostics output to runtime terminal console
            print(f"[DEBUG] User Input: '{user_input}' | Active Context State: {st.session_state.active_context}")
            print(f"[DEBUG] Model Classified Tag: '{predicted_tag}' | Match Confidence: {confidence:.2f}")
            
            if confidence < 0.40:
                reply = random.choice(intents.get("fallback_responses", ["I am listening closely. Tell me more about what's going on."]))
            else:
                for intent in intents['intents']:
                    if intent['tag'] == predicted_tag:
                        reply = random.choice(intent['responses'])
                        break
                
                if predicted_tag in CONTEXT_SETTERS:
                    st.session_state.active_context = CONTEXT_SETTERS[predicted_tag]
                    print(f"[DEBUG] Context Memory State Raised To: '{st.session_state.active_context}'")

    # Render Assistant's computed chat bubble and commit to persistent state
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Forces an internal layout cycle refresh to keep UI state clean and input fields ready
    st.rerun()
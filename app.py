import streamlit as st
import pickle
import re
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# LOAD MODEL + TOKENIZER
# =========================================
model = load_model("criticality_model.h5", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================================
# CLEAN TEXT
# =========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================================
# ADVANCED SENTIMENT DETECTION
# =========================================
positive_words = [
    "like", "good", "great", "excellent",
    "amazing", "satisfied", "happy", "love",
    "fast"
]

negative_words = [
    "late", "delay", "damaged", "bad",
    "worst", "broken", "poor", "refund",
    "complaint"
]

negation_phrases = [
    "don't like",
    "do not like",
    "not good",
    "not satisfied",
    "never buy",
    "not happy"
]

def detect_sentiment(text):
    
    text_lower = text.lower()
    polarity = TextBlob(text).sentiment.polarity
    
    # 1️⃣ Check negation phrases
    for phrase in negation_phrases:
        if phrase in text_lower:
            return "Negative", -0.7
    
    # 2️⃣ Keyword detection
    words = text_lower.split()
    
    pos_flag = any(word in words for word in positive_words)
    neg_flag = any(word in words for word in negative_words)
    
    if pos_flag and not neg_flag:
        return "Positive", polarity + 0.5
    
    elif neg_flag and not pos_flag:
        return "Negative", polarity - 0.5
    
    elif neg_flag and pos_flag:
        return "Mixed", polarity
    
    # 3️⃣ Fallback
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity


# =========================================
# UI
# =========================================
st.title("Complaint Criticality Prediction System")

user_input = st.text_area("Enter your complaint:")

if st.button("Predict Criticality"):
    
    if user_input.strip() == "":
        st.warning("Please enter complaint.")
    
    else:
        
        # Clean text
        cleaned = clean_text(user_input)
        
        # Model prediction
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=50)
        
        pred = model.predict(pad)[0][0]
        score = pred * 5
        
        # Sentiment detection
        sentiment, polarity = detect_sentiment(user_input)
        
        # =========================================
        # SCORE ADJUSTMENT
        # =========================================
        if sentiment == "Positive":
            score *= 0.3
        
        elif sentiment == "Neutral":
            score *= 0.7
        
        elif sentiment == "Negative":
            score *= 1.2
        
        # Clamp score
        score = max(0, min(score, 5))
        
        # =========================================
        # SEVERITY LEVEL
        # =========================================
        if score < 2:
            level = "Low"
            color = "green"
        
        elif score < 3.5:
            level = "Medium"
            color = "orange"
        
        else:
            level = "High"
            color = "red"
        
        # =========================================
        # OUTPUT
        # =========================================
        st.subheader(f"Criticality Score: {score:.2f}")
        
        st.markdown(
            f"<h3 style='color:{color};'>Severity Level: {level}</h3>",
            unsafe_allow_html=True
        )
        
        st.write(f"Detected Sentiment: {sentiment}")
        st.write(f"Polarity Score: {polarity:.2f}")
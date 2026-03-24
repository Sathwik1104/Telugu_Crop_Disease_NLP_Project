import streamlit as st
import pickle
import pandas as pd
import re
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googletrans import Translator

# Load model and tools
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

translator = Translator()

# Load datasets
pesticides = pd.read_csv("pesticides.csv")
prices = pd.read_csv("pesticide_prices.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Keyword disease detection (Rule-based NLP)
def keyword_disease_detection(text):
    text = text.lower()

    if "yellow" in text:
        return "Nutrient Deficiency"
    elif "white" in text or "fungus" in text or "powder" in text:
        return "Powdery Mildew"
    elif "hole" in text or "insect" in text:
        return "Insect Attack"
    elif "curl" in text:
        return "Leaf Curl"
    elif "wilt" in text or "droop" in text:
        return "Fusarium Wilt"
    elif "root" in text:
        return "Root Rot"
    elif "brown" in text or "dry" in text:
        return "Leaf Blight"
    elif "spot" in text:
        return "Leaf Spot"
    else:
        return None

st.title("AI Based Telugu Crop Disease Diagnosis & Pesticide Recommendation System")

# Telugu input
telugu_input = st.text_input("వ్యాధి లక్షణాలు తెలుగులో నమోదు చేయండి:")

# Store disease in session
if "disease" not in st.session_state:
    st.session_state.disease = None

if st.button("Predict Disease"):
    translated = translator.translate(telugu_input, src='te', dest='en')
    english_text = translated.text
    st.write("Translated Text:", english_text)

    # Clean text
    english_text = clean_text(english_text)

    # First try keyword detection
    disease = keyword_disease_detection(english_text)

    # If no keyword match, use ML model
    if disease is None:
        seq = tokenizer.texts_to_sequences([english_text])
        padded = pad_sequences(seq, maxlen=10)
        pred = model.predict(padded)
        disease = le.inverse_transform([pred.argmax()])[0]

    st.session_state.disease = disease

# Show results if disease predicted
if st.session_state.disease:
    disease = st.session_state.disease
    st.success(f"Predicted Disease: {disease}")

    row = pesticides[pesticides['disease'] == disease].iloc[0]
    pesticide = row['pesticide']
    low_cost = row['low_cost']
    usage = row['usage']

    st.write("Recommended Pesticide:", pesticide)
    st.write("Low Cost Option:", low_cost)
    st.write("Usage:", usage)

    # Location input
    location = st.text_input("Enter your location for price:")

    if location:
        price_row = prices[
            (prices['location'].str.lower() == location.lower()) &
            (prices['pesticide'] == pesticide)
        ]

        if not price_row.empty:
            price = price_row.iloc[0]['price_rs']
            shop = price_row.iloc[0]['shop']
            st.write("Nearby Shop:", shop)
            st.write("Price (Rs):", price)
        else:
            st.write("Price not available for this location.")
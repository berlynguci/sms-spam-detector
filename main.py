import streamlit as st
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📱 SMS Spam Detector")

user_input = st.text_area("Enter your SMS message:")

if st.button("Check"):
    if user_input:
        input_vector = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vector)[0]
        st.success(f"The message is: **{prediction.upper()}**")
    else:
        st.warning("Please enter a message.")
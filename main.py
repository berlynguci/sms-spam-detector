import streamlit as st
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import json
from streamlit_lottie import st_lottie

# Load model and vectorizer
model = joblib.load("models/spam_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

st.set_page_config(page_title="SMS Spam Detector", layout="centered")
st.markdown("""
    <style>
        .header-title {
            font-size: 2.5rem;
            background: linear-gradient(to right, #2563eb, #9333ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .result-box {
            border-left: 6px solid;
            padding: 1.5rem;
            border-radius: 1rem;
            background-color: #fff;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            margin-top: 1rem;
            color: green;
        }
        .element-container:has(canvas),
        .stLottie {
            background: transparent !important;
            box-shadow: none !important;
        }
    </style>
""", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_stats = load_lottiefile("lottie/stats.json")
lottie_spam = load_lottiefile("lottie/spam.json")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h1 class="header-title"> SMS Spam Detector</h1>', unsafe_allow_html=True)
    st.write("Analyze your SMS message with machine learning to detect spam.")

    user_input = st.text_area("Enter your SMS message:", height=120)

    show_stats_lottie = False
    if st.button("Analyze"):
        if user_input.strip():
            input_vector = vectorizer.transform([user_input.lower()])
            prediction = model.predict(input_vector)[0]
            probabilities = model.predict_proba(input_vector)[0]
            confidence = max(probabilities) * 100

            border_color = "#ef4444" if prediction == "spam" else "#10b981"
            label = "SPAM" if prediction == "spam" else "HAM"

            st.markdown(f"""
                <div class='result-box' style='border-color: {border_color};'>
                    <h3>Prediction: <span style='color: {border_color};'>{label}</span></h3>
                    <p>Confidence: {confidence:.2f}%</p>
                    <small>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("Classification Probability")
            labels = model.classes_
            fig, ax = plt.subplots()
            ax.barh(labels, probabilities * 100, color=['#ef4444', '#10b981'])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Message Classification Confidence")
            st.pyplot(fig)
            show_stats_lottie = True
        else:
            st.warning("Please enter a message.")

with col2:
    st_lottie(
        lottie_spam,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        key="spam",
        height=200,
        width=200,
    )
    if 'show_stats_lottie' in locals() and show_stats_lottie:
        st.write("")  
        st_lottie(
            lottie_stats,
            key="stats_result_col2",
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=200,
            width=200,
        )
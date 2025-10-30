import streamlit as st
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import BytesIO
from dotenv import load_dotenv
import os


# =========================
# 🔧 CONFIGURATION
# =========================
st.set_page_config(page_title="🌾 Crop Advisor AI", layout="wide")
st.title("🌾 Crop Advisor AI — ML + Gemini Agent")


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    st.error("🚨 Please add your GEMINI_API_KEY in .env file.")
    st.stop()


genai.configure(api_key=GEMINI_API_KEY)


# =========================
# 🧠 LOAD MODEL ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


model, scaler, encoder = load_artifacts()
if model is None:
    st.stop()


# ======================================
# INITIALIZE GEMINI CHAT SESSION (with memory)
# ======================================
if "chat_session" not in st.session_state:
    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    st.session_state.chat_session = model_ai.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "You are an agricultural AI assistant. You explain crop predictions, suggest fertilizers, and answer follow-up questions clearly."
                ],
            },
            {
                "role": "model",
                "parts": [
                    "I understand. I am here to help with agricultural advice, crop recommendations, and farming practices."
                ],
            }
        ]
    )


# ======================================
# STREAMLIT UI SETUP
# ======================================


st.title("🌾 Smart Crop Advisor")
st.markdown("### Predict the best crop and chat with an AI agriculture expert!")


st.subheader("🧮 Enter Your Soil and Climate Data")


col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 150, 90)
    P = st.number_input("Phosphorus (P)", 0, 150, 42)
    K = st.number_input("Potassium (K)", 0, 150, 43)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)


# ======================================
# PREDICTION SECTION
# ======================================
if st.button("🚜 Predict Crop"):
    # Create input array with correct feature order: N, P, K, temperature, humidity, ph, rainfall
    input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    crop_prediction_encoded = model.predict(input_scaled)[0]
    crop_prediction = encoder.inverse_transform([crop_prediction_encoded])[0]


    st.session_state["crop_prediction"] = crop_prediction
    st.success(f"🌱 Recommended Crop: **{crop_prediction}**")


    # Ask Gemini for a natural explanation
    ai_prompt = (
        f"The model predicted '{crop_prediction}'. "
        f"Explain why this crop is suitable for N={N}, P={P}, K={K}, temperature={temperature}, "
        f"humidity={humidity}, ph={ph}, rainfall={rainfall}."
    )
    ai_response = st.session_state.chat_session.send_message(ai_prompt)
    st.session_state["ai_explanation"] = ai_response.text


    st.markdown("### 🤖 AI Agent Explanation")
    st.write(ai_response.text)


# ======================================
# CHAT SECTION (ROOT LEVEL)
# ======================================
st.divider()
st.markdown("## 💬 Chat with Your AI Crop Advisor")


# Display previous chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Root-level chat input (must not be inside any container)
user_input = st.chat_input("Ask about crops, fertilizers, or alternatives...")


if user_input:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    # Gemini AI reply (context preserved automatically)
    ai_reply = st.session_state.chat_session.send_message(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply.text})
    with st.chat_message("assistant"):
        st.markdown(ai_reply.text)


# ======================================
# REPORT DOWNLOAD
# ======================================
st.divider()
st.markdown("## 📄 Download Your Report")


if "crop_prediction" in st.session_state:
    report_content = f"""
🌾 Crop Recommendation Report
-----------------------------


📊 Input Data:
Nitrogen (N): {N}
Phosphorus (P): {P}
Potassium (K): {K}
Soil pH: {ph}
Temperature: {temperature}°C
Humidity: {humidity}%
Rainfall: {rainfall} mm


✅ Predicted Crop: {st.session_state['crop_prediction']}


🤖 AI Explanation:
{st.session_state['ai_explanation']}


💬 Chat Summary:
"""


    for msg in st.session_state.chat_history:
        role = "User" if msg["role"] == "user" else "AI"
        report_content += f"\n{role}: {msg['content']}"


    st.download_button(
        label="⬇️ Download Report (.txt)",
        data=report_content,
        file_name="crop_recommendation_report.txt",
        mime="text/plain"
    )
else:
    st.info("Predict a crop first to generate the report.")

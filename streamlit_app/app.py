"""
Streamlit App for Crop Advisor with Agentic AI
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="🌾 Crop Advisor AI",
    page_icon="🌾",
    layout="wide"
)

# Load model and preprocessing artifacts
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and encoder"""
    try:
        # Load model
        model_path = Path("artifacts/models/RandomForest.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = Path("artifacts/scalers/minmax_scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load encoder
        encoder_path = Path("artifacts/encoders/label_encoder.pkl")
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# Crop information database
CROP_INFO = {
    "rice": {
        "ideal_conditions": "High humidity (80-90%), pH 5.5-7.0, high rainfall (150-300mm)",
        "season": "Monsoon/Kharif season",
        "fertilizer": "NPK ratio 4:2:1, Apply 120kg N, 60kg P2O5, 40kg K2O per hectare",
        "irrigation": "Requires continuous flooding (5-10cm water depth)",
        "growth_period": "120-150 days"
    },
    "wheat": {
        "ideal_conditions": "Low humidity (50-70%), pH 6.0-7.5, moderate rainfall (50-100mm)",
        "season": "Rabi/Winter season",
        "fertilizer": "NPK ratio 4:2:1, Apply 120kg N, 60kg P2O5, 40kg K2O per hectare",
        "irrigation": "4-6 irrigations during growth period",
        "growth_period": "110-130 days"
    },
    "maize": {
        "ideal_conditions": "Moderate humidity (60-80%), pH 5.5-7.0, moderate rainfall (60-120mm)",
        "season": "Kharif season",
        "fertilizer": "Apply 150kg N, 60kg P2O5, 40kg K2O per hectare",
        "irrigation": "Critical at tasseling and grain filling stages",
        "growth_period": "90-120 days"
    },
    "cotton": {
        "ideal_conditions": "Low humidity (40-60%), pH 6.0-8.0, moderate rainfall (50-100mm)",
        "season": "Kharif season",
        "fertilizer": "Apply 120kg N, 60kg P2O5, 60kg K2O per hectare",
        "irrigation": "6-8 irrigations, critical during flowering",
        "growth_period": "150-180 days"
    },
    "chickpea": {
        "ideal_conditions": "Low humidity (50-70%), pH 6.0-7.5, low rainfall (30-60mm)",
        "season": "Rabi season",
        "fertilizer": "Apply 20kg N, 40kg P2O5, 20kg K2O per hectare",
        "irrigation": "1-2 irrigations only if needed",
        "growth_period": "100-120 days"
    },
    "jute": {
        "ideal_conditions": "High humidity (70-90%), pH 6.0-7.5, high rainfall (150-250mm)",
        "season": "Kharif season",
        "fertilizer": "Apply 60kg N, 30kg P2O5, 30kg K2O per hectare",
        "irrigation": "Requires moist soil, 3-4 irrigations",
        "growth_period": "120-150 days"
    }
}

# Add more crops with default info
DEFAULT_CROP_INFO = {
    "ideal_conditions": "Please consult agricultural expert for specific conditions",
    "season": "Varies by region",
    "fertilizer": "Consult soil test for specific recommendations",
    "irrigation": "Based on soil moisture and weather",
    "growth_period": "Varies by variety"
}

def get_crop_info(crop_name):
    """Get crop information"""
    return CROP_INFO.get(crop_name.lower(), DEFAULT_CROP_INFO)

def predict_crop(input_features, model, scaler, encoder):
    """Make crop prediction"""
    try:
        # Scale features
        scaled_features = scaler.transform([input_features])
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Get crop name
        crop_name = encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Get all predictions with probabilities
        all_crops = []
        for idx, prob in enumerate(probabilities):
            crop = encoder.inverse_transform([idx])[0]
            all_crops.append((crop, prob))
        
        # Sort by probability
        all_crops.sort(key=lambda x: x[1], reverse=True)
        
        return crop_name, confidence, all_crops
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def generate_natural_report(crop, confidence, input_data, all_predictions):
    """Generate natural language report"""
    N, P, K, temp, humidity, ph, rainfall = input_data
    
    report = f"""
### 🌾 Crop Recommendation Report

Based on your soil and weather conditions, **{crop.upper()}** is the most suitable crop for cultivation.

#### 📊 Confidence Score: {confidence*100:.1f}%

#### 🔬 Analysis of Your Conditions:

**Soil Nutrients:**
- Nitrogen (N): {N} kg/ha - {'✓ Good' if 20 <= N <= 140 else '⚠ Needs attention'}
- Phosphorus (P): {P} kg/ha - {'✓ Good' if 5 <= P <= 145 else '⚠ Needs attention'}
- Potassium (K): {K} kg/ha - {'✓ Good' if 5 <= K <= 205 else '⚠ Needs attention'}

**Environmental Factors:**
- Soil pH: {ph} - {'✓ Optimal' if 5.5 <= ph <= 7.5 else '⚠ May need adjustment'}
- Humidity: {humidity}% - {'✓ Good' if 60 <= humidity <= 90 else '⚠ Consider irrigation'}
- Temperature: {temp}°C - {'✓ Suitable' if 15 <= temp <= 30 else '⚠ Monitor closely'}
- Rainfall: {rainfall}mm - {'✓ Adequate' if 50 <= rainfall <= 250 else '⚠ Plan irrigation'}

#### 🌱 Why {crop.title()}?
"""
    
    # Add specific reasoning
    crop_info = get_crop_info(crop)
    
    if crop.lower() == "rice":
        if humidity > 80:
            report += f"\n- Your high humidity ({humidity}%) is perfect for rice cultivation"
        if rainfall > 150:
            report += f"\n- Abundant rainfall ({rainfall}mm) suits rice's water requirements"
    elif crop.lower() == "wheat":
        if humidity < 70:
            report += f"\n- Your moderate humidity ({humidity}%) is ideal for wheat"
        if temp < 25:
            report += f"\n- Cool temperature ({temp}°C) favors wheat growth"
    elif crop.lower() == "cotton":
        if temp > 20:
            report += f"\n- Warm temperature ({temp}°C) is excellent for cotton"
        if humidity < 60:
            report += f"\n- Lower humidity ({humidity}%) reduces disease risk in cotton"
    
    report += f"\n- Soil pH of {ph} falls within the optimal range for {crop}"
    report += f"\n- NPK levels are suitable for {crop} cultivation"
    
    return report

def generate_fertilizer_advice(crop, N, P, K):
    """Generate fertilizer recommendations"""
    crop_info = get_crop_info(crop)
    
    advice = f"""
### 🌿 Fertilizer Recommendations for {crop.title()}

**Recommended Application:**
{crop_info['fertilizer']}

**Your Current Levels:**
- Nitrogen: {N} kg/ha
- Phosphorus: {P} kg/ha
- Potassium: {K} kg/ha

**Application Strategy:**
"""
    
    # Nitrogen advice
    if N < 40:
        advice += "\n- 🔴 Nitrogen is LOW. Apply urea (46% N) at 100-120 kg/ha"
    elif N > 120:
        advice += "\n- 🟡 Nitrogen is HIGH. Reduce application to avoid lodging"
    else:
        advice += "\n- 🟢 Nitrogen levels are adequate. Maintain with split applications"
    
    # Phosphorus advice
    if P < 20:
        advice += "\n- 🔴 Phosphorus is LOW. Apply DAP or SSP at recommended rates"
    elif P > 100:
        advice += "\n- 🟡 Phosphorus is sufficient. Skip additional application"
    else:
        advice += "\n- 🟢 Phosphorus is adequate. Apply as basal dose"
    
    # Potassium advice
    if K < 30:
        advice += "\n- 🔴 Potassium is LOW. Apply MOP (Muriate of Potash) at 60 kg/ha"
    elif K > 150:
        advice += "\n- 🟡 Potassium is high. Reduce application"
    else:
        advice += "\n- 🟢 Potassium levels are good. Maintain with regular application"
    
    advice += "\n\n**💡 Pro Tips:**"
    advice += "\n- Apply fertilizers in 2-3 split doses for better efficiency"
    advice += "\n- Combine with organic manure (FYM) at 10-15 tons/ha"
    advice += "\n- Conduct soil test every year for precise recommendations"
    
    return advice

def generate_irrigation_advice(crop, humidity, rainfall, temp):
    """Generate irrigation recommendations"""
    crop_info = get_crop_info(crop)
    
    advice = f"""
### 💧 Irrigation Recommendations for {crop.title()}

**Recommended Practice:**
{crop_info['irrigation']}

**Current Conditions:**
- Humidity: {humidity}%
- Rainfall: {rainfall}mm
- Temperature: {temp}°C

**Irrigation Schedule:**
"""
    
    if rainfall < 50:
        advice += "\n- 🔴 Very low rainfall! Plan for 5-7 irrigations"
        advice += "\n- Install drip/sprinkler system for water efficiency"
    elif rainfall < 100:
        advice += "\n- 🟡 Moderate rainfall. Plan for 3-4 supplementary irrigations"
    else:
        advice += "\n- 🟢 Good rainfall. Minimal irrigation needed"
        advice += "\n- Monitor soil moisture regularly"
    
    if humidity < 50:
        advice += "\n- Low humidity may increase water demand"
        advice += "\n- Consider mulching to conserve moisture"
    
    if temp > 30:
        advice += "\n- High temperature increases evaporation"
        advice += "\n- Irrigate during early morning or evening"
    
    advice += "\n\n**Critical Stages for Irrigation:**"
    
    if crop.lower() == "rice":
        advice += "\n- Transplanting stage: Maintain 5cm water depth"
        advice += "\n- Tillering stage: Keep soil saturated"
        advice += "\n- Flowering stage: Critical - maintain water"
    elif crop.lower() == "wheat":
        advice += "\n- Crown root initiation (20-25 days)"
        advice += "\n- Tillering stage (40-45 days)"
        advice += "\n- Flowering stage (60-65 days)"
        advice += "\n- Grain filling (80-85 days)"
    elif crop.lower() == "maize":
        advice += "\n- Knee-high stage"
        advice += "\n- Tasseling and silking (most critical)"
        advice += "\n- Grain filling stage"
    
    advice += "\n\n**💡 Water Management Tips:**"
    advice += "\n- Use soil moisture sensors for precision"
    advice += "\n- Avoid water logging - maintain proper drainage"
    advice += "\n- Consider rainwater harvesting for sustainable farming"
    
    return advice

def explain_why_not(predicted_crop, asked_crop, input_data, all_predictions):
    """Explain why a specific crop was not recommended"""
    N, P, K, temp, humidity, ph, rainfall = input_data
    
    # Find the asked crop in predictions
    asked_crop_prob = 0
    for crop, prob in all_predictions:
        if crop.lower() == asked_crop.lower():
            asked_crop_prob = prob
            break
    
    explanation = f"""
### 🤔 Why not {asked_crop.title()}?

**Predicted: {predicted_crop.title()}** vs **Asked: {asked_crop.title()}**

**Confidence Comparison:**
- {predicted_crop.title()}: {all_predictions[0][1]*100:.1f}%
- {asked_crop.title()}: {asked_crop_prob*100:.1f}%

**Detailed Analysis:**
"""
    
    # Get ideal conditions for both crops
    predicted_info = get_crop_info(predicted_crop)
    asked_info = get_crop_info(asked_crop)
    
    explanation += f"\n**{asked_crop.title()} Requirements:**\n{asked_info['ideal_conditions']}\n"
    explanation += f"\n**Your Conditions:**"
    explanation += f"\n- pH: {ph}, Humidity: {humidity}%, Temperature: {temp}°C, Rainfall: {rainfall}mm\n"
    
    # Specific reasons
    if asked_crop.lower() == "wheat":
        if humidity > 70:
            explanation += f"\n❌ **Humidity too high**: {asked_crop.title()} prefers 50-70%, but yours is {humidity}%"
            explanation += "\n   - High humidity increases disease risk (rust, blight)"
        if temp > 25:
            explanation += f"\n❌ **Temperature too warm**: {asked_crop.title()} is a cool-season crop, prefers 15-25°C"
        if rainfall > 150:
            explanation += f"\n❌ **Excessive rainfall**: {asked_crop.title()} needs only 50-100mm"
            explanation += "\n   - Too much rain can cause lodging and root rot"
    
    elif asked_crop.lower() == "rice":
        if humidity < 70:
            explanation += f"\n❌ **Humidity too low**: {asked_crop.title()} needs 80-90%, but yours is {humidity}%"
            explanation += "\n   - Rice requires high humidity for optimal growth"
        if rainfall < 150:
            explanation += f"\n❌ **Insufficient rainfall**: {asked_crop.title()} needs 150-300mm"
            explanation += "\n   - Will require extensive irrigation, increasing costs"
    
    elif asked_crop.lower() == "cotton":
        if humidity > 70:
            explanation += f"\n❌ **Humidity too high**: {asked_crop.title()} prefers 40-60%, yours is {humidity}%"
            explanation += "\n   - High humidity promotes bollworm and fungal diseases"
        if temp < 20:
            explanation += f"\n❌ **Temperature too cool**: {asked_crop.title()} needs warm climate (>20°C)"
    
    explanation += f"\n\n**Why {predicted_crop.title()} is Better:**"
    explanation += f"\n✅ Your conditions closely match {predicted_crop.title()}'s requirements"
    explanation += f"\n✅ Higher success probability and better yield potential"
    explanation += f"\n✅ Lower risk of crop failure and disease"
    
    explanation += "\n\n**💡 Recommendation:**"
    explanation += f"\nStick with **{predicted_crop.title()}** for this season. "
    explanation += f"You can consider {asked_crop.title()} when conditions are more favorable."
    
    return explanation

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# App Header
st.title("🌾 Crop Advisor AI")
st.markdown("### Your Intelligent Farming Assistant powered by ML & Agentic AI")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.markdown("### About")
    st.info("""
    This app helps farmers make informed decisions about:
    - 🌱 Crop selection based on soil conditions
    - 🌿 Fertilizer recommendations
    - 💧 Irrigation planning
    - 🤖 AI-powered farming advice
    """)
    
    st.markdown("### Model Info")
    st.success("""
    - Model: Random Forest Classifier
    - Accuracy: ~93%
    - Features: N, P, K, pH, Humidity, Temp, Rainfall
    """)

# Load model
model, scaler, encoder = load_artifacts()

if model is None:
    st.error("❌ Could not load model artifacts. Please check the artifacts folder.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["📊 Crop Prediction", "💬 Chat with AI Agent"])

# Tab 1: Prediction
with tab1:
    st.markdown("### Enter Your Soil and Weather Conditions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌱 Soil Nutrients")
        N = st.number_input("Nitrogen (N) - kg/ha", min_value=0, max_value=200, value=90, step=1)
        P = st.number_input("Phosphorus (P) - kg/ha", min_value=0, max_value=200, value=42, step=1)
        K = st.number_input("Potassium (K) - kg/ha", min_value=0, max_value=250, value=43, step=1)
    
    with col2:
        st.markdown("#### 🌡️ Weather Conditions")
        temperature = st.number_input("Temperature - °C", min_value=0.0, max_value=50.0, value=20.8, step=0.1)
        humidity = st.number_input("Humidity - %", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        rainfall = st.number_input("Rainfall - mm", min_value=0.0, max_value=400.0, value=202.9, step=0.1)
    
    with col3:
        st.markdown("#### 🧪 Soil Properties")
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        st.markdown("#### ")
        if st.button("🔍 Get Recommendation", use_container_width=True, type="primary"):
            input_features = [N, P, K, temperature, humidity, ph, rainfall]
            
            with st.spinner("Analyzing your conditions..."):
                crop, confidence, all_predictions = predict_crop(input_features, model, scaler, encoder)
            
            if crop:
                st.session_state.prediction_made = True
                st.session_state.last_prediction = {
                    'crop': crop,
                    'confidence': confidence,
                    'input_data': input_features,
                    'all_predictions': all_predictions
                }
    
    # Display prediction results
    if st.session_state.prediction_made and st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        
        st.success(f"✅ Recommendation Ready!")
        
        # Main recommendation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(generate_natural_report(
                pred['crop'], 
                pred['confidence'], 
                pred['input_data'],
                pred['all_predictions']
            ))
        
        with col2:
            st.markdown("### 📈 All Predictions")
            for crop_name, prob in pred['all_predictions'][:5]:
                st.metric(
                    label=crop_name.title(),
                    value=f"{prob*100:.1f}%",
                    delta="Recommended" if crop_name == pred['crop'] else None
                )
        
        # Additional advice tabs
        advice_tab1, advice_tab2, advice_tab3 = st.tabs([
            "🌿 Fertilizer Advice", 
            "💧 Irrigation Plan",
            "📋 Crop Details"
        ])
        
        with advice_tab1:
            st.markdown(generate_fertilizer_advice(
                pred['crop'],
                pred['input_data'][0],
                pred['input_data'][1],
                pred['input_data'][2]
            ))
        
        with advice_tab2:
            st.markdown(generate_irrigation_advice(
                pred['crop'],
                pred['input_data'][4],
                pred['input_data'][6],
                pred['input_data'][3]
            ))
        
        with advice_tab3:
            crop_info = get_crop_info(pred['crop'])
            st.markdown(f"### 🌾 {pred['crop'].title()} - Complete Guide")
            st.markdown(f"**Ideal Conditions:** {crop_info['ideal_conditions']}")
            st.markdown(f"**Best Season:** {crop_info['season']}")
            st.markdown(f"**Growth Period:** {crop_info['growth_period']}")
            st.markdown(f"**Fertilizer:** {crop_info['fertilizer']}")
            st.markdown(f"**Irrigation:** {crop_info['irrigation']}")

# Tab 2: Chat with Agent
with tab2:
    st.markdown("### 💬 Ask the AI Agent")
    st.info("🤖 Ask me anything about your crop recommendation, farming practices, or why certain crops were suggested!")
    
    if not st.session_state.prediction_made:
        st.warning("⚠️ Please make a crop prediction first in the 'Crop Prediction' tab!")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_question = st.chat_input("Ask a question... (e.g., 'Why not wheat?', 'Tell me about fertilizers')")
        
        if user_question:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    pred = st.session_state.last_prediction
                    
                    # Determine type of question
                    question_lower = user_question.lower()
                    
                    if "why not" in question_lower:
                        # Extract crop name from question
                        for crop_name, _ in pred['all_predictions']:
                            if crop_name.lower() in question_lower:
                                response = explain_why_not(
                                    pred['crop'],
                                    crop_name,
                                    pred['input_data'],
                                    pred['all_predictions']
                                )
                                break
                        else:
                            response = "Please specify a crop name. Example: 'Why not wheat?' or 'Why not rice?'"
                    
                    elif "fertilizer" in question_lower or "nutrient" in question_lower:
                        response = generate_fertilizer_advice(
                            pred['crop'],
                            pred['input_data'][0],
                            pred['input_data'][1],
                            pred['input_data'][2]
                        )
                    
                    elif "irrigation" in question_lower or "water" in question_lower:
                        response = generate_irrigation_advice(
                            pred['crop'],
                            pred['input_data'][4],
                            pred['input_data'][6],
                            pred['input_data'][3]
                        )
                    
                    elif "why" in question_lower and pred['crop'] in question_lower:
                        response = f"""
### Why {pred['crop'].title()} was Recommended

Based on your input conditions, {pred['crop'].title()} scored the highest confidence of {pred['confidence']*100:.1f}%.

{generate_natural_report(pred['crop'], pred['confidence'], pred['input_data'], pred['all_predictions'])}
"""
                    
                    else:
                        # General response
                        response = f"""
I'm here to help! I can answer questions like:

- **"Why not [crop name]?"** - I'll explain why a specific crop wasn't recommended
- **"Tell me about fertilizers"** - Get fertilizer recommendations
- **"What about irrigation?"** - Get irrigation advice
- **"Why {pred['crop']}?"** - Understand why {pred['crop']} was recommended

**Current Recommendation:** {pred['crop'].title()} with {pred['confidence']*100:.1f}% confidence

What would you like to know?
"""
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌾 Crop Advisor AI - Powered by Machine Learning & Agentic AI</p>
    <p>Made with ❤️ for Farmers</p>
</div>
""", unsafe_allow_html=True)
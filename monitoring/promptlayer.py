# import os
# import uuid
# import pickle
# import numpy as np
# import pandas as pd
# import streamlit as st
# from io import BytesIO
# from dotenv import load_dotenv

# # Optional: you still import google.generativeai for local dev parity,
# # but the actual LLM call is routed through PromptLayer's pl.run to Gemini.
# import google.generativeai as genai

# # PromptLayer SDK
# from promptlayer import PromptLayer

# # =========================
# # 🔧 CONFIGURATION
# # =========================
# st.set_page_config(page_title="🌾 Crop Advisor AI", layout="wide")
# st.title("🌾 Crop Advisor AI — ML + Gemini Agent + PromptLayer")

# load_dotenv()
# # Your existing key for Google AI Studio (Gemini API)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # PromptLayer API key
# PROMPTLAYER_API_KEY = os.getenv("PROMPTLAYER_API_KEY")

# if not GEMINI_API_KEY:
#     st.error("🚨 Please add your GEMINI_API_KEY in .env file.")
#     st.stop()

# if not PROMPTLAYER_API_KEY:
#     st.error("🚨 Please add your PROMPTLAYER_API_KEY in .env file.")
#     st.stop()

# # Configure the Google Gemini SDK (not strictly required for PromptLayer pl.run, but harmless)
# genai.configure(api_key=GEMINI_API_KEY)

# # IMPORTANT for PromptLayer -> Gemini:
# # PromptLayer requires provider API keys via env variables (not passed in code),
# # so ensure GOOGLE_API_KEY is present for Gemini calls via pl.run.
# os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

# # If you want to use Vertex AI instead of direct Gemini API, uncomment and set these:
# # os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
# # os.environ["GOOGLE_CLOUD_PROJECT"] = "<your-gcp-project-id>"
# # os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service_account.json"

# # Initialize PromptLayer client (reads PROMPTLAYER_API_KEY from env)
# pl = PromptLayer()

# # Session-scoped ids for analytics
# if "session_id" not in st.session_state:
#     st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"

# # Allow runtime model selection (A/B) and release labeling
# with st.sidebar:
#     st.header("⚙️ Inference Settings")
#     model_choice = st.radio(
#         "Gemini 2.5 model",
#         options=["gemini-2.5-flash", "gemini-2.5-pro"],
#         index=0,
#         help="Choose flash for speed, pro for higher quality."
#     )
#     release_label = st.text_input(
#         "Prompt Release Label",
#         value="prod",
#         help="Use PromptLayer release labels like prod/staging to version and ship safely."
#     )
#     add_scoring = st.checkbox(
#         "Log a sample score to PromptLayer",
#         value=False,
#         help="If enabled, logs a dummy 'helpfulness' score after each run for demo purposes."
#     )

# # =========================
# # 🧠 LOAD MODEL ARTIFACTS
# # =========================
# @st.cache_resource
# def load_artifacts():
#     try:
#         with open("models/model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("models/scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         with open("models/label_encoder.pkl", "rb") as f:
#             encoder = pickle.load(f)
#         return model, scaler, encoder
#     except Exception as e:
#         st.error(f"Error loading model files: {e}")
#         return None, None, None

# model, scaler, encoder = load_artifacts()
# if model is None:
#     st.stop()

# # ======================================
# # PROMPTLAYER HELPER
# # ======================================
# def extract_text_from_pl_response(resp: dict) -> str:
#     """
#     Try to robustly extract the final text from PromptLayer pl.run response.
#     Strategy:
#       1) resp.get('text') or resp.get('output_text') if present
#       2) resp['raw_response'] in Google format: candidates[0].content.parts[0].text
#       3) prompt_blueprint fallback: messages[-1].content[-1].text
#     """
#     # Direct keys
#     for k in ("text", "output_text"):
#         v = resp.get(k)
#         if isinstance(v, str) and v.strip():
#             return v

#     # Raw response (Gemini SDK-like)
#     raw = resp.get("raw_response")
#     if isinstance(raw, dict):
#         cands = raw.get("candidates")
#         if isinstance(cands, list) and len(cands) > 0:
#             content = cands[0].get("content", {})
#             parts = content.get("parts", [])
#             if isinstance(parts, list) and len(parts) > 0 and isinstance(parts[0], dict):
#                 if "text" in parts[0] and isinstance(parts[0]["text"], str):
#                     return parts[0]["text"]

#     # Blueprint fallback
#     try:
#         return resp["prompt_blueprint"]["prompt_template"]["messages"][-1]["content"][-1]["text"]
#     except Exception:
#         return ""

# def run_pl_prompt(prompt_name: str, input_vars: dict, model: str, release: str, tags=None, extra_meta=None):
#     """
#     Execute a PromptLayer-managed prompt on Gemini 2.5.
#     Assumes you've created a PromptLayer template with the given prompt_name and variables.
#     """
#     tags = tags or ["gemini25", "crop-advisor"]
#     metadata = {"session_id": st.session_state.session_id}
#     if extra_meta:
#         # PromptLayer metadata keys/values should be strings.
#         metadata.update({str(k): str(v) for k, v in extra_meta.items()})

#     # Core managed run: binds to template, release, tags, metadata, and overrides provider+model
#     resp = pl.run(
#         prompt_name=prompt_name,
#         input_variables=input_vars,
#         prompt_release_label=release,
#         provider="google",            # Use Google provider for Gemini
#         model=model,                  # e.g., "gemini-2.5-flash" or "gemini-2.5-pro"
#         tags=tags,
#         metadata=metadata,
#     )

#     request_id = resp.get("request_id")
#     text = extract_text_from_pl_response(resp)

#     return request_id, text

# def track_optional_score(pl_request_id: str, score_value: int = 95, score_name: str = "helpfulness"):
#     """
#     Optionally score a run to enable evaluation/analytics in PromptLayer dashboards.
#     """
#     try:
#         pl.track.score(request_id=pl_request_id, score=score_value, name=score_name)
#     except Exception:
#         # Non-fatal if scoring fails (keep UX smooth)
#         pass

# # ======================================
# # STREAMLIT UI SETUP
# # ======================================
# st.title("🌾 Smart Crop Advisor")
# st.markdown("### Predict the best crop and chat with an AI agriculture expert!")

# st.subheader("🧮 Enter Your Soil and Climate Data")

# col1, col2 = st.columns(2)
# with col1:
#     N = st.number_input("Nitrogen (N)", 0, 150, 90)
#     P = st.number_input("Phosphorus (P)", 0, 150, 42)
#     K = st.number_input("Potassium (K)", 0, 150, 43)
#     ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
# with col2:
#     temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
#     humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
#     rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

# # ======================================
# # PREDICTION SECTION
# # ======================================
# if st.button("🚜 Predict Crop"):
#     # Create input array with correct feature order: N, P, K, temperature, humidity, ph, rainfall
#     input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#     # Scale the input
#     input_scaled = scaler.transform(input_array)

#     # Make prediction
#     crop_prediction_encoded = model.predict(input_scaled)[0]
#     crop_prediction = encoder.inverse_transform([crop_prediction_encoded])[0]

#     st.session_state["crop_prediction"] = crop_prediction
#     st.success(f"🌱 Recommended Crop: **{crop_prediction}**")

#     # PromptLayer-managed explanation (requires a PromptLayer template named 'crop-explanation')
#     # Template example (define in PromptLayer):
#     # "Given soil and climate features N={{N}}, P={{P}}, K={{K}}, temperature={{temperature}} °C,
#     # humidity={{humidity}} %, pH={{ph}}, rainfall={{rainfall}} mm, explain succinctly why the
#     # recommended crop '{{crop}}' is suitable and list 2–3 actionable tips and 1 fertilizer plan."
#     pl_id, ai_text = run_pl_prompt(
#         prompt_name="crop-explanation",
#         input_vars={
#             "crop": crop_prediction,
#             "N": N, "P": P, "K": K,
#             "temperature": temperature,
#             "humidity": humidity,
#             "ph": ph,
#             "rainfall": rainfall
#         },
#         model=model_choice,
#         release=release_label,
#         tags=["gemini25", "crop-explain"],
#         extra_meta={"event": "predict", "user_id": "anon"}  # sample metadata
#     )

#     if add_scoring and pl_id:
#         track_optional_score(pl_id, score_value=95, score_name="helpfulness")

#     st.session_state["ai_explanation"] = ai_text

#     st.markdown("### 🤖 AI Agent Explanation")
#     st.write(ai_text)

# # ======================================
# # CHAT SECTION (ROOT LEVEL)
# # ======================================
# st.divider()
# st.markdown("## 💬 Chat with Your AI Crop Advisor")

# # Display previous chat messages
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Root-level chat input (must not be inside any container)
# user_input = st.chat_input("Ask about crops, fertilizers, or alternatives...")

# if user_input:
#     # Display user message
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     # Flatten history for the template variable
#     history_text = "\n".join([
#         f"{'User' if m['role']=='user' else 'AI'}: {m['content']}"
#         for m in st.session_state.chat_history
#         if m["content"].strip()
#     ])

#     # PromptLayer-managed chat turn (requires a PromptLayer template named 'crop-chat')
#     # Template example:
#     # "System: {{system}}\nConversation so far:\n{{history}}\nUser: {{user_input}}\n
#     # Assistant: Provide precise, farmer-friendly guidance with concrete steps and local best practices."
#     pl_id, chat_text = run_pl_prompt(
#         prompt_name="crop-chat",
#         input_vars={
#             "system": "You are an agricultural AI assistant. You explain crop predictions, suggest fertilizers, and answer follow-up questions clearly.",
#             "history": history_text,
#             "user_input": user_input
#         },
#         model=model_choice,
#         release=release_label,
#         tags=["gemini25", "chat"],
#         extra_meta={"event": "chat", "user_id": "anon"}
#     )

#     if add_scoring and pl_id:
#         track_optional_score(pl_id, score_value=92, score_name="helpfulness")

#     st.session_state.chat_history.append({"role": "assistant", "content": chat_text})
#     with st.chat_message("assistant"):
#         st.markdown(chat_text)

# # ======================================
# # REPORT DOWNLOAD
# # ======================================
# st.divider()
# st.markdown("## 📄 Download Your Report")

# if "crop_prediction" in st.session_state:
#     # Use the latest UI inputs as they reflect the on-screen state
#     report_content = f"""
# 🌾 Crop Recommendation Report
# -----------------------------

# 📊 Input Data:
# Nitrogen (N): {N}
# Phosphorus (P): {P}
# Potassium (K): {K}
# Soil pH: {ph}
# Temperature: {temperature}°C
# Humidity: {humidity}%
# Rainfall: {rainfall} mm

# ✅ Predicted Crop: {st.session_state['crop_prediction']}

# 🤖 AI Explanation:
# {st.session_state.get('ai_explanation', '')}

# 💬 Chat Summary:
# """

#     for msg in st.session_state.chat_history:
#         role = "User" if msg["role"] == "user" else "AI"
#         report_content += f"\n{role}: {msg['content']}"

#     st.download_button(
#         label="⬇ Download Report (.txt)",
#         data=report_content,
#         file_name="crop_recommendation_report.txt",
#         mime="text/plain"
#     )
# else:
#     st.info("Predict a crop first to generate the report.")
# from promptlayer import PromptLayer
# pl = PromptLayer()
# print("PromptLayer client OK")
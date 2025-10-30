from openai import OpenAI
import opik
import os
from opik.evaluation import evaluate_prompt
from opik.evaluation.metrics import  Hallucination
# import google.generativeai as genai
# from opik.integrations.genai import track_genai

from dotenv import load_dotenv
load_dotenv() # <-- THIS MUST BE AT THE TOP!

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 🧩 Original system prompt
BASE_SYSTEM_PROMPT = """
You are an agricultural AI assistant.
You explain crop predictions, suggest fertilizers, and answer follow-up questions clearly.
"""

# 🌾 Enhanced System v2
ENHANCED_SYSTEM_PROMPT = """
You are 'Crop Advisor AI', an expert agronomist assistant. Your primary role is to interpret a machine learning model's crop prediction for a user.

Your goals are:
1. **Explain Clearly:** When a prediction is provided, first explain WHY that crop is suitable, directly linking your reasoning to the user's data (N, P, K, pH, temp, etc.).
2. **Give Actionable Advice:** ALWAYS provide a "Next Steps" section. This must include:
   * Specific fertilizer recommendations (e.g., "a nitrogen-rich fertilizer like Urea," "a balanced 10-10-10 NPK mix," or "organic compost").
   * One or two critical tips for that crop (e.g., watering needs, soil preparation).
3. **Be a Helpful Assistant:** Answer all follow-up questions clearly. If you don't know, say so.
4. **Formatting:** Use markdown (bolding, bullet points) to make your answers easy to read and scannable.
"""


opik_client = opik.Opik()

dataset = opik_client.get_or_create_dataset("Crop_Advisor_Prompt_Eval")


dataset.insert([
    {
        "input": {
            "N": 90, "P": 42, "K": 43,
            "temperature": 25, "humidity": 85, "ph": 6.5, "rainfall": 200
        },
        "expected_output": "Rice is suitable due to high humidity and nitrogen-rich soil."
    },
    {
        "input": {
            "N": 40, "P": 50, "K": 60,
            "temperature": 32, "humidity": 45, "ph": 7.2, "rainfall": 80
        },
        "expected_output": "Maize fits well because it tolerates moderate rainfall and high temperature."
    }
])


# def generate_ai_prompt(crop_prediction, N, P, K, temperature, humidity, ph, rainfall):
#     return (
#         f"The machine learning model predicted **{crop_prediction}** as the best crop.\n\n"
#         f"**User's Data:**\n"
#         f"* N (Nitrogen): {N}\n"
#         f"* P (Phosphorus): {P}\n"
#         f"* K (Potassium): {K}\n"
#         f"* Soil pH: {ph}\n"
#         f"* Temperature: {temperature}°C\n"
#         f"* Humidity: {humidity}%\n"
#         f"* Rainfall: {rainfall} mm\n\n"
#         f"Please provide your expert analysis based on our instructions (explain why, recommend fertilizer, and give next steps)."
#     )


# def get_crop_advice(system_prompt, ai_prompt, user_input, model_name="gemini-1.5-flash"):
#     """
#     Logs and monitors all 3 prompt layers (system, auto explanation, and user chat)
#     using Opik tracing.
#     """
#     model_ai = model_name

#     chat = model_ai.start_chat(
#         history=[{"role": "system", "parts": [system_prompt]}]
#     )

#     # Step 1: Automated explanation
#     explanation = chat.send_message(ai_prompt)

#     # Step 2: Handle user input (optional)
#     user_reply = None
#     if user_input:
#         user_reply = chat.send_message(user_input)

#     return {
#         "explanation": explanation.text,
#         "user_reply": user_reply.text if user_reply else None,
#     }


# ======================================================
# 5️⃣ Evaluate Prompts Programmatically
# ======================================================

# 🧪 Evaluate BASE prompt
evaluate_prompt(
    dataset=dataset,
    messages=[
        {
            "role": "system",
            "content": BASE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Explain why this crop is suitable for N={{N}}, P={{P}}, K={{K}}, "
                "temperature={{temperature}}, humidity={{humidity}}, ph={{ph}}, rainfall={{rainfall}}."
            ),
        },
    ],
    model="gpt-3.5-turbo",
    scoring_metrics=[Hallucination()],
)

# 🧠 Evaluate ENHANCED prompt
evaluate_prompt(
    dataset=dataset,
    messages=[
        {
            "role": "system",
            "content": ENHANCED_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Explain why this crop is suitable for N={{N}}, P={{P}}, K={{K}}, "
                "temperature={{temperature}}, humidity={{humidity}}, ph={{ph}}, rainfall={{rainfall}}."
            ),
        },
    ],
    model="gpt-3.5-turbo",
    scoring_metrics=[Hallucination()],
)


# ======================================================
# 6️⃣ Example: Trace a Real Chat Session
# ======================================================

if __name__ == "__main__":
    COMET_API_KEY = os.environ["COMET_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    crop_prediction = "Rice"

    # ai_prompt = generate_ai_prompt(
    #     crop_prediction=crop_prediction,
    #     N=90, P=42, K=43, temperature=25, humidity=85, ph=6.5, rainfall=200
    # )

    user_input = "Can you suggest any organic fertilizers for this crop?"

    # result = get_crop_advice(
    #     system_prompt=ENHANCED_SYSTEM_PROMPT,
    #     ai_prompt=ai_prompt,
    #     user_input=user_input,
    # )

    # print("=== 🌾 AI Explanation ===")
    # print(result["explanation"])
    # print("\n=== 💬 User Reply ===")
    # print(result["user_reply"])
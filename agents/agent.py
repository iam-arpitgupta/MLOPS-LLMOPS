import pickle
import numpy as np
import google.generativeai as genai
import os
import yaml  # You'll need to install this: pip install pyyaml
from dotenv import load_dotenv

class CropAdvisorAgent:
    """
    Encapsulates all logic for the Crop Advisor AI,
    independent of any UI framework.
    """
    def __init__(self, prompt_file="prompts.yaml"):
        print("Initializing agent...")
        self.model = None
        self.scaler = None
        self.encoder = None
        self.prompts = None
        self.chat_session = None

        # State variables
        self.chat_history = []
        self.last_inputs = None
        self.last_prediction = None
        self.last_explanation = None

        self.load_prompts(prompt_file)
        self.load_artifacts()
        self.initialize_chat()
        print("Agent ready.")

    def load_prompts(self, prompt_file):
        """Loads prompt templates from a YAML file."""
        try:
            with open(prompt_file, 'r') as f:
                self.prompts = yaml.safe_load(f)
        except Exception as e:
            print(f"🚨 Error loading prompts from {prompt_file}: {e}")
            raise

    def load_artifacts(self):
        """Loads the pickled ML model, scaler, and encoder."""
        try:
            with open("models/model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open("models/label_encoder.pkl", "rb") as f:
                self.encoder = pickle.load(f)
        except Exception as e:
            print(f"🚨 Error loading model files: {e}")
            raise

    def initialize_chat(self):
        """Configures and starts a new Gemini chat session."""
        try:
            # Using 1.5-flash as recommended for better instruction following
            model_ai = genai.GenerativeModel("gemini-1.5-flash")
            
            # Use the loaded prompts for history
            system_prompt = self.prompts['system_prompt']['template']
            model_response = self.prompts['system_response']['template']

            self.chat_session = model_ai.start_chat(
                history=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": [model_response]}
                ]
            )
            # Add the system prompts to our internal history for report generation
            self.chat_history.append({"role": "user", "content": system_prompt})
            self.chat_history.append({"role": "assistant", "content": model_response})

        except Exception as e:
            print(f"🚨 Error initializing Gemini chat: {e}")
            raise

    def get_prediction(self, inputs_dict):
        """
        Generates a crop prediction from the ML model.
        :param inputs_dict: A dictionary of the input features.
        :return: The predicted crop name as a string.
        """
        # Store inputs for later
        self.last_inputs = inputs_dict
        
        # Create input array in the correct feature order
        input_array = np.array([[
            inputs_dict['N'],
            inputs_dict['P'],
            inputs_dict['K'],
            inputs_dict['temperature'],
            inputs_dict['humidity'],
            inputs_dict['ph'],
            inputs_dict['rainfall']
        ]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_array)
        
        # Make prediction
        crop_prediction_encoded = self.model.predict(input_scaled)[0]
        crop_prediction = self.encoder.inverse_transform([crop_prediction_encoded])[0]
        
        self.last_prediction = crop_prediction
        return crop_prediction

    def get_ai_explanation(self):
        """
        Gets a natural language explanation for the last prediction
        from the Gemini model.
        :return: The AI's explanation as a string.
        """
        if not self.last_prediction or not self.last_inputs:
            return "Please run a prediction first."

        # Format the prompt from the YAML file
        template = self.prompts['prediction_prompt']['template']
        ai_prompt = template.format(
            crop_prediction=self.last_prediction,
            **self.last_inputs
        )
        
        ai_response = self.chat_session.send_message(ai_prompt)
        self.last_explanation = ai_response.text

        # Add this interaction to history
        self.chat_history.append({"role": "user", "content": ai_prompt})
        self.chat_history.append({"role": "assistant", "content": self.last_explanation})
        
        return self.last_explanation

    def handle_chat(self, user_input):
        """
        Sends a user's follow-up message to Gemini and returns the response.
        :param user_input: The user's text query.
        :return: The AI's response as a string.
        """
        self.chat_history.append({"role": "user", "content": user_input})
        
        ai_reply = self.chat_session.send_message(user_input)
        
        self.chat_history.append({"role": "assistant", "content": ai_reply.text})
        return ai_reply.text

    def generate_report(self):
        """
        Generates a text report of the full session.
        :return: A formatted report string.
        """
        if not self.last_prediction:
            return "No prediction has been made yet. Please run a prediction first."

        report_content = f"""
🌾 Crop Recommendation Report
-----------------------------

📊 Input Data:
Nitrogen (N): {self.last_inputs['N']}
Phosphorus (P): {self.last_inputs['P']}
Potassium (K): {self.last_inputs['K']}
Soil pH: {self.last_inputs['ph']}
Temperature: {self.last_inputs['temperature']}°C
Humidity: {self.last_inputs['humidity']}%
Rainfall: {self.last_inputs['rainfall']} mm

✅ Predicted Crop: {self.last_prediction}

🤖 AI Explanation:
{self.last_explanation}

💬 Full Chat Summary:
---------------------
"""
        # Iterate over chat history, skipping the long system prompts
        for msg in self.chat_history[2:]: # Skip the first two system prompts
            role = "User" if msg["role"] == "user" else "AI"
            
            # Don't re-print the automatic prediction prompt
            if msg["content"].startswith("The machine learning model predicted"):
                continue
                
            report_content += f"\n{role}: {msg['content']}\n"

        return report_content

def get_numeric_input(prompt_text, default_value):
    """Helper function to get validated numeric input from the user."""
    while True:
        val = input(f"{prompt_text} (default: {default_value}): ")
        if val == "":
            return default_value
        try:
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a number.")

def main_cli():
    """
    Main function to run the agent in a command-line interface.
    """
    print("🌾 Welcome to the Crop Advisor AI (CLI)")
    print("---------------------------------------")
    
    try:
        agent = CropAdvisorAgent()
    except Exception as e:
        print(f"Failed to initialize agent. Exiting. Error: {e}")
        return

    print("\n🧮 Enter Your Soil and Climate Data")
    
    inputs = {
        'N': get_numeric_input("Nitrogen (N)", 90),
        'P': get_numeric_input("Phosphorus (P)", 42),
        'K': get_numeric_input("Potassium (K)", 43),
        'ph': get_numeric_input("Soil pH", 6.5),
        'temperature': get_numeric_input("Temperature (°C)", 25.0),
        'humidity': get_numeric_input("Humidity (%)", 80.0),
        'rainfall': get_numeric_input("Rainfall (mm)", 200.0)
    }

    # --- Prediction Section ---
    print("\n🚜 Predicting crop...")
    prediction = agent.get_prediction(inputs)
    print(f"🌱 Recommended Crop: **{prediction}**")

    print("\n🤖 Asking AI for explanation...")
    explanation = agent.get_ai_explanation()
    print("\n### 🤖 AI Agent Explanation ###")
    print(explanation)
    print("-----------------------------")

    # --- Chat Section ---
    print("\n## 💬 Chat with Your AI Crop Advisor ##")
    print("(Type 'report' to save a report, or 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'report':
            print("\n📄 Generating report...")
            report = agent.generate_report()
            with open("crop_recommendation_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            print("Report saved to 'crop_recommendation_report.txt'")
            continue

        ai_reply = agent.handle_chat(user_input)
        print(f"\n🤖 AI: {ai_reply}")

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    # --- Load API Key ---
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("🚨 Please add your GEMINI_API_KEY in .env file.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        main_cli()

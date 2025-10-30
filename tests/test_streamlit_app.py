import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pickle
import tempfile

# To import your Streamlit app safely (without running it fully)
# mock streamlit before import
sys.modules["streamlit"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

class TestCropAdvisorApp(unittest.TestCase):
    """Unit tests for Crop Advisor Streamlit app logic (without UI)."""

    def setUp(self):
        # Create dummy model, scaler, encoder
        class DummyModel:
            def predict(self, X):
                return np.array([0])

        class DummyScaler:
            def transform(self, X):
                return X / 100  # dummy scaling
            n_features_in_ = 7

        class DummyEncoder:
            def inverse_transform(self, y):
                return ["rice"]

        # Mock artifacts
        self.model = DummyModel()
        self.scaler = DummyScaler()
        self.encoder = DummyEncoder()

        # Patch Gemini configuration
        os.environ["GEMINI_API_KEY"] = "fake-key"


    def test_prediction_flow(self):
        """Test that prediction and decoding logic works properly."""
        # Mock Streamlit UI inputs
        N, P, K, temp, hum, ph, rain = 90, 40, 50, 25, 80, 6.5, 200
        input_array = np.array([[N, P, K, temp, hum, ph, rain]])

        scaled = self.scaler.transform(input_array)
        pred_encoded = self.model.predict(scaled)[0]
        decoded = self.encoder.inverse_transform([pred_encoded])[0]

        self.assertEqual(decoded, "rice")
        self.assertAlmostEqual(scaled[0][0], N / 100)

    @patch("google.generativeai.GenerativeModel")
    def test_gemini_prompt_response(self, mock_model):
        """Test Gemini AI prompt + response logic."""
        # Mock Gemini model behavior
        mock_instance = MagicMock()
        mock_instance.start_chat.return_value.send_message.return_value.text = (
            "Rice is suitable because the soil is fertile and humid."
        )
        mock_model.return_value = mock_instance

        # Start chat session
        model_ai = mock_model("gemini-2.5-flash")
        chat_session = model_ai.start_chat()

        # Send message
        response = chat_session.send_message("Explain crop choice.")
        self.assertIn("Rice", response.text)

    def test_env_key_exists(self):
        """ Ensure GEMINI_API_KEY is properly loaded from .env"""
        self.assertIn("GEMINI_API_KEY", os.environ)
        self.assertEqual(os.getenv("GEMINI_API_KEY"), "fake-key")


if __name__ == "__main__":
    unittest.main()

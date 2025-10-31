from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os 
import pickle 

# --- Load Model ---
# (Assuming 'model.pkl' and the CSV file are in the correct paths)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'model.pkl' not found.")
    model = None  # Set model to None to avoid app crashing on load

try:
    DATA_PATH = "notebooks[experimentation]/Crop_recommendation.csv"
    full_data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    full_data = pd.DataFrame() # Create empty dataframe


app = FastAPI(
    title="Crop Recommendation API",
    description="An API to predict crop recommendations based on soil and weather data."
)

# --- Pydantic Models ---

class CropFeatures(BaseModel):
    """Pydantic model for features required for prediction."""
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropData(CropFeatures):
    """Pydantic model representing a full data record (features + label)."""
    label: str


# --- Helper Functions ---

def load_data():
    """Load the data from the data path."""
    # Reads the pre-loaded dataframe
    return full_data


# Note: 'save_data' function was defined but never used in your endpoints.
# You can add an endpoint to call it if needed.


# --- API Endpoints ---

@app.get("/")
def hello():
    return {"message": "Welcome to the Crop Recommendation API"}


@app.get('/about')
def about():
    # Corrected the message to match the app
    return {'message': 'A fully functional API for crop recommendations.'}


@app.get('/view')
def view_all_data():
    """Returns the entire dataset as JSON."""
    # Note: Returning an entire large CSV can be slow.
    data = load_data()
    return data.to_dict(orient="records")


@app.post('/predict')
def predict(data: CropFeatures):
    """
    Predict the best crop to grow based on input features.
    """
    if model is None:
        return {"error": "Model is not loaded. Cannot make prediction."}

    # 1. Create DataFrame from the 7 input features
    #    (No 'label' is included here)
    input_df = pd.DataFrame([{
            'N': data.N,
            'P': data.P,
            'K': data.K,
            'temperature': data.temperature,
            'humidity': data.humidity,
            'ph': data.ph,
            'rainfall': data.rainfall
    }])

    # 2. Make prediction
    #    (Assuming your model was trained on columns in this order)
    try:
        prediction = model.predict(input_df)[0]
        
        # 3. Return a clean JSON response
        return {"predicted_crop": prediction}
        
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}
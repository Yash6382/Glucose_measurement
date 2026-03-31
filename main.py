import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = FastAPI(title="GlucoSense API", description="Non-invasive glucose prediction")

# Enable CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model & Scaler ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# --- Data Models ---
class PredictRequest(BaseModel):
    s11_freq_2ghz: float
    s11_db_2ghz: float
    s11_freq_5ghz: float
    s11_db_5ghz: float

class PredictResponse(BaseModel):
    glucose_mgdl: float
    level: str
    confidence_note: str

# --- Logic ---
def classify(value: float):
    if value < 70:
        return "Low", "Blood glucose is below 70 mg/dL. Consider consuming fast-acting carbohydrates."
    elif 70 <= value <= 99:
        return "Normal", "Blood glucose is within the healthy fasting range (70-99 mg/dL)."
    elif 100 <= value <= 125:
        return "Pre-diabetic", "Blood glucose is elevated (100-125 mg/dL). Consult a healthcare provider."
    else:
        return "Diabetic", "Blood glucose is in the diabetic range (126+ mg/dL). Seek medical advice."

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not initialized on server.")
    
    try:
        # Your scaler expects 10 features. We provide the 4 from the app 
        # and pad the remaining 6 with 0.0 to satisfy the matrix dimensions.
        input_list = [
            payload.s11_freq_2ghz,
            payload.s11_db_2ghz,
            payload.s11_freq_5ghz,
            payload.s11_db_5ghz,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding for the 6 missing columns
        ]
        
        # Convert to numpy array and reshape for the scaler
        features = np.array([input_list])
        
        # Apply Scaling
        features_scaled = scaler.transform(features)
        
        # Run Prediction
        # Note: If the model was trained on 10 features, it will receive the 10 scaled values.
        prediction = float(model.predict(features_scaled)[0])
        
        # Clean up output
        prediction = max(0.0, round(prediction, 1))
        level, note = classify(prediction)
        
        return PredictResponse(
            glucose_mgdl=prediction,
            level=level,
            confidence_note=note
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

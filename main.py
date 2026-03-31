"""
GlucoSense — FastAPI backend
Deploy on Render: https://render.com

Files required in the same directory:
  - best_model.pkl
  - scaler.pkl
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── Load model & scaler once at startup ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file not found: {e}. "
        "Make sure best_model.pkl and scaler.pkl are in the same folder as main.py."
    )

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GlucoSense API",
    description="Non-invasive blood glucose prediction from antenna S-parameter readings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    s11_freq_2ghz: float = Field(..., description="S11 resonant frequency ~2.2 GHz band (GHz)", example=2.19)
    s11_db_2ghz:   float = Field(..., description="S11 magnitude ~2.2 GHz band (dB, negative)", example=-6.75)
    s11_freq_5ghz: float = Field(..., description="S11 resonant frequency ~5.5 GHz band (GHz)", example=5.188)
    s11_db_5ghz:   float = Field(..., description="S11 magnitude ~5.5 GHz band (dB, negative)", example=-13.57)

class PredictResponse(BaseModel):
    glucose_mgdl:    float
    level:           str
    confidence_note: str

# ── Classifier ────────────────────────────────────────────────────────────────
def classify(glucose: float) -> tuple[str, str]:
    if glucose < 70:
        return ("Low",
                "Blood glucose is below 70 mg/dL (hypoglycemia). Consume fast-acting carbohydrates immediately.")
    elif glucose <= 99:
        return ("Normal",
                "Blood glucose is within the normal fasting range (70–99 mg/dL). No immediate action needed.")
    elif glucose <= 125:
        return ("Pre-diabetic",
                "Blood glucose is in the pre-diabetic range (100–125 mg/dL). Consult a healthcare provider.")
    else:
        return ("Diabetic",
                "Blood glucose is in the diabetic range (≥126 mg/dL). Seek medical advice promptly.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    """
    Health check — polled by the frontend every 2.5 s to detect when Render
    wakes up from sleep. Returns immediately once model is loaded.
    """
    return {"status": "ok", "message": "GlucoSense API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Predict blood glucose from four antenna S-parameter features.
    Column order matches training data:
      [S11_freq_2_2_4_GHz, S11_dB_2_2_4, S11_freq_5_5_3_GHz, S11_dB_5_5_3]
    """
    try:
        features = np.array([[
            payload.s11_freq_2ghz,
            payload.s11_db_2ghz,
            payload.s11_freq_5ghz,
            payload.s11_db_5ghz,
        ]])
        features_scaled = scaler.transform(features)
        prediction      = float(model.predict(features_scaled)[0])
        prediction      = max(0.0, round(prediction, 1))
        level, note     = classify(prediction)
        return PredictResponse(glucose_mgdl=prediction, level=level, confidence_note=note)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

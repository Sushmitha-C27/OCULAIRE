# /content/OCULAIRE/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="OCULAIRE — AI Glaucoma Detection API",
    description="Test version — verify docs route works",
    version="1.0.0"
)

class RNFLTInput(BaseModel):
    mean_rnflt: float
    std_rnflt: float
    min_rnflt: float
    max_rnflt: float

@app.get("/")
def root():
    return {"message": "Welcome to OCULAIRE API (Docs test)"}

@app.post("/predict")
def predict_rnflt(data: RNFLTInput):
    return {
        "prediction": "Healthy-like" if data.mean_rnflt > 80 else "Glaucoma-like",
        "confidence": 0.95
    }

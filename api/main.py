# api/main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from xgboost import XGBClassifier

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_PATH = "ml/models/xgboost_virality.json"

app = FastAPI(
    title="HackerNews Virality Predictor",
    description="Predicts whether a HN story will go viral (score >= 100)",
    version="1.0.0"
)

model = XGBClassifier()
model.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")


# ── Request schema ─────────────────────────────────────────────────────────────
class StoryFeatures(BaseModel):
    hour_of_day:      int   # 0-23, what hour was it posted
    day_of_week:      int   # 1=Sun, 7=Sat
    is_weekend:       int   # 0 or 1
    title_length:     int   # number of characters in title
    title_word_count: int   # number of words in title
    has_url:          int   # 0 or 1
    is_show_hn:       int   # 0 or 1
    is_ask_hn:        int   # 0 or 1

    class Config:
        json_schema_extra = {
            "example": {
                "hour_of_day":      14,
                "day_of_week":      3,
                "is_weekend":       0,
                "title_length":     52,
                "title_word_count": 8,
                "has_url":          1,
                "is_show_hn":       1,
                "is_ask_hn":        0
            }
        }


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    is_viral:         int
    viral_probability: float
    verdict:          str
    features_used:    dict


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "HN Virality Predictor",
        "status":  "running",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET /health",
            "docs":    "GET /docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(story: StoryFeatures):
    """
    Given story features, predict whether it will go viral.
    Returns probability and a human-readable verdict.
    """
    try:
        features = np.array([[
            story.hour_of_day,
            story.day_of_week,
            story.is_weekend,
            story.title_length,
            story.title_word_count,
            story.has_url,
            story.is_show_hn,
            story.is_ask_hn,
        ]])

        prediction   = int(model.predict(features)[0])
        probability  = float(model.predict_proba(features)[0][1])

        if probability >= 0.7:
            verdict = "Very likely to go viral!"
        elif probability >= 0.5:
            verdict = "Has a good chance of going viral"
        elif probability >= 0.3:
            verdict = "Unlikely to go viral"
        else:
            verdict = "Very unlikely to go viral"

        return PredictionResponse(
            is_viral          = prediction,
            viral_probability = round(probability, 4),
            verdict           = verdict,
            features_used     = story.dict()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

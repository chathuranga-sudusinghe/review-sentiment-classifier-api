from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.predictor import predictor
from app.schemas import PredictionRequest, PredictionResponse

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title="Review Sentiment Classifier API",
    description="A simple FastAPI service for predicting review sentiment.",
    version="1.0.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """
    Root endpoint.
    """
    return {"message": "Review Sentiment Classifier API is running."}


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest) -> PredictionResponse: # prediction endpoint that takes a review text and returns the predicted sentiment and confidence score
    """
    Predict sentiment for a single review text.
    """
    try:
        result = predictor.predict(request.text)

        return PredictionResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probability=result["probability"],
        )

    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
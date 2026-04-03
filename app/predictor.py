from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"
LABEL_CONFIG_PATH = ARTIFACTS_DIR / "label_config.json"


def clean_text(text: str) -> str:
    """
    Clean input review text for inference.
    This should match the same cleaning logic used in training.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class SentimentPredictor:
    """
    Load artifacts once and serve predictions for review sentiment.
    """

    def __init__(self) -> None:
        self.model: keras.Model | None = None
        self.tokenizer: Tokenizer | None = None
        self.label_config: dict[str, Any] | None = None
        self.max_len: int | None = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load model, tokenizer, and label configuration from disk.
        """
        missing_files = [
            path for path in [MODEL_PATH, TOKENIZER_PATH, LABEL_CONFIG_PATH]
            if not path.exists()
        ]
        if missing_files:
            missing_str = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                f"Required artifact files are missing: {missing_str}"
            )

        logger.info("Loading sentiment model artifacts...")

        self.model = keras.models.load_model(MODEL_PATH)

        with TOKENIZER_PATH.open("rb") as file:
            self.tokenizer = pickle.load(file)

        with LABEL_CONFIG_PATH.open("r", encoding="utf-8") as file:
            self.label_config = json.load(file)

        self.max_len = int(self.label_config["max_len"])

        logger.info("Artifacts loaded successfully.")

    def _prepare_text(self, text: str) -> np.ndarray:
        """
        Convert raw input text into padded numeric sequence.
        """
        if self.tokenizer is None or self.max_len is None:
            raise ValueError("Tokenizer or max_len is not loaded properly.")

        cleaned_text = clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.max_len,
            padding="post",
            truncating="post",
        )
        return padded_sequence

    def predict(self, text: str) -> dict[str, float | str]:
        """
        Predict sentiment label and confidence for a single review.
        """
        if self.model is None:
            raise ValueError("Model is not loaded properly.")

        prepared_text = self._prepare_text(text)
        probability = float(self.model.predict(prepared_text, verbose=0)[0][0])

        sentiment = "positive" if probability >= 0.5 else "negative"
        confidence = probability if sentiment == "positive" else 1 - probability

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probability": round(probability, 4),
        }


# Optional singleton instance for easy import in FastAPI
predictor = SentimentPredictor()
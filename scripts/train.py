from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "reviews.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RANDOM_STATE = 42
VOCAB_SIZE = 5000
MAX_LEN = 100
EMBEDDING_DIM = 64
TEST_SIZE = 0.2
EPOCHS = 5
BATCH_SIZE = 16

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean input review text for basic NLP preprocessing.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load dataset and validate required columns.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)

    logger.info("Dataset loaded successfully with %s rows.", len(df))
    return df


def prepare_tokenizer(texts: pd.Series) -> Tokenizer:  
    """
    Fit tokenizer on training texts.
    """
    tokenizer = Tokenizer(       # word-to-number dictionary
        num_words=VOCAB_SIZE, 
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts) # tokeniz text wise because "fit_on_texts" is designed for text data
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: pd.Series) -> np.ndarray:
    """
    Convert text to padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


def build_model() -> keras.Model:
    """
    Build a simple 1D CNN model for binary sentiment classification.
    """
    model = keras.Sequential(
        [
            Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(model: keras.Model, tokenizer: Tokenizer) -> None:
    """
    Save trained model, tokenizer, and label config.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "model.keras"
    tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"
    label_config_path = ARTIFACTS_DIR / "label_config.json"

    model.save(model_path)

    with tokenizer_path.open("wb") as f:
        pickle.dump(tokenizer, f)

    label_config = {
        "negative": 0,
        "positive": 1,
        "max_len": MAX_LEN,
        "vocab_size": VOCAB_SIZE,
    }
    with label_config_path.open("w", encoding="utf-8") as f:
        json.dump(label_config, f, indent=2)

    logger.info("Artifacts saved successfully in %s", ARTIFACTS_DIR)


def main() -> None:
    """
    Run the full training pipeline.
    """
    logger.info("Starting training pipeline...")

    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    tokenizer = prepare_tokenizer(X_train)

    X_train_padded = encode_texts(tokenizer, X_train)
    X_test_padded = encode_texts(tokenizer, X_test)

    model = build_model()

    logger.info("Training model...")
    model.fit(   # train the model using the padded sequences and labels
        X_train_padded,
        y_train,
        validation_data=(X_test_padded, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    logger.info("Test loss: %.4f", loss)
    logger.info("Test accuracy: %.4f", accuracy)

    save_artifacts(model, tokenizer)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
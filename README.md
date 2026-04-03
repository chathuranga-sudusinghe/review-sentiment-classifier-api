# Review Sentiment Classifier API

A simple FastAPI-based NLP mini project for predicting review sentiment as **positive** or **negative** using a small deep learning model.

## Project Overview

This project is a lightweight sentiment analysis API built for skill polishing in:

- NLP preprocessing
- Deep learning-based text classification
- FastAPI API development
- Docker containerization
- GitHub Actions CI

The API accepts a review text and returns:

- predicted sentiment
- confidence score

---

## Features

- Binary sentiment prediction: **positive / negative**
- FastAPI REST API
- Input validation with Pydantic
- Trained TensorFlow/Keras model
- Docker support
- API testing with Pytest
- GitHub Actions CI

---

## Project Structure

```text
review-sentiment-classifier-api/
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── artifacts/
│   ├── model.keras
│   ├── tokenizer.pkl
│   └── label_config.json
├── data/
│   └── reviews.csv
├── scripts/
│   └── train.py
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md

## Tech Stack

- Python 3.12
- FastAPI
- TensorFlow / Keras
- Pydantic
- Pytest
- Docker
- GitHub Actions
```
---

## How It Works

### Training Flow

1. Load review dataset
2. Clean and preprocess text
3. Tokenize and pad sequences
4. Train a simple sentiment model
5. Save model artifacts

### Inference Flow

1. User sends review text to `/predict`
2. API validates the request
3. Saved tokenizer and model process the text
4. API returns sentiment and confidence

---

## Setup Instructions

### 1. Create and activate virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Run the API Locally
```
uvicorn app.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs

```

## Example Prediction Request
#### Request body
```
{
  "text": "This product is really good and very useful."
}
```


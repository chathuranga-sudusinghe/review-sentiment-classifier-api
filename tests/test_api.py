from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_success() -> None:
    """
    Test that a valid review returns 200 OK
    and contains the expected response fields.
    """
    payload = {"text": "This product is really good and very useful."}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data

    assert data["sentiment"] in ["positive", "negative"]
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_text() -> None:
    """
    Test that empty text is rejected by validation.
    """
    payload = {"text": "   "}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_text_field() -> None:
    """
    Test that missing required field causes validation error.
    """
    payload = {}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_body_type() -> None:
    """
    Test that invalid request body type is rejected.
    """
    payload = {"text": 12345}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
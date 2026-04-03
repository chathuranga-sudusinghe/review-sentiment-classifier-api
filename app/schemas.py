from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel): # input data schema for the prediction endpoint
    """
    Request schema for sentiment prediction.
    """

    text: str = Field(..., min_length=1, description="Review text to classify")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Reject empty or whitespace-only review text.
        """
        cleaned_value = value.strip() # remove leading/trailing whitespace: start/end of the string
        if not cleaned_value:
            raise ValueError("Review text must not be empty or only whitespace.")
        return cleaned_value


class PredictionResponse(BaseModel): # output data schema for the prediction endpoint
    """
    Response schema for sentiment prediction.
    """

    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )
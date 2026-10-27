"""
Pydantic Data Schemas for TruthLens API
"""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="The news headline or article body to analyze."
    )
    num_explanation_words: int = Field(
        default=8,
        ge=3,
        le=20,
        description="Number of top influential words to extract via LIME."
    )
    source_url: str | None = Field(default=None, max_length=2048)

    @field_validator("text")
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 10:
            raise ValueError("News text must be at least 10 characters long.")
        if len(cleaned.split()) < 3:
            raise ValueError("News text must contain at least 3 words for accurate analysis.")
        return cleaned


class InfluentialWord(BaseModel):
    word: str
    weight: float
    supports: str = Field(description="'Real' or 'Fake'")
    importance: float


class PredictionResponse(BaseModel):
    label: str = Field(description="'Real' or 'Fake'")
    confidence_score: float = Field(description="Confidence percentage (0 to 100)")
    probabilities: dict[str, float] = Field(description="Probabilities for each class")
    top_influential_words: list[InfluentialWord] = Field(description="Words most influencing the prediction")
    cleaned_text_preview: str | None = None
    credibility: dict | None = None


class CredibilityResponse(BaseModel):
    domain: str
    rating: str
    source_name: str | None = None
    message: str


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)

    @field_validator("text")
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned.split()) < 3:
            raise ValueError("News text must contain at least 3 words for accurate analysis.")
        return cleaned


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    pipeline_type: str


class MetricsResponse(BaseModel):
    best_model: str
    models: list[dict]

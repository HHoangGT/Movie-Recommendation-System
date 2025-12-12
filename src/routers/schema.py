"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Generic message response."""

    detail: str


# ========== LLM Schemas ==========


class LLMInput(BaseModel):
    """Input for LLM-based recommendations."""

    movie_name: str = Field(..., description="Name of the input movie")
    movie_genre: str = Field("", description="Genre(s) of the movie")
    movie_overview: str = Field("", description="Overview/description of the movie")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations")


# ========== Movie Management Schemas ==========


class MovieAddRequest(BaseModel):
    """Request to add a movie to history."""

    movie_id: int = Field(..., description="Movie ID to add")


class MovieRemoveRequest(BaseModel):
    """Request to remove a movie from history."""

    movie_id: int = Field(..., description="Movie ID to remove")


class MovieBatchRequest(BaseModel):
    """Request to add multiple movies to history."""

    movie_ids: list[int] = Field(..., description="List of movie IDs")


# ========== Model Management Schemas ==========


class ModelSwitchRequest(BaseModel):
    """Request to switch active recommendation model."""

    model: str = Field(..., description="Model name: 'nextitnet', 'bivae', or 'llm'")


# ========== Recommendation Schemas ==========


class RecommendationRequest(BaseModel):
    """Generic recommendation request."""

    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations")
    exclude_history: bool = Field(True, description="Exclude movies from history")


class SequenceRequest(BaseModel):
    """Request for sequence-based recommendations (NextItNet)."""

    movie_ids: list[int] = Field(..., description="Sequence of movie IDs")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations")
    exclude_input: bool = Field(True, description="Exclude input movies from results")

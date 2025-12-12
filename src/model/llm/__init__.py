"""
LLM-based Movie Recommendation Module
Using DSPy for Language Model inference
"""

from .inference import LLMRecommender
from .llm import MovieRecommendationProgram, MovieRecommendation

__all__ = ["LLMRecommender", "MovieRecommendationProgram", "MovieRecommendation"]

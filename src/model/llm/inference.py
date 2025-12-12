"""
LLM-based Movie Recommendation Inference
Using DSPy with Language Models and Google Search
"""

import os
import dspy
from typing import Any
from dotenv import load_dotenv

from .llm import MovieRecommendationProgram
from ...config.config import Settings


class LLMRecommender:
    """LLM-based Movie Recommender using DSPy."""

    def __init__(self, config: Settings):
        self.config = config
        self._is_ready = False
        self.program: MovieRecommendationProgram | None = None
        self.lm = None

        self._initialize()

    def _initialize(self):
        """Initialize DSPy with Language Model."""
        try:
            # Load environment variables
            load_dotenv()

            # Check if API key is available (including OpenRouter)
            api_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or self.config.openrouter_api_key
            )

            if not api_key:
                print(
                    "⚠️  No LLM API key found. Set OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
                )
                return

            # Create LM configuration
            lm_config = {
                "model": self.config.llm_model,
                "cache": self.config.llm_cache,
                "num_retries": self.config.llm_num_retries,
                "max_tokens": self.config.llm_max_tokens,
            }

            # Set OpenRouter API key if using OpenRouter models
            if self.config.openrouter_api_key and "openrouter/" in self.config.model:
                os.environ["OPENROUTER_API_KEY"] = self.config.openrouter_api_key

            self.lm = dspy.LM(**lm_config)

            # Configure Google Search tool if available
            tools = []
            if os.getenv("GOOGLE_SEARCH_API_KEY"):
                tools.append({"googleSearch": {}})

            # Configure DSPy
            dspy.configure(lm=self.lm, tools=tools)

            # Initialize the program
            self.program = MovieRecommendationProgram()

            self._is_ready = True
            print(f"✅ LLM Recommender initialized with model: {self.config.llm_model}")

        except Exception as e:
            print(f"❌ Error initializing LLM Recommender: {e}")
            import traceback

            traceback.print_exc()
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def get_recommendations(
        self,
        movie_name: str,
        movie_genre: str = "",
        movie_overview: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Get movie recommendations using LLM.

        Args:
            movie_name: Name of the input movie
            movie_genre: Genre(s) of the movie
            movie_overview: Overview/description of the movie
            top_k: Number of recommendations (default 10)

        Returns:
            Dictionary with recommendations
        """
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "LLM Recommender not initialized. Check API keys.",
            }

        try:
            # Call DSPy program
            result = await self.program.aforward(
                movie_name=movie_name,
                movie_genre=movie_genre if movie_genre else "Unknown",
                movie_overview=movie_overview
                if movie_overview
                else "No overview available",
            )

            # Parse recommendations
            recommendations = []
            if hasattr(result, "recommendations") and result.recommendations:
                for idx, rec in enumerate(result.recommendations[:top_k]):
                    recommendations.append(
                        {
                            "title": rec.movie_name,
                            "genres": [g.strip() for g in rec.movie_genre.split(",")],
                            "overview": rec.movie_overview,
                            # Decreasing score by position
                            "score": 1.0 - (idx * 0.05),
                            "model": "LLM (DSPy)",
                            "source": "llm",
                        }
                    )

            return {
                "success": True,
                "recommendations": recommendations,
                "model_used": "llm",
                "message": f"Generated {len(recommendations)} recommendations using LLM",
                "input_movie": {
                    "title": movie_name,
                    "genres": movie_genre,
                    "overview": movie_overview,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "recommendations": [],
                "message": f"LLM error: {str(e)}",
            }

    def get_recommendations_sync(
        self,
        movie_name: str,
        movie_genre: str = "",
        movie_overview: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Synchronous version of get_recommendations."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "LLM Recommender not initialized. Check API keys.",
            }

        try:
            # Call DSPy program synchronously
            result = self.program.forward(
                movie_name=movie_name,
                movie_genre=movie_genre if movie_genre else "Unknown",
                movie_overview=movie_overview
                if movie_overview
                else "No overview available",
            )

            # Parse recommendations
            recommendations = []
            if hasattr(result, "recommendations") and result.recommendations:
                for idx, rec in enumerate(result.recommendations[:top_k]):
                    recommendations.append(
                        {
                            "title": rec.movie_name,
                            "genres": [g.strip() for g in rec.movie_genre.split(",")],
                            "overview": rec.movie_overview,
                            "score": 1.0 - (idx * 0.05),
                            "model": "LLM (DSPy)",
                            "source": "llm",
                        }
                    )

            return {
                "success": True,
                "recommendations": recommendations,
                "model_used": "llm",
                "message": f"Generated {len(recommendations)} recommendations using LLM",
                "input_movie": {
                    "title": movie_name,
                    "genres": movie_genre,
                    "overview": movie_overview,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "recommendations": [],
                "message": f"LLM error: {str(e)}",
            }

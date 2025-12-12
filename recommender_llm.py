"""
LLM Recommender Wrapper for Production

This module provides LLM-based movie recommendations using DSPy framework.
Supports multiple models through DSPy's unified interface.
"""

import os
from typing import Any

try:
    import dspy
    from pydantic import BaseModel, Field

    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    print("dspy package not installed. LLM features disabled.")


class MovieRecommendation(BaseModel):
    """Single movie recommendation structure."""

    movie_name: str = Field(description="Recommended movie title")
    movie_genre: str = Field(description="Movie genre")
    movie_overview: str = Field(description="Brief movie description")
    similarity_score: float = Field(
        description="Similarity score with input movie (0.0-1.0)", ge=0.0, le=1.0
    )


class MovieRecommendationSignature(dspy.Signature):
    """
    **Movie Recommendation Expert Task**

    You are a movie recommendation expert with deep analytical abilities in content, genres, and characteristics of films.
    Your task is to recommend similar movies based on the input movie information (title, genre, description) that would
    best suit viewers who enjoyed the input movie.

    **Analysis Guidelines:**

    1. **Input Movie Analysis:**
       - Identify key elements: genre, theme, tone, style
       - Recognize prominent features in the description (action, psychological, comedy, horror, etc.)

    2. **Recommendation Criteria:**
       - **Genre Similarity**: Prioritize movies in the same or closely related genres
       - **Theme and Content**: Find movies with similar plot, setting, or message
       - **Style and Tone**: Maintain consistency in emotion and storytelling style
       - **Diversification**: Include moderate variety to provide a rich experience
       - **Quality**: Prioritize well-reviewed and critically acclaimed films

    3. **Calculating Similarity Score (0.0 - 1.0):**
       - **1.0 - 0.9**: Very similar - same main genre, theme and style nearly identical
       - **0.9 - 0.8**: Highly similar - same genre, similar theme with minor differences
       - **0.8 - 0.7**: Moderately similar - related genre, many common content elements
       - **0.7 - 0.6**: Somewhat similar - some common elements in genre or theme
       - **< 0.6**: Low similarity - few common elements

       Consider these factors when scoring:
       - Primary and secondary genres (35%)
       - Theme and plot content (30%)
       - Style and tone (20%)
       - Target audience (10%)
       - Time period/setting (5%)

    4. **Output Format:**
       - Each recommendation must include: `movie_name`, `movie_genre`, `movie_overview`, `similarity_score`
       - `movie_overview` should be concise (2-3 sentences) but informative enough for viewers to understand the main content
       - `similarity_score` must be a decimal from 0.0 to 1.0, accurately reflecting the similarity level
       - Ensure information is accurate and up-to-date

    **IMPORTANT PRINCIPLES:**
    - Recommendations must be BASED ON FACTUAL and accurate information about real movies
    - Do NOT fabricate movie titles or information that doesn't exist
    - Recommendations must be relevant and valuable to viewers who enjoyed the input movie
    - ONLY recommend movies that ACTUALLY EXIST

    **Output Requirements:**
    - Return a list of recommended movies (quantity as requested)
    - Each movie must have complete information: title, genre, description, and similarity score
    - Sort by similarity_score from high to low (most similar movies first)
    - Ensure similarity_score is calculated carefully and accurately
    """

    movie_name = dspy.InputField(desc="Input movie title")
    movie_genre = dspy.InputField(desc="Input movie genre")
    movie_overview = dspy.InputField(desc="Input movie description summary")

    recommendations: list[MovieRecommendation] = dspy.OutputField(
        default_factory=list, desc="List of recommended movies"
    )


class MovieRecommendationProgram(dspy.Module):
    """DSPy Module for movie recommendations using Chain of Thought."""

    def __init__(self):
        super().__init__()
        self._module = dspy.ChainOfThought(MovieRecommendationSignature)

    def forward(
        self, movie_name: str, movie_genre: str, movie_overview: str
    ) -> dspy.Prediction:
        """Synchronous forward pass."""
        return self._module(
            movie_name=movie_name,
            movie_genre=movie_genre,
            movie_overview=movie_overview,
        )


class LLMRecommender:
    """LLM-based Movie Recommender using DSPy framework."""

    def __init__(self, config=None):
        self.config = config
        self._is_ready = False
        self.model_name = None
        self.program = None

        self._initialize()

    def _initialize(self):
        """Initialize the LLM recommender with DSPy."""
        if not HAS_DSPY:
            print("LLM Recommender: dspy package not installed")
            return

        # Load from .env file if exists
        self._load_dotenv()

        # Get API key and model configuration
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key and self.config:
            api_key = getattr(self.config, "openrouter_api_key", None)

        if not api_key:
            print("LLM Recommender: OPENROUTER_API_KEY not set")
            return

        # Get model from env
        model = os.environ.get("MODEL", "openrouter/google/gemini-2.0-flash-001")

        # Ensure model has 'openrouter/' prefix for DSPy
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        self.model_name = model

        try:
            # Configure DSPy with OpenRouter
            lm = dspy.LM(
                model=self.model_name,
                api_key=api_key,
                max_tokens=int(os.environ.get("MAX_TOKENS", 4096)),
                cache=False,
            )

            # Set as default LM for DSPy
            dspy.configure(lm=lm)

            # Initialize the recommendation program
            self.program = MovieRecommendationProgram()

            self._is_ready = True
            print(f"LLM Recommender: DSPy initialized with model {self.model_name}")
        except Exception as e:
            print(f"LLM Recommender: Failed to initialize DSPy - {e}")

    def _load_dotenv(self):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv

            # Try to load from production folder
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"LLM Recommender: Loaded .env from {env_path}")
        except ImportError:
            pass  # dotenv not installed, use system env vars

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(
        self,
        movie_name: str,
        movie_genre: str = "",
        movie_overview: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Get movie recommendations based on content similarity using LLM via DSPy."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "LLM Recommender not initialized. Set OPENROUTER_API_KEY.",
            }

        try:
            # Call DSPy program
            prediction = self.program(
                movie_name=movie_name,
                movie_genre=movie_genre if movie_genre else "Not specified",
                movie_overview=movie_overview if movie_overview else "Not specified",
            )

            # Extract recommendations from prediction
            recommendations = self._format_recommendations(
                prediction.recommendations, top_k
            )

            return {
                "success": True,
                "recommendations": recommendations,
                "message": f"Generated {len(recommendations)} recommendations",
                "input_movie": {
                    "title": movie_name,
                    "genres": movie_genre,
                    "overview": movie_overview,
                },
                "model": self.model_name,
            }

        except Exception as e:
            return {
                "success": False,
                "recommendations": [],
                "message": f"LLM error: {str(e)}",
            }

    def _format_recommendations(self, recommendations: list, top_k: int) -> list:
        """Format DSPy recommendations to match expected output format."""
        formatted = []

        for idx, rec in enumerate(recommendations[:top_k]):
            # Handle both Pydantic models and dict-like objects
            if isinstance(rec, MovieRecommendation):
                movie_name = rec.movie_name
                movie_genre = rec.movie_genre
                movie_overview = rec.movie_overview
                similarity_score = rec.similarity_score
            elif isinstance(rec, dict):
                movie_name = rec.get("movie_name", "")
                movie_genre = rec.get("movie_genre", "")
                movie_overview = rec.get("movie_overview", "")
                similarity_score = rec.get("similarity_score", 1.0 - (idx * 0.05))
            else:
                # Try to access as attributes
                movie_name = getattr(rec, "movie_name", "")
                movie_genre = getattr(rec, "movie_genre", "")
                movie_overview = getattr(rec, "movie_overview", "")
                similarity_score = getattr(rec, "similarity_score", 1.0 - (idx * 0.05))

            # Skip empty recommendations
            if not movie_name:
                continue

            # Parse genre into list if it's a string
            if isinstance(movie_genre, str):
                genres = [g.strip() for g in movie_genre.split(",") if g.strip()]
            else:
                genres = movie_genre if isinstance(movie_genre, list) else []

            formatted_rec = {
                "title": movie_name,
                "genres": genres,
                "overview": movie_overview,
                "score": similarity_score,  # Use LLM-generated score
                "source": "llm",
            }

            formatted.append(formatted_rec)

        return formatted

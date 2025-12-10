import dspy
from dotenv import load_dotenv
from loguru import logger
from fastapi.encoders import jsonable_encoder

from src.config import get_config
from .llm import MovieRecommendationProgram

settings = get_config()


def create_lm(model: str) -> dspy.LM:
    """
    Create and configure a language model instance with Google Search tool.

    Args:
        model: The model identifier string

    Returns:
        Configured dspy.LM instance with Google Search capabilities
    """
    load_dotenv()

    lm_config = {
        "model": model,
        "cache": settings.cache,
        "num_retries": settings.num_retries,
        "max_tokens": settings.max_tokens,
    }

    return dspy.LM(**lm_config)


# Initialize the DSPy program
dspy_program = MovieRecommendationProgram()

# Configure DSPy with the language model
lm = create_lm(settings.model)
# Configure Google Search tool
tools = [{"googleSearch": {}}]
dspy.configure(lm=lm, tools=tools)


async def movie_recommendation_inference(
    movie_name: str, movie_genre: str, movie_overview: str
) -> tuple:
    """
    Main inference function for movie recommendations.

    Args:
        movie_name: Name of the input movie
        movie_genre: Genre of the input movie
        movie_overview: Overview description of the input movie

    Returns:
        Tuple containing (result_dict, display_string)
    """
    try:
        # Call the DSPy program
        result = await dspy_program.acall(
            movie_name=movie_name,
            movie_genre=movie_genre,
            movie_overview=movie_overview,
        )
        return jsonable_encoder(result)

    except Exception as e:
        logger.error(f"Error in movie recommendation inference: {str(e)}")
        raise e

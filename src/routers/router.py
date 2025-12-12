"""
API Router for Movie Recommendation System
Handles all recommendation endpoints for NextItNet, BiVAE, and LLM models
"""

from loguru import logger
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import get_config
from ..model.nextitnet import NextItNetRecommender
from ..model.bivae import BiVAERecommender
from ..model.llm import LLMRecommender
from . import schema

# Initialize router and limiter
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
config = get_config()

# Initialize recommenders (will be set in main.py)
nextitnet_recommender: NextItNetRecommender | None = None
bivae_recommender: BiVAERecommender | None = None
llm_recommender: LLMRecommender | None = None

# App state
app_state = {"active_model": config.default_model}


def init_recommenders(nextitnet, bivae, llm):
    """Initialize recommenders from main app."""
    global nextitnet_recommender, bivae_recommender, llm_recommender
    nextitnet_recommender = nextitnet
    bivae_recommender = bivae
    llm_recommender = llm


# ==================== Health & Status ====================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "success": True,
        "active_model": app_state["active_model"],
        "models": {
            "nextitnet": {
                "status": "ready"
                if nextitnet_recommender and nextitnet_recommender.is_ready
                else "not loaded",
                "type": "Sequential (Collaborative Filtering)",
            },
            "llm": {
                "status": "ready"
                if llm_recommender and llm_recommender.is_ready
                else "not configured",
                "type": "Content-Based (LLM/DSPy)",
            },
            "bivae": {
                "status": "ready"
                if bivae_recommender and bivae_recommender.is_ready
                else "not loaded",
                "type": "Collaborative Filtering (VAE)",
            },
        },
        "num_movies": nextitnet_recommender.data_manager.num_items
        if nextitnet_recommender and nextitnet_recommender.is_ready
        else 0,
    }


# ==================== Model Management ====================


@router.post("/model/switch")
async def switch_model(request: schema.ModelSwitchRequest):
    """Switch the active recommendation model."""
    model = request.model

    if model not in ["nextitnet", "llm", "bivae"]:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid model: {model}. Use "nextitnet", "llm", or "bivae"',
        )

    if model == "llm" and (not llm_recommender or not llm_recommender.is_ready):
        raise HTTPException(
            status_code=400, detail="LLM recommender not configured. Check API keys."
        )

    if model == "bivae" and (not bivae_recommender or not bivae_recommender.is_ready):
        raise HTTPException(
            status_code=400, detail="BiVAE model not ready. Train it first."
        )

    app_state["active_model"] = model

    return {
        "success": True,
        "active_model": model,
        "message": f"Switched to {model} model",
    }


@router.get("/model/active")
async def get_active_model():
    """Get the currently active model."""
    return {"success": True, "active_model": app_state["active_model"]}


# ==================== Recommendations ====================


@router.get("/recommendations/{user_id}")
@limiter.limit(f"{config.limiter_requests}/minute")
async def get_recommendations(
    request: Request, user_id: str, top_k: int = 10, exclude_history: bool = True
):
    """Get recommendations using the currently active model."""

    if app_state["active_model"] == "nextitnet":
        if not nextitnet_recommender:
            raise HTTPException(status_code=503, detail="NextItNet not initialized")
        result = nextitnet_recommender.get_recommendations(
            user_id, top_k, exclude_history
        )

    elif app_state["active_model"] == "bivae":
        if not bivae_recommender:
            raise HTTPException(status_code=503, detail="BiVAE not initialized")
        result = bivae_recommender.get_recommendations(user_id, top_k)

    else:  # LLM
        if not llm_recommender:
            raise HTTPException(status_code=503, detail="LLM not initialized")

        # Get last movie from history for LLM
        if not nextitnet_recommender:
            raise HTTPException(status_code=503, detail="Data manager not initialized")

        history = nextitnet_recommender.get_user_history(user_id)
        if not history["history"]:
            return {
                "success": False,
                "recommendations": [],
                "message": "No history for LLM recommendations. Add movies first.",
            }

        last_movie = history["history"][-1]
        result = await llm_recommender.get_recommendations(
            movie_name=last_movie.get("title", ""),
            movie_genre=", ".join(last_movie.get("genres", [])),
            movie_overview=last_movie.get("overview", ""),
            top_k=top_k,
        )

    result["model_used"] = app_state["active_model"]
    return result


@router.post("/recommendations/llm")
@limiter.limit(f"{config.limiter_requests}/minute")
async def get_llm_recommendations(request: Request, input_data: schema.LLMInput):
    """Get recommendations from LLM based on a movie."""
    if not llm_recommender or not llm_recommender.is_ready:
        raise HTTPException(
            status_code=400, detail="LLM recommender not configured. Set API keys."
        )

    try:
        result = await llm_recommender.get_recommendations(
            movie_name=input_data.movie_name,
            movie_genre=input_data.movie_genre,
            movie_overview=input_data.movie_overview,
            top_k=input_data.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"LLM recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ==================== User History ====================


@router.get("/history/{user_id}")
async def get_user_history(user_id: str):
    """Get user's viewing history."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.get_user_history(user_id)


@router.post("/history/{user_id}")
async def add_to_history(user_id: str, request: schema.MovieAddRequest):
    """Add a movie to user's history."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.add_to_history(user_id, request.movie_id)


@router.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear user's viewing history."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.clear_user_history(user_id)


@router.post("/history/{user_id}/remove")
async def remove_from_history(user_id: str, request: schema.MovieRemoveRequest):
    """Remove a specific movie from user's history."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.remove_from_history(user_id, request.movie_id)


# ==================== Movies ====================


@router.get("/movies/search")
async def search_movies(q: str, limit: int = 10):
    """Search for movies by title."""
    if not q:
        raise HTTPException(status_code=400, detail='Query parameter "q" is required')

    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.search_movies(q, limit)


@router.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details for a specific movie."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.get_movie(movie_id)


@router.get("/movies")
async def get_movies(limit: int = 20, vocab_only: bool = False):
    """Get list of movies."""
    if not nextitnet_recommender:
        raise HTTPException(status_code=503, detail="System not initialized")

    return nextitnet_recommender.get_popular_movies(limit, vocab_only)

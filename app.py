"""
Unified Movie Recommendation System - Production Server (FastAPI)
Combines NextItNet and LLM-based recommendations

This FastAPI application provides:
- NextItNet: Sequential recommendation based on user history
- LLM (OpenRouter/Gemini): Content-based recommendation using AI
- Model switching capability from the Data Scientist dashboard
"""

import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    print("python-dotenv not installed, using system environment variables")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from recommender_nextitnet import NextItNetRecommender
from recommender_llm import LLMRecommender
from config import Config
from local_config import verify_setup
from training_manager import training_manager

from recommender_bivae import BiVAERecommender

# Verify setup
if not verify_setup():
    print("\n⚠️  Some files are missing. Run: python setup_files.py")
    print("Continuing anyway...")

# Configuration
config = Config()

# Initialize recommenders
print("\nInitializing recommenders...")
nextitnet = NextItNetRecommender()
llm_recommender = LLMRecommender(config)
bivae_recommender = BiVAERecommender()

# Active model state (default: nextitnet)
app_state = {
    "active_model": "llm",  # or 'llm'
    "user_sessions": {},
}

# Create FastAPI app
app = FastAPI(
    title="Unified Movie Recommendation System",
    description="NextItNet + LLM (OpenRouter) Movie Recommendations",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================
# Pydantic Models
# ============================================


class MovieAddRequest(BaseModel):
    movie_id: int


class MovieRemoveRequest(BaseModel):
    movie_id: int


class MovieBatchRequest(BaseModel):
    movie_ids: list[int]


class ModelSwitchRequest(BaseModel):
    model: str


class LLMRequest(BaseModel):
    movie_name: str
    movie_genre: str = ""
    movie_overview: str = ""
    top_k: int = 10


class SequenceRequest(BaseModel):
    movie_ids: list[int]
    top_k: int = 10
    exclude_input: bool = True


class TrainingRequest(BaseModel):
    model: str
    version: str
    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 256
    use_gpu: bool = True
    early_stopping: bool = True


# ============================================
# ROUTES
# ============================================


@app.get("/")
async def index():
    """Serve the main web application."""
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))


@app.get("/login")
async def login_page():
    """Serve the login page."""
    return FileResponse(os.path.join(TEMPLATES_DIR, "login.html"))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "success": True,
        "is_ready": nextitnet.is_ready,
        "device": str(nextitnet.device) if nextitnet.is_ready else "N/A",
        "active_model": app_state["active_model"],
        "models": {
            "nextitnet": {
                "status": "ready" if nextitnet.is_ready else "not loaded",
                "type": "Sequential (Collaborative Filtering)",
            },
            "llm": {
                "status": "ready" if llm_recommender.is_ready else "not configured",
                "type": "Content-Based (LLM)",
            },
            "bivae": {
                "status": "ready" if bivae_recommender.is_ready else "not configured",
                "type": "Collaborative Filtering",
            },
        },
        "num_movies": nextitnet.data_manager.num_items if nextitnet.is_ready else 0,
        "num_sessions": len(app_state["user_sessions"]),
    }


@app.post("/api/model/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch the active recommendation model."""
    model = request.model

    if model not in ["nextitnet", "llm", "bivae"]:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid model: {model}. Use "nextitnet", "llm" or "bivae"',
        )

    if model == "llm" and not llm_recommender.is_ready:
        raise HTTPException(
            status_code=400,
            detail="LLM recommender not configured. Please set OPENROUTER_API_KEY.",
        )
    if model == "bivae" and not bivae_recommender.is_ready:
        raise HTTPException(
            status_code=400, detail="BiVAE model not ready. Train it first."
        )
    app_state["active_model"] = model

    return {
        "success": True,
        "active_model": model,
        "message": f"Switched to {model} model",
    }


@app.get("/api/model/active")
async def get_active_model():
    """Get the currently active model."""
    return {"success": True, "active_model": app_state["active_model"]}


# ============================================
# RECOMMENDATION ENDPOINTS
# ============================================


@app.get("/api/recommendations/{user_id}")
async def get_recommendations(
    user_id: str, top_k: int = 10, exclude_history: bool = True
):
    """Get recommendations for a user using the currently active model."""

    if app_state["active_model"] == "nextitnet":
        result = nextitnet.get_recommendations(user_id, top_k, exclude_history)

    elif app_state["active_model"] == "bivae":
        result = bivae_recommender.get_recommendations(user_id, top_k)

    else:
        # For LLM, we need the last movie from history
        history = nextitnet.get_user_history(user_id)
        if not history["history"]:
            return {
                "success": False,
                "recommendations": [],
                "message": "No history for LLM recommendations. Add movies first.",
            }

        last_movie = history["history"][-1]
        result = llm_recommender.get_recommendations(
            movie_name=last_movie.get("title", ""),
            movie_genre=", ".join(last_movie.get("genres", [])),
            movie_overview=last_movie.get("overview", ""),
            top_k=top_k,
        )

    result["model_used"] = app_state["active_model"]
    return result


@app.get("/api/recommendations/nextitnet/{user_id}")
async def get_nextitnet_recommendations(
    user_id: str, top_k: int = 10, exclude_history: bool = True
):
    """Get recommendations specifically from NextItNet."""
    result = nextitnet.get_recommendations(user_id, top_k, exclude_history)
    result["model_used"] = "nextitnet"
    return result


@app.post("/api/recommendations/llm")
async def get_llm_recommendations(request: LLMRequest):
    """Get recommendations from LLM based on a movie."""
    if not llm_recommender.is_ready:
        raise HTTPException(
            status_code=400,
            detail="LLM recommender not configured. Set OPENROUTER_API_KEY.",
        )

    if not request.movie_name:
        raise HTTPException(status_code=400, detail="movie_name is required")

    result = llm_recommender.get_recommendations(
        movie_name=request.movie_name,
        movie_genre=request.movie_genre,
        movie_overview=request.movie_overview,
        top_k=request.top_k,
    )
    result["model_used"] = "llm"
    return result


@app.post("/api/recommendations/sequence")
async def get_recommendations_for_sequence(request: SequenceRequest):
    """Get NextItNet recommendations for a sequence of movies."""
    result = nextitnet.get_recommendations_for_sequence(
        request.movie_ids, request.top_k, request.exclude_input
    )
    result["model_used"] = "nextitnet"
    return result


# ============================================
# USER HISTORY ENDPOINTS
# ============================================


@app.get("/api/history/{user_id}")
async def get_user_history(user_id: str):
    """Get user's viewing history."""
    result = nextitnet.get_user_history(user_id)
    return result


@app.post("/api/history/{user_id}")
async def add_to_history(user_id: str, request: MovieAddRequest):
    """Add a movie to user's history."""
    result = nextitnet.add_to_history(user_id, request.movie_id)
    return result


@app.post("/api/history/{user_id}/batch")
async def add_multiple_to_history(user_id: str, request: MovieBatchRequest):
    """Add multiple movies to user's history."""
    result = nextitnet.add_multiple_to_history(user_id, request.movie_ids)
    return result


@app.delete("/api/history/{user_id}")
async def clear_history(user_id: str):
    """Clear user's viewing history."""
    result = nextitnet.clear_user_history(user_id)
    return result


@app.post("/api/history/{user_id}/remove")
async def remove_from_history(user_id: str, request: MovieRemoveRequest):
    """Remove a specific movie from user's history."""
    result = nextitnet.remove_from_history(user_id, request.movie_id)
    return result


# ============================================
# MOVIE ENDPOINTS
# ============================================


@app.get("/api/movies/search")
async def search_movies(q: str, limit: int = 10):
    """Search for movies by title."""
    if not q:
        raise HTTPException(status_code=400, detail='Query parameter "q" is required')

    result = nextitnet.search_movies(q, limit)
    return result


@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details for a specific movie."""
    result = nextitnet.get_movie(movie_id)
    return result


@app.get("/api/movies")
async def get_movies(limit: int = None, vocab_only: bool = False):
    """
    Get list of movies.

    Args:
        limit: Maximum number of movies (default: all 9,066 movies)
        vocab_only: Only return movies in vocabulary if True
    """
    return nextitnet.get_popular_movies(limit, vocab_only)


# ============================================
# ADMIN ENDPOINTS
# ============================================


@app.get("/api/admin/stats")
async def get_admin_stats():
    """Get system statistics for admin."""
    return {
        "success": True,
        "stats": {
            "total_movies": nextitnet.data_manager.num_items
            if nextitnet.is_ready
            else 0,
            "total_sessions": len(nextitnet.data_manager.user_sessions)
            if nextitnet.is_ready
            else 0,
            "active_model": app_state["active_model"],
            "nextitnet_status": "ready" if nextitnet.is_ready else "not loaded",
            "llm_status": "ready" if llm_recommender.is_ready else "not configured",
        },
    }


@app.get("/api/admin/movies/all")
async def get_all_movies_admin(limit: int = 100, offset: int = 0):
    """Get all movies for admin panel."""
    if not nextitnet.is_ready:
        return {"success": False, "message": "System not ready"}

    all_movies = nextitnet.data_manager.get_all_movies(None)
    total = len(all_movies)
    paginated = all_movies[offset : offset + limit]

    # Sanitize each movie (convert NaN to None for JSON)
    sanitized_movies = []
    for m in paginated:
        if "movie_id" not in m and "id" in m:
            m["movie_id"] = m["id"]
        # Use the sanitize method from nextitnet recommender
        sanitized_movies.append(nextitnet._sanitize_movie_data(m))

    return {
        "success": True,
        "movies": sanitized_movies,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/admin/users")
async def get_all_users_admin():
    """Get all user sessions for admin."""
    if not nextitnet.is_ready:
        return {"success": False, "message": "System not ready"}

    users = []
    for user_id, history in nextitnet.data_manager.user_sessions.items():
        users.append(
            {
                "user_id": user_id,
                "history_length": len(history),
                "last_active": "N/A",  # Could add timestamp tracking
            }
        )

    return {"success": True, "users": users, "total": len(users)}


# ============================================
# TRAINING ENDPOINTS (Data Scientist Only)
# ============================================


@app.post("/api/training/start")
async def start_training(training_req: TrainingRequest):
    """Start a new training job."""
    # Validate model
    if training_req.model.lower() not in ["nextitnet"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{training_req.model}' not supported for training",
        )

    # Create training job
    job_id = training_manager.create_job(training_req.dict())

    # Start training in background
    success = training_manager.start_training(job_id)

    if not success:
        job = training_manager.get_job(job_id)
        raise HTTPException(
            status_code=400,
            detail=job.error_message if job else "Failed to start training",
        )

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Training job started for {training_req.model} v{training_req.version}",
    }


@app.get("/api/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status and progress."""
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {"success": True, "job": job.to_dict()}


@app.get("/api/training/jobs")
async def list_training_jobs():
    """List all training jobs."""
    jobs = training_manager.list_jobs()
    return {"success": True, "jobs": jobs, "total": len(jobs)}


@app.delete("/api/training/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a running training job."""
    success = training_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not running or not found")

    return {"success": True, "message": "Training job cancelled"}


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Movie Recommendation System (FastAPI)")
    print("=" * 60)
    print(f"NextItNet: {'Ready' if nextitnet.is_ready else 'Not Ready'}")
    print(f"LLM: {'Ready' if llm_recommender.is_ready else 'Not Configured'}")
    print(f"Active Model: {app_state['active_model']}")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:{config.port}")
    print(f"Access at: http://localhost:{config.port}")
    print("=" * 60)

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")

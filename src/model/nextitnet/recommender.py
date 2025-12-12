"""
NextItNet Recommender Service
Wrapper for NextItNet model with data management
"""

import torch
from typing import Any
from pathlib import Path

from .model import NextItNet
from ...services.data_manager import DataManager
from ...config.config import Settings


class NextItNetRecommender:
    """NextItNet Recommender Service."""

    def __init__(self, config: Settings):
        self.config = config
        self.model: NextItNet | None = None
        self.data_manager: DataManager | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_ready = False

        self._initialize()

    def _initialize(self):
        """Initialize the recommender."""
        try:
            # Load data manager
            self.data_manager = DataManager(self.config)

            # Model path
            model_path = Path(self.config.models_dir) / "nextitnet" / "best_model.pth"

            if not model_path.exists():
                print(f"⚠️  NextItNet model not found at {model_path}")
                return

            # Load model
            dilation_rates = self.config.nextitnet_dilations

            self.model = NextItNet(
                num_items=self.config.nextitnet_num_items,
                embedding_dim=self.config.nextitnet_embedding_dim,
                num_blocks=self.config.nextitnet_num_blocks,
                kernel_size=self.config.nextitnet_kernel_size,
                dilation_rates=dilation_rates,
                dropout=self.config.nextitnet_dropout,
            ).to(self.device)

            # Load weights
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self._is_ready = True
            print(f"✅ NextItNet loaded on {self.device}")

        except Exception as e:
            print(f"❌ Error initializing NextItNet: {e}")
            import traceback

            traceback.print_exc()
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(
        self, user_id: str, top_k: int = 10, exclude_history: bool = True
    ) -> dict[str, Any]:
        """Get recommendations for a user."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "NextItNet not initialized",
            }

        # Get vocabulary movies from history
        history = self.data_manager.get_user_history(user_id)

        # Fallback if no vocabulary movies in history
        if len(history) < 1:
            return self._get_fallback_recommendations(user_id, top_k)

        input_seq = self.data_manager.prepare_model_input(user_id)
        if input_seq is None:
            return {
                "success": False,
                "recommendations": [],
                "message": "Failed to prepare input",
            }

        input_tensor = torch.LongTensor([input_seq]).to(self.device)
        exclude_items = history if exclude_history else None

        top_items, top_scores = self.model.predict(
            input_tensor, top_k=top_k, exclude_items=exclude_items
        )

        recommendations = []
        for idx, score in zip(top_items[0].tolist(), top_scores[0].tolist()):
            movie_info = self.data_manager.get_movie_by_idx(idx)
            if movie_info:
                rec = {
                    **movie_info,
                    "score": float(score),
                    "internal_idx": idx,
                    "model": "NextItNet",
                }
                recommendations.append(self._sanitize_movie_data(rec))

        return {
            "success": True,
            "recommendations": recommendations,
            "model_used": "nextitnet",
            "message": f"Generated {len(recommendations)} recommendations",
            "user_history_length": len(history),
        }

    def _get_fallback_recommendations(
        self, user_id: str, top_k: int = 10
    ) -> dict[str, Any]:
        """Fallback recommendations when user has no vocabulary movies in history."""
        vocab_movies = []
        for movie_id in self.data_manager.item2idx.keys():
            movie_info = self.data_manager.get_movie_info(movie_id)
            if movie_info and "vote_average" in movie_info:
                vocab_movies.append(movie_info)

        # Sort by rating
        vocab_movies.sort(key=lambda x: x.get("vote_average", 0), reverse=True)

        # Exclude user's watched movies
        user_movie_ids = self.data_manager.user_sessions.get(user_id, [])
        recommendations = []
        for movie in vocab_movies:
            if movie["id"] not in user_movie_ids:
                vote_avg = movie.get("vote_average", 0)
                score = (
                    float(vote_avg)
                    if not (
                        isinstance(vote_avg, float)
                        and (vote_avg != vote_avg or vote_avg == float("inf"))
                    )
                    else 0.0
                )

                rec = {
                    **movie,
                    "score": score,
                    "fallback": True,
                    "model": "NextItNet (Popular)",
                }
                recommendations.append(self._sanitize_movie_data(rec))
                if len(recommendations) >= top_k:
                    break

        return {
            "success": True,
            "recommendations": recommendations,
            "model_used": "nextitnet_fallback",
            "message": "Showing popular movies (no history available)",
        }

    def add_to_history(self, user_id: str, movie_id: int) -> dict[str, Any]:
        """Add movie to user history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        success = self.data_manager.add_to_history(user_id, movie_id)

        if success:
            movie_info = self.data_manager.get_movie_info(movie_id)
            return {
                "success": True,
                "message": "Added to history",
                "movie": movie_info,
                "history_length": len(self.data_manager.get_user_history(user_id)),
            }
        else:
            return {"success": False, "message": f"Movie ID {movie_id} not found"}

    def get_user_history(self, user_id: str) -> dict[str, Any]:
        """Get user's history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        movies = self.data_manager.get_user_history_movies(user_id)
        sanitized_movies = []
        for m in movies:
            if "movie_id" not in m and "id" in m:
                m["movie_id"] = m["id"]
            sanitized_movies.append(self._sanitize_movie_data(m))

        return {
            "success": True,
            "history": sanitized_movies,
            "length": len(sanitized_movies),
        }

    def clear_user_history(self, user_id: str) -> dict[str, Any]:
        """Clear user's history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        self.data_manager.user_sessions[user_id] = []

        return {"success": True, "message": "History cleared"}

    def remove_from_history(self, user_id: str, movie_id: int) -> dict[str, Any]:
        """Remove a specific movie from user's history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        if user_id not in self.data_manager.user_sessions:
            return {"success": False, "message": "User not found"}

        history = self.data_manager.user_sessions[user_id]
        original_length = len(history)
        self.data_manager.user_sessions[user_id] = [
            mid for mid in history if mid != movie_id
        ]
        removed_count = original_length - len(self.data_manager.user_sessions[user_id])

        if removed_count == 0:
            return {"success": False, "message": "Movie not in history"}

        return {
            "success": True,
            "message": f"Removed {removed_count} occurrence(s)",
            "removed_count": removed_count,
            "history_length": len(self.data_manager.user_sessions[user_id]),
        }

    def search_movies(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search movies by title."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        results = self.data_manager.search_movies(query, limit)
        sanitized_results = []
        for r in results:
            if "movie_id" not in r and "id" in r:
                r["movie_id"] = r["id"]
            sanitized_results.append(self._sanitize_movie_data(r))

        return {
            "success": True,
            "query": query,
            "results": sanitized_results,
            "count": len(sanitized_results),
        }

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        """Get movie details."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        movie_info = self.data_manager.get_movie_info(movie_id)

        if movie_info:
            in_vocab = movie_id in self.data_manager.item2idx
            movie_info["movie_id"] = movie_id
            movie_info = self._sanitize_movie_data(movie_info)
            return {"success": True, "movie": movie_info, "in_vocabulary": in_vocab}
        else:
            return {"success": False, "message": f"Movie ID {movie_id} not found"}

    def get_popular_movies(
        self, limit: int = 20, vocab_only: bool = False
    ) -> dict[str, Any]:
        """Get list of popular movies."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        movies = self.data_manager.get_all_movies(limit, vocab_only=vocab_only)

        sanitized_movies = []
        for m in movies:
            if "movie_id" not in m and "id" in m:
                m["movie_id"] = m["id"]
            sanitized_movies.append(self._sanitize_movie_data(m))

        return {
            "success": True,
            "movies": sanitized_movies,
            "count": len(sanitized_movies),
            "total_in_vocabulary": self.data_manager.num_items,
        }

    def _sanitize_movie_data(self, movie_info: dict[str, Any]) -> dict[str, Any]:
        """Sanitize movie data to remove NaN values."""
        import math

        sanitized = {}
        for key, value in movie_info.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

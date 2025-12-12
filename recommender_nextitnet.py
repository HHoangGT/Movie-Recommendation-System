"""
NextItNet Recommender Wrapper for Production

This module wraps the deployment module's NextItNet recommender
for use in the unified production system.
"""

import torch
import torch.nn as nn
from typing import Any

from data_manager import DataManager
from local_config import MODEL_PATH, MODEL_CONFIG


class ResidualBlock(nn.Module):
    """Residual block with dilated causal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = out[:, : x.size(1), :]
        out = self.ln(out)
        out = self.relu(out)
        return out + x if out.size(-1) == x.size(-1) else out


class NextItNet(nn.Module):
    """NextItNet model for sequential recommendations."""

    def __init__(
        self,
        num_items,
        embedding_dim=128,
        num_blocks=6,
        kernel_size=3,
        dilation_rates=None,
        dropout=0.2,
    ):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 1, 2, 4]

        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(embedding_dim, embedding_dim, kernel_size, d)
                for d in dilation_rates
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, num_items)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)
        return self.output_layer(x)

    def predict(self, input_seq, top_k=10, exclude_items=None):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_seq)
            last_logits = logits[:, -1, :]

            if exclude_items is not None:
                for item in exclude_items:
                    if 0 <= item < self.num_items:
                        last_logits[:, item] = float("-inf")

            last_logits[:, 0] = float("-inf")

            scores = torch.softmax(last_logits, dim=-1)
            top_scores, top_items = torch.topk(scores, top_k, dim=-1)

        return top_items, top_scores


class NextItNetRecommender:
    """NextItNet Recommender Service."""

    def __init__(self):
        self.model: NextItNet | None = None
        self.data_manager: DataManager | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_ready = False

        self._initialize()

    def _initialize(self):
        """Initialize the recommender."""
        try:
            # Load data manager
            self.data_manager = DataManager()

            # Load model
            dilation_rates = MODEL_CONFIG.get("dilations", [1, 2, 4, 1, 2, 4])

            self.model = NextItNet(
                num_items=MODEL_CONFIG["num_items"],
                embedding_dim=MODEL_CONFIG["embedding_dim"],
                num_blocks=MODEL_CONFIG["num_blocks"],
                kernel_size=MODEL_CONFIG["kernel_size"],
                dilation_rates=dilation_rates,
                dropout=MODEL_CONFIG["dropout"],
            ).to(self.device)

            # Load weights
            checkpoint = torch.load(
                MODEL_PATH, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self._is_ready = True
            print(f"NextItNet loaded on {self.device}")

        except Exception as e:
            print(f"Error initializing NextItNet: {e}")
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(
        self, user_id: str, top_k: int = 10, exclude_history: bool = True
    ) -> dict[str, Any]:
        """Get recommendations for a user. Falls back to popular movies if no vocabulary movies in history."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "NextItNet not initialized",
            }

        # Get vocabulary movies from history (filters out non-vocab)
        history = self.data_manager.get_user_history(user_id)

        # FALLBACK: If no vocabulary movies in history, recommend popular movies
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
                rec = {**movie_info, "score": float(score), "internal_idx": idx}
                recommendations.append(rec)

        return {
            "success": True,
            "recommendations": recommendations,
            "message": f"Generated {len(recommendations)} recommendations",
            "user_history_length": len(history),
        }

    def _get_fallback_recommendations(
        self, user_id: str, top_k: int = 10
    ) -> dict[str, Any]:
        """
        Fallback recommendations when user has no vocabulary movies in history.
        Returns popular/highly-rated vocabulary movies.
        """
        # Get all vocabulary movies sorted by rating
        vocab_movies = []
        for movie_id in self.data_manager.item2idx.keys():
            movie_info = self.data_manager.get_movie_info(movie_id)
            if movie_info and "vote_average" in movie_info:
                vocab_movies.append(movie_info)

        # Sort by vote_average (highest first)
        vocab_movies.sort(key=lambda x: x.get("vote_average", 0), reverse=True)

        # Get top_k (excluding any non-vocab movies the user watched)
        user_movie_ids = self.data_manager.user_sessions.get(user_id, [])
        recommendations = []
        for movie in vocab_movies:
            if (
                movie["id"] not in user_movie_ids
            ):  # Don't recommend what they already watched
                # Sanitize float values (NaN -> None for JSON)
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
                    "fallback": True,  # Mark as fallback recommendation
                }
                # Sanitize the entire recommendation
                recommendations.append(self._sanitize_movie_data(rec))
                if len(recommendations) >= top_k:
                    break

        return {
            "success": True,
            "recommendations": recommendations,
            "model_used": "nextitnet_fallback",
            "message": "Showing popular movies (only non-vocabulary movies in your history)",
        }

    def get_recommendations_for_sequence(
        self, movie_ids: list[int], top_k: int = 10, exclude_input: bool = True
    ) -> dict[str, Any]:
        """Get recommendations for a sequence of movies."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "NextItNet not initialized",
            }

        indices = []
        for movie_id in movie_ids:
            idx = self.data_manager.movie_id_to_idx(movie_id)
            if idx is not None:
                indices.append(idx)

        if len(indices) < 1:
            return {
                "success": False,
                "recommendations": [],
                "message": "Need at least 1 valid movie",
            }

        max_len = self.data_manager.max_seq_length
        if len(indices) < max_len:
            padded = [0] * (max_len - len(indices)) + indices
        else:
            padded = indices[-max_len:]

        input_tensor = torch.LongTensor([padded]).to(self.device)
        exclude_items = indices if exclude_input else None

        top_items, top_scores = self.model.predict(
            input_tensor, top_k=top_k, exclude_items=exclude_items
        )

        recommendations = []
        for idx, score in zip(top_items[0].tolist(), top_scores[0].tolist()):
            movie_info = self.data_manager.get_movie_by_idx(idx)
            if movie_info:
                rec = {**movie_info, "score": float(score), "internal_idx": idx}
                recommendations.append(rec)

        return {
            "success": True,
            "recommendations": recommendations,
            "message": f"Generated {len(recommendations)} recommendations",
            "input_length": len(indices),
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

    def add_multiple_to_history(
        self, user_id: str, movie_ids: list[int]
    ) -> dict[str, Any]:
        """Add multiple movies to history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        count = self.data_manager.add_multiple_to_history(user_id, movie_ids)

        return {
            "success": True,
            "message": f"Added {count} of {len(movie_ids)} movies",
            "added_count": count,
            "history_length": len(self.data_manager.get_user_history(user_id)),
        }

    def get_user_history(self, user_id: str) -> dict[str, Any]:
        """Get user's history."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        movies = self.data_manager.get_user_history_movies(user_id)
        # Ensure movie_id is in each result and sanitize
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

        # Clear history or create empty session if user doesn't exist
        if user_id in self.data_manager.user_sessions:
            self.data_manager.user_sessions[user_id] = []
        else:
            # Create empty session for new user
            self.data_manager.user_sessions[user_id] = []

        return {"success": True, "message": "History cleared"}

    def remove_from_history(self, user_id: str, movie_id: int) -> dict[str, Any]:
        """Remove a specific movie from user's history (works for any movie, even non-vocab)."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        if user_id not in self.data_manager.user_sessions:
            return {"success": False, "message": "User not found"}

        # Since user_sessions now stores movie_ids (not indices), remove directly
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
        # Ensure movie_id is in each result and sanitize
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

    def _sanitize_movie_data(self, movie_info: dict[str, Any]) -> dict[str, Any]:
        """Sanitize movie data to remove NaN and None values."""
        import math

        sanitized = {}
        for key, value in movie_info.items():
            if isinstance(value, float) and math.isnan(value):
                sanitized[key] = None
            elif value is None:
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        """Get movie details."""
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        movie_info = self.data_manager.get_movie_info(movie_id)

        if movie_info:
            in_vocab = movie_id in self.data_manager.item2idx
            # Add movie_id to the info
            movie_info["movie_id"] = movie_id
            # Sanitize data to remove NaN values
            movie_info = self._sanitize_movie_data(movie_info)
            return {"success": True, "movie": movie_info, "in_vocabulary": in_vocab}
        else:
            return {"success": False, "message": f"Movie ID {movie_id} not found"}

    def get_popular_movies(
        self, limit: int = None, vocab_only: bool = False
    ) -> dict[str, Any]:
        """
        Get list of movies.

        Args:
            limit: Maximum number of movies to return (None = all movies)
            vocab_only: If True, only return movies in vocabulary
        """
        if not self.is_ready:
            return {"success": False, "message": "Not initialized"}

        # If no limit specified, get ALL movies from dataset (9,066 movies)
        if limit is None:
            limit = 10000  # Increased to accommodate all movies

        movies = self.data_manager.get_all_movies(limit, vocab_only=vocab_only)

        # Ensure movie_id is in each result and sanitize
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

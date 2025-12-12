"""
BiVAE Recommender Service
Bilateral Variational Autoencoder for Collaborative Filtering using Cornac
"""

import cornac
from typing import Any
from pathlib import Path

from ...services.data_manager import DataManager
from ...config.config import Settings


class BiVAERecommender:
    """BiVAE Recommender Service using Cornac."""

    def __init__(self, config: Settings):
        self.config = config
        self.model: cornac.models.BiVAECF | None = None
        self.data_manager: DataManager | None = None
        self._is_ready = False

        self._initialize()

    def _initialize(self):
        """Initialize DataManager and load BiVAE model."""
        try:
            print("🔄 Loading BiVAE model...")
            self.data_manager = DataManager(self.config)

            # Path to saved model
            model_path = Path(self.config.models_dir) / "bivae" / "BiVAECF"

            if model_path.exists():
                # Load BiVAE model
                self.model = cornac.models.BiVAECF.load(str(model_path))
                self._is_ready = True
                print(f"✅ BiVAE model loaded from {model_path}")
            else:
                print(f"⚠️  BiVAE model not found at {model_path}")
                print("   Run training script to train BiVAE model first.")

        except Exception as e:
            print(f"❌ Error initializing BiVAE: {e}")
            import traceback

            traceback.print_exc()
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(self, user_id: str, top_k: int = 10) -> dict[str, Any]:
        """Get movie recommendations for user using BiVAE."""
        if not self.is_ready:
            return {
                "success": False,
                "recommendations": [],
                "message": "BiVAE model not loaded",
            }

        if self.model.train_set is None:
            return {
                "success": False,
                "recommendations": [],
                "message": "Model train_set is missing",
            }

        uid_map = self.model.train_set.uid_map

        # Map user_id to internal index
        if user_id in uid_map:
            u_idx = uid_map[user_id]
        else:
            # Fallback for new users: use dummy user '-1' if available
            u_idx = uid_map.get("-1", 0)

        # Get rankings from model
        item_indices, scores = self.model.rank(u_idx)

        # Get top-K
        top_indices = item_indices[:top_k]
        top_scores = scores[:top_k]

        recommendations = []
        for idx, score in zip(top_indices, top_scores):
            movie_info = self.data_manager.get_movie_by_idx(int(idx))
            if movie_info:
                rec = {**movie_info, "score": float(score), "model": "BiVAE"}
                recommendations.append(self._sanitize_movie_data(rec))

        return {
            "success": True,
            "recommendations": recommendations,
            "model_used": "bivae",
            "message": f"Generated {len(recommendations)} recommendations using BiVAE",
        }

    def _sanitize_movie_data(self, movie_info: dict[str, Any]) -> dict[str, Any]:
        """Sanitize movie data to avoid JSON errors (NaN, Infinity)."""
        import math

        sanitized = {}
        for key, value in movie_info.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

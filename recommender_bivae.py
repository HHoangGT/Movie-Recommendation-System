"""
BiVAE Recommender Wrapper
"""

import os
import cornac
from typing import Any

from data_manager import DataManager
from local_config import MODELS_DIR


class BiVAERecommender:
    def __init__(self):
        self.model = None
        self.data_manager = None
        self._is_ready = False
        self._initialize()

    def _initialize(self):
        """Khởi tạo DataManager và load model BiVAE đã train."""
        try:
            print("🔄 Loading BiVAE model...")
            self.data_manager = DataManager()

            # Đường dẫn tới thư mục model đã lưu
            model_path = os.path.join(MODELS_DIR, "bivae_context/BiVAECF")
            print(model_path, os.path.exists(model_path))

            if os.path.exists(model_path):
                # Load model BiVAE từ thư mục
                self.model = cornac.models.BiVAECF.load(model_path)
                self._is_ready = True
                print("✅ BiVAE model loaded successfully!")
            else:
                print(
                    f"❌ BiVAE model not found at {model_path}. Please run train_bivae.py first."
                )

        except Exception as e:
            print(f"❌ Error initializing BiVAE: {e}")
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(self, user_id: str, top_k: int = 10) -> dict[str, Any]:
        """Lấy gợi ý phim cho user dựa trên model BiVAE."""
        if not self.is_ready:
            return {"success": False, "message": "BiVAE model not loaded"}

        if self.model.train_set is None:
            return {"success": False, "message": "Model train_set is missing"}

        uid_map = self.model.train_set.uid_map

        if user_id in uid_map:
            u_idx = uid_map[user_id]
        else:
            # Fallback cho user mới: dùng user ảo '-1' đã tạo lúc train
            u_idx = uid_map.get("-1", 0)

        item_indices, scores = self.model.rank(u_idx)

        # Lấy Top-K
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
        """Xử lý dữ liệu để tránh lỗi JSON (NaN, Infinity)."""
        import math

        sanitized = {}
        for key, value in movie_info.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

"""
BiVAE Recommender Wrapper
"""

import os
import glob
import pickle
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

    # def _initialize(self):
    #     """Khởi tạo DataManager và load model BiVAE đã train."""
    #     try:
    #         print("🔄 Loading BiVAE model...")
    #         self.data_manager = DataManager()

    #         # Đường dẫn tới thư mục model đã lưu
    #         model_path = os.path.join(MODELS_DIR, "bivae_context/BiVAECF")

    #         if os.path.exists(model_path):
    #             # Load model BiVAE từ thư mục
    #             self.model = cornac.models.BiVAECF.load(model_path)
    #             self._is_ready = True
    #             print("✅ BiVAE model loaded successfully!")
    #         else:
    #             print(
    #                 f"❌ BiVAE model not found at {model_path}. Please run train_bivae.py first."
    #             )

    #     except Exception as e:
    #         print(f"❌ Error initializing BiVAE: {e}")
    #         self._is_ready = False
    def _initialize(self):
        """Khởi tạo DataManager và load model BiVAE đã train."""
        try:
            print("🔄 Loading BiVAE model...")
            self.data_manager = DataManager()

            # Đường dẫn tới thư mục chứa model
            base_path = os.path.join(MODELS_DIR, "bivae_context/BiVAECF")
            model_file = None

            # Logic tìm file .pkl mới nhất trong thư mục
            if os.path.isdir(base_path):
                pkl_files = glob.glob(os.path.join(base_path, "*.pkl"))
                if pkl_files:
                    # Lấy file mới nhất dựa trên thời gian sửa đổi
                    model_file = max(pkl_files, key=os.path.getmtime)
                    print(f"Found model file: {os.path.basename(model_file)}")
                else:
                    print(f"No .pkl model files found in {base_path}")
            elif os.path.exists(base_path):
                # Fallback nếu base_path trỏ trực tiếp vào file
                model_file = base_path

            if model_file and os.path.exists(model_file):
                # Load model BiVAE từ file cụ thể
                self.model = cornac.models.BiVAECF.load(model_file)

                # Fix: Kiểm tra và load thủ công train_set nếu Cornac không tự load
                # Cornac thường tự load file .trainset nếu nó nằm cùng thư mục và có tên đúng
                if not hasattr(self.model, "train_set") or self.model.train_set is None:
                    trainset_path = model_file + ".trainset"
                    if os.path.exists(trainset_path):
                        # print("'train_set' attribute missing. Attempting manual load...")
                        with open(trainset_path, "rb") as f:
                            self.model.train_set = pickle.load(f)
                        # print("Trainset loaded manually.")
                    else:
                        print(f"Warning: Trainset file not found at {trainset_path}")

                self._is_ready = True
                print("BiVAE model loaded successfully!")
            else:
                print(
                    f"BiVAE model file not found at {base_path}. Please run train_bivae.py first."
                )

        except Exception as e:
            print(f"Error initializing BiVAE: {e}")
            import traceback

            traceback.print_exc()
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_recommendations(self, user_id: str, top_k: int = 10) -> dict[str, Any]:
        """Get recommendations using BiVAE, falling back to popularity for new users."""
        if not self.is_ready:
            return {"success": False, "message": "BiVAE model not loaded"}

        # Safety check for train_set
        if self.model.train_set is None:
            print("BiVAE train_set is missing. Falling back to popular.")
            return self._get_popular_fallback(top_k, "BiVAE model incomplete")

        # Access mappings
        uid_map = self.model.train_set.uid_map

        # --- [NEW USER LOGIC] ---
        # If user is not in the training data, return popular movies
        if user_id not in uid_map:
            print(f"User '{user_id}' is new. Returning popular movies.")
            return self._get_popular_fallback(
                top_k, "New user - Showing popular movies"
            )

        # --- [EXISTING USER LOGIC] ---
        try:
            u_idx = uid_map[user_id]

            # Rank items for this user (returns all items sorted)
            # This returns internal indices
            item_indices, scores = self.model.rank(u_idx)

            # Get Top-K
            top_indices = item_indices[:top_k]
            top_scores = scores[:top_k]

            recommendations = []
            for idx, score in zip(top_indices, top_scores):
                # We need to convert Cornac internal index -> Original Item ID
                # Since we trained with 'all_items' logic, the indices usually align,
                # but to be safe we use the train_set map if possible.
                # In your setup, DataManager indices align with Cornac indices.
                movie_info = self.data_manager.get_movie_by_idx(int(idx))

                if movie_info:
                    rec = {**movie_info, "score": float(score), "model": "BiVAE"}
                    recommendations.append(self._sanitize_movie_data(rec))

            return {
                "success": True,
                "recommendations": recommendations,
                "model_used": "bivae",
                "message": f"Generated {len(recommendations)} personal recommendations",
            }
        except Exception as e:
            print(f"Error during BiVAE inference: {e}")
            return self._get_popular_fallback(top_k, "Error in prediction")

    def _get_popular_fallback(self, top_k: int, reason: str) -> dict[str, Any]:
        """Return popular movies based on vote_average from metadata."""
        # 1. Get all movies that have a rating
        all_movies = []
        for mid, info in self.data_manager.movie_info.items():
            if info.get("vote_average") is not None:
                all_movies.append(info)

        # 2. Sort by rating (descending)
        # We use a simple heuristic: vote_average.
        # (Optional: You could filter for vote_count > 50 if you have that data)
        all_movies.sort(
            key=lambda x: float(x.get("vote_average", 0) or 0), reverse=True
        )

        # 3. Take Top K
        recommendations = []
        for movie in all_movies[:top_k]:
            rec = {
                **movie,
                "score": float(movie.get("vote_average", 0)),
                "model": "popular_fallback",
            }
            recommendations.append(self._sanitize_movie_data(rec))

        return {
            "success": True,
            "recommendations": recommendations,
            "model_used": "bivae_popular",
            "message": reason,
        }

    def _sanitize_movie_data(self, movie_info: dict[str, Any]) -> dict[str, Any]:
        """Handle NaN/Inf values for JSON safety."""
        import math

        sanitized = {}
        for key, value in movie_info.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                sanitized[key] = None
            else:
                sanitized[key] = value
        return sanitized

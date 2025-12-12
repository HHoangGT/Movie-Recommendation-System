"""
Data Manager for Movie Recommendation System (Production Version)

Handles loading movie metadata, vocabulary mappings, and user session management.
All data loaded from local production folder.
"""

import os
import json
import pickle
import pandas as pd
from typing import Any
from collections import defaultdict

from local_config import VOCAB_PATH, MOVIES_METADATA_PATH, MAX_HISTORY_LENGTH


class DataManager:
    """
    Manages movie data, vocabulary mappings, and user sessions.

    This class provides:
    - Movie metadata loading and lookup
    - Item ID to Movie ID mapping (and vice versa)
    - User session management (in-memory)
    - History preprocessing for model input
    """

    def __init__(self, vocab_path: str = None, movies_path: str = None):
        """
        Initialize the DataManager.

        Args:
            vocab_path: Path to vocabulary pickle file
            movies_path: Path to movies metadata CSV
        """
        self.vocab_path = str(vocab_path or VOCAB_PATH)
        self.movies_path = str(movies_path or MOVIES_METADATA_PATH)

        # Vocabulary mappings
        self.item2idx: dict[int, int] = {}  # movie_id -> internal_idx
        self.idx2item: dict[int, int] = {}  # internal_idx -> movie_id
        self.num_items: int = 0
        self.max_seq_length: int = MAX_HISTORY_LENGTH

        # Movie metadata
        self.movies_df: pd.DataFrame | None = None
        # movie_id -> info dict
        self.movie_info: dict[int, dict[str, Any]] = {}
        # Total movies in metadata (not just vocab)
        self.num_total_movies: int = 0

        # User sessions (in-memory storage)
        # Format: {user_id: [list of movie_ids]} (NOT indices, to support non-vocab movies)
        self.user_sessions: dict[str, list[int]] = defaultdict(list)

        # Load data
        self._load_vocabulary()
        self._load_movies()

    def _load_vocabulary(self):
        """Load vocabulary mappings from pickle file."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with open(self.vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.item2idx = vocab["item2idx"]
        self.idx2item = vocab["idx2item"]
        self.num_items = vocab["num_items"]
        self.max_seq_length = vocab.get("max_seq_length", MAX_HISTORY_LENGTH)

        print(
            f"Loaded vocabulary: {self.num_items} items, max_seq_length={self.max_seq_length}"
        )

    def _load_movies(self):
        """Load movie metadata from CSV."""
        if not os.path.exists(self.movies_path):
            print(f"Warning: Movies metadata not found: {self.movies_path}")
            return

        try:
            self.movies_df = pd.read_csv(self.movies_path, low_memory=False)

            # Clean and process movie data
            self.movies_df["id"] = pd.to_numeric(self.movies_df["id"], errors="coerce")
            self.movies_df = self.movies_df.dropna(subset=["id"])
            self.movies_df["id"] = self.movies_df["id"].astype(int)

            # Parse genres
            def parse_genres(genres_str):
                try:
                    genres_list = json.loads(genres_str.replace("'", '"'))
                    return [g["name"] for g in genres_list]
                except Exception:
                    return []

            if "genres" in self.movies_df.columns:
                self.movies_df["genres_list"] = self.movies_df["genres"].apply(
                    parse_genres
                )

            # Build movie info dictionary
            for _, row in self.movies_df.iterrows():
                movie_id = int(row["id"])
                self.movie_info[movie_id] = {
                    "id": movie_id,
                    "title": row.get("title", f"Movie {movie_id}"),
                    "genres": row.get("genres_list", [])
                    if "genres_list" in row
                    else [],
                    "overview": row.get("overview", ""),
                    "vote_average": row.get("vote_average", 0.0),
                    "release_date": row.get("release_date", ""),
                    "poster_path": row.get("poster_path", ""),
                }

            self.num_total_movies = len(self.movie_info)
            print(f"Loaded {self.num_total_movies} movies from metadata")

        except Exception as e:
            print(f"Error loading movies: {e}")

    def movie_id_to_idx(self, movie_id: int) -> int | None:
        """Convert movie ID to internal item index."""
        return self.item2idx.get(movie_id)

    def idx_to_movie_id(self, idx: int) -> int | None:
        """Convert internal item index to movie ID."""
        return self.idx2item.get(idx)

    def get_movie_info(self, movie_id: int) -> dict[str, Any] | None:
        """Get movie information by movie ID."""
        return self.movie_info.get(movie_id)

    def get_movie_by_idx(self, idx: int) -> dict[str, Any] | None:
        """Get movie information by internal index."""
        movie_id = self.idx_to_movie_id(idx)
        if movie_id:
            return self.get_movie_info(movie_id)
        return None

    def search_movies(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search movies by title.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of movie info dictionaries
        """
        query_lower = query.lower()
        results = []

        for movie_id, info in self.movie_info.items():
            title = info.get("title", "")
            # Handle NaN or non-string titles
            if not isinstance(title, str):
                continue
            if query_lower in title.lower():
                # Only include movies that are in our vocabulary
                if movie_id in self.item2idx:
                    results.append(info)
                    if len(results) >= limit:
                        break

        return results

    def get_all_movies(
        self, limit: int = None, vocab_only: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get all movies from the dataset.

        Args:
            limit: Maximum number of movies to return
            vocab_only: If True, only return movies in the vocabulary (used by model)

        Returns:
            List of movie info dictionaries
        """
        movies = []

        if vocab_only:
            # Only return movies in the vocabulary (can be used for recommendations)
            for movie_id in self.item2idx.keys():
                info = self.get_movie_info(movie_id)
                if info:
                    info["in_vocabulary"] = True
                    movies.append(info)
                    if limit and len(movies) >= limit:
                        break
        else:
            # Return ALL movies from metadata (including ones not in vocabulary)
            for movie_id, info in self.movie_info.items():
                movie_copy = info.copy()
                movie_copy["in_vocabulary"] = movie_id in self.item2idx
                movies.append(movie_copy)
                if limit and len(movies) >= limit:
                    break

        return movies

    # ==================== User Session Management ====================

    def add_to_history(self, user_id: str, movie_id: int) -> bool:
        """
        Add a movie to user's viewing history.
        Accepts ANY movie that exists in metadata (not just vocabulary).
        Non-vocabulary movies can be watched but won't affect recommendations.

        Args:
            user_id: User identifier
            movie_id: Movie ID to add

        Returns:
            True if added successfully, False if movie not found in metadata
        """
        # Check if movie exists in metadata (not just vocabulary)
        if movie_id not in self.movie_info:
            return False

        # Create session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []

        # Store movie_id directly (not index) to support non-vocabulary movies
        history = self.user_sessions[user_id]
        if not history or history[-1] != movie_id:
            history.append(movie_id)

            # Trim to max length
            if len(history) > self.max_seq_length:
                self.user_sessions[user_id] = history[-self.max_seq_length :]

        return True

    def get_user_history(self, user_id: str) -> list[int]:
        """
        Get user's viewing history as internal indices.
        Filters out non-vocabulary movies (they're in history but won't affect recommendations).

        Args:
            user_id: User identifier

        Returns:
            List of internal item indices (only vocabulary movies)
        """
        movie_ids = self.user_sessions.get(user_id, [])
        # Convert movie_ids to indices, filtering out non-vocabulary movies
        indices = []
        for movie_id in movie_ids:
            idx = self.movie_id_to_idx(movie_id)
            if idx is not None:  # Only include vocabulary movies for model input
                indices.append(idx)
        return indices

    def get_user_history_movies(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get user's viewing history as movie info dictionaries.
        Includes ALL movies (even non-vocabulary ones).

        Args:
            user_id: User identifier

        Returns:
            List of movie info dictionaries
        """
        movie_ids = self.user_sessions.get(user_id, [])
        movies = []
        for movie_id in movie_ids:
            movie_info = self.get_movie_info(movie_id)
            if movie_info:
                # Add in_vocabulary flag
                movie_info["in_vocabulary"] = movie_id in self.item2idx
                movies.append(movie_info)
        return movies

    def clear_user_history(self, user_id: str) -> bool:
        """
        Clear a user's viewing history.

        Args:
            user_id: User identifier

        Returns:
            True if cleared, False if user not found
        """
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = []
            return True
        return False

    def prepare_model_input(self, user_id: str) -> list[int] | None:
        """
        Prepare user history for model input.

        Pads the sequence to max_seq_length with zeros at the beginning.

        Args:
            user_id: User identifier

        Returns:
            Padded sequence of item indices, or None if no history
        """
        history = self.get_user_history(user_id)

        if not history:
            return None

        # Pad with zeros at the beginning
        if len(history) < self.max_seq_length:
            padded = [0] * (self.max_seq_length - len(history)) + history
        else:
            padded = history[-self.max_seq_length :]

        return padded

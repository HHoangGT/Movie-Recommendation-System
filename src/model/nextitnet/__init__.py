"""
NextItNet Model Package
Sequential recommendation using dilated causal convolutions
"""

from .model import NextItNet, ResidualBlock
from .recommender import NextItNetRecommender

__all__ = ["NextItNet", "ResidualBlock", "NextItNetRecommender"]

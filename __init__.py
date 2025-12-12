"""
Production Deployment Package

Unified Movie Recommendation System combining:
- NextItNet: Sequential recommendation model
- LLM (Gemini): Content-based AI recommendations
"""

from .config import Config

__all__ = ["Config"]

from pydantic import BaseModel

from src.config import get_config

settings = get_config()


class Message(BaseModel):
    detail: str


class LLMInput(BaseModel):
    movie_name: str
    movie_genre: str
    movie_overview: str

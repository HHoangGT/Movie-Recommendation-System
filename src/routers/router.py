from loguru import logger
from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_config
from src.model.llm.inference import movie_recommendation_inference
from . import schema

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/recommend/llm",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": schema.Message},
        status.HTTP_401_UNAUTHORIZED: {"model": schema.Message},
        status.HTTP_403_FORBIDDEN: {"model": schema.Message},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": schema.Message},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schema.Message},
    },
)
@limiter.limit(f"{get_config().limiter_requests}/minute")
async def analyze(
    request: Request,
    input_data: schema.LLMInput,
):
    try:
        result = await movie_recommendation_inference(
            movie_name=input_data.movie_name,
            movie_genre=input_data.movie_genre,
            movie_overview=input_data.movie_overview,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)},
        )

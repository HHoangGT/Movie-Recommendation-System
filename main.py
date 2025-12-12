"""
Movie Recommendation System - FastAPI Application
Unified system with NextItNet, BiVAE, and LLM (DSPy) recommendations
"""

import uvicorn
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import DefaultFormatter, AccessFormatter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

from src.config import get_config
from src.model.nextitnet import NextItNetRecommender
from src.model.bivae import BiVAERecommender
from src.model.llm import LLMRecommender
from src.routers import router
from src.routers.router import init_recommenders

# Get configuration
settings = get_config()

# Setup paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    limiter = Limiter(key_func=get_remote_address)

    application = FastAPI(
        title=settings.project_name,
        description="Unified Movie Recommendation System with NextItNet, BiVAE, and LLM",
        version="2.0.0",
    )

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        application.mount(
            "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
        )
    else:
        print(f"⚠️  Static directory not found: {STATIC_DIR}")

    # Initialize recommenders
    print("\n" + "=" * 60)
    print("🚀 Initializing Movie Recommendation System")
    print("=" * 60)

    print("\n📊 Initializing NextItNet...")
    nextitnet_recommender = NextItNetRecommender(settings)

    print("\n📊 Initializing BiVAE...")
    bivae_recommender = BiVAERecommender(settings)

    print("\n📊 Initializing LLM Recommender (DSPy)...")
    llm_recommender = LLMRecommender(settings)

    # Initialize router with recommenders
    init_recommenders(nextitnet_recommender, bivae_recommender, llm_recommender)

    # Include router
    application.include_router(router, prefix="/api")

    print("\n" + "=" * 60)
    print("✅ System Initialization Complete")
    print("=" * 60)
    print(
        f"   NextItNet: {'✅ Ready' if nextitnet_recommender.is_ready else '❌ Not Ready'}"
    )
    print(
        f"   BiVAE:     {'✅ Ready' if bivae_recommender.is_ready else '❌ Not Ready'}"
    )
    print(f"   LLM:       {'✅ Ready' if llm_recommender.is_ready else '❌ Not Ready'}")
    print("=" * 60 + "\n")

    # Frontend routes
    @application.get("/")
    async def index():
        """Serve the main web application."""
        index_path = TEMPLATES_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Frontend not available. Check templates directory."}

    @application.get("/login")
    async def login_page():
        """Serve the login page."""
        login_path = TEMPLATES_DIR / "login.html"
        if login_path.exists():
            return FileResponse(str(login_path))
        return {"message": "Login page not available."}

    return application


app = create_app()


# Custom log formatters with Ho Chi Minh City timezone
class HoChiMinhFormatter:
    """Base formatter with Ho Chi Minh City timezone (UTC+7)"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Ho_Chi_Minh"))
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


class HoChiMinhDefaultFormatter(HoChiMinhFormatter, DefaultFormatter):
    pass


class HoChiMinhAccessFormatter(HoChiMinhFormatter, AccessFormatter):
    pass


if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["default"] = {
        "()": HoChiMinhDefaultFormatter,
        "fmt": "%(asctime)s %(levelprefix)s %(message)s",
    }
    LOGGING_CONFIG["formatters"]["access"] = {
        "()": HoChiMinhAccessFormatter,
        "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
    }

    uvicorn.run(
        "main:app", host=settings.host, port=settings.port_number, reload=settings.debug
    )

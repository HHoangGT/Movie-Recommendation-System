from datetime import datetime
from zoneinfo import ZoneInfo

import uvicorn
from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import DefaultFormatter, AccessFormatter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import get_config
from src.routers import router

settings = get_config()


def create_app() -> FastAPI:
    limiter = Limiter(key_func=get_remote_address)

    application = FastAPI(title=settings.project_name)

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routers
    application.include_router(router)

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

    uvicorn.run(app, host="0.0.0.0", port=settings.port_number)

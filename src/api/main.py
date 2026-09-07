"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.core.config import settings
from src.core.db import init_db

log = logger.bind(name="API")

STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    log.info("Database initialized")
    yield


def create_app() -> FastAPI:
    if not settings.secret_key:
        raise RuntimeError(
            "SECRET_KEY is not set in .env — the app refuses to start without it "
            "(it's required to encrypt stored API keys). Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    app = FastAPI(title="AquaOracle API", lifespan=_lifespan)

    from src.api.routes import auth as auth_routes
    from src.api.routes import chats as chats_routes
    from src.api.routes import documents as documents_routes
    from src.api.routes import settings as settings_routes

    app.include_router(auth_routes.router)
    app.include_router(chats_routes.router)
    app.include_router(documents_routes.router)
    app.include_router(settings_routes.router)

    @app.get("/", include_in_schema=False)
    async def root():
        # No index.html — auth.js itself redirects to /app.html if already
        # logged in, so sending everyone through /auth.html first is safe.
        return RedirectResponse(url="/auth.html")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:
        log.warning(f"Static frontend dir not found at {STATIC_DIR} — API-only mode")

    return app

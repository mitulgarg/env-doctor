"""FastAPI application for the env-doctor dashboard."""
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import database as _db
from .auth import require_token
from .routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB on startup."""
    await _db.init_db()
    yield


app = FastAPI(title="env-doctor dashboard", lifespan=lifespan)


def _cors_origins() -> list[str]:
    """Resolve CORS origins from ENV_DOCTOR_CORS_ORIGINS (comma-separated)."""
    raw = os.environ.get("ENV_DOCTOR_CORS_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All /api/* routes require the shared bearer token (see server/auth.py).
app.include_router(api_router, prefix="/api", dependencies=[Depends(require_token)])

# Serve React static files (built output)
_WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
_WEB_DIR = os.path.normpath(_WEB_DIR)

if os.path.isdir(_WEB_DIR):
    # Mount /assets for JS/CSS bundles
    _ASSETS_DIR = os.path.join(_WEB_DIR, "assets")
    if os.path.isdir(_ASSETS_DIR):
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA — all non-API routes return index.html."""
        # Try to serve the exact file first (e.g., favicon.ico, vite.svg)
        file_path = os.path.join(_WEB_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html for client-side routing
        index = os.path.join(_WEB_DIR, "index.html")
        if os.path.isfile(index):
            return FileResponse(index)
        return {"detail": "Frontend not built. Run 'npm run build' in web/ directory."}
else:
    @app.get("/")
    async def no_frontend():
        return {
            "detail": "Frontend not built. Run 'npm run build' in web/ directory.",
            "api_docs": "/docs",
        }

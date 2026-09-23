"""
main.py
-------
Entry point for the Autonomous Data Quality Investigation Agent.

Starts the FastAPI application, serves static frontend assets, and registers all routers.
Run with:
    python main.py
    -- or --
    uvicorn main:app --reload
"""

from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.utils.config import settings

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # --- Startup ---
    print(
        f"[startup] Autonomous Data Quality Investigation Agent — "
        f"Phase 11 | env={settings.APP_ENV} | "
        f"LLM configured={settings.is_llm_configured()}"
    )
    yield
    # --- Shutdown ---
    print("[shutdown] Application stopped.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Data Quality Investigation Agent",
    description=(
        "An agentic AI system that detects data-quality problems, investigates "
        "root causes, recommends corrective actions, validates results, and "
        "generates investigation reports via LangGraph orchestration."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Register API routers
app.include_router(router)

# ---------------------------------------------------------------------------
# Frontend Static Files & Dashboard Route
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", tags=["Frontend"], include_in_schema=False)
    async def serve_frontend():
        """Serve the Data Quality Investigation Web Dashboard."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

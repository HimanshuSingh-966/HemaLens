"""
HemaLens — FastAPI Backend
==========================
Run development server:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from api.routes import router
from ml.config import API_HOST, API_PORT


def _cors_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


# ── LIFESPAN (model warm-up) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load specialist models on startup so first request isn't slow."""
    print("HemaLens API starting up...")
    try:
        from ml.specialist_inference import load_all_specialists
        specialists = load_all_specialists()
        if specialists:
            names = ", ".join(sorted(specialists.keys()))
            print(f"Specialist models loaded successfully: {names}")
        else:
            print("No specialist models found. Run `python ml/train_specialists.py` first.")
    except Exception as exc:
        print(f"Specialist model warm-up failed: {exc}")
    yield
    print("HemaLens API shutting down.")


# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="HemaLens Blood Report Analysis API",
    description=(
        "AI-powered blood report analysis.\n\n"
        "- **POST /api/v1/analyze/file** — Upload a PDF/image/txt lab report\n"
        "- **POST /api/v1/analyze/params** — Submit parameters as JSON\n"
        "- **POST /api/v1/extract** — NLP extraction only (no diagnosis)\n"
        "- **GET  /api/v1/health** — Health check\n"
        "- **GET  /api/v1/reference-ranges** — All normal ranges\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTES ───────────────────────────────────────────────
app.include_router(router, prefix="/api/v1", tags=["Analysis"])

# ── SERVE UI (optional) ──────────────────────────────────
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "dist")
if os.path.isdir(UI_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(UI_DIR, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_ui(full_path: str = ""):
        # If it's an API route, let it pass through (handled by prefixes)
        if full_path.startswith("api/"):
            return
        return FileResponse(os.path.join(UI_DIR, "index.html"))


# ── ROOT ─────────────────────────────────────────────────
@app.get("/ping", tags=["Meta"])
def ping():
    return {"pong": True, "version": "2.0.0"}


# ── DEV ENTRYPOINT ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )

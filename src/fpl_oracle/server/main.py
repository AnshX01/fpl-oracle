"""
FastAPI Main Application Entry Point.
Runs FPL Oracle backend server and serves single-page dashboard at http://localhost:8000.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from fpl_oracle.server.routes.api import router as api_router
from fpl_oracle.server.jobs import start_scheduler, stop_scheduler
from fpl_oracle.server.safe_json import SafeJSONResponse, register_fastapi_safe_encoders
from fpl_oracle.config import BASE_DIR

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

# Register safe JSON encoders globally
register_fastapi_safe_encoders()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background scheduler
    start_scheduler()
    yield
    # Shutdown
    stop_scheduler()

app = FastAPI(
    title="FPL Oracle",
    description="Local ML-driven Fantasy Premier League Expert & Optimization Engine (Season 2026/27)",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount Static Files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>FPL Oracle Server Running</h1><p>Frontend file not found.</p>")

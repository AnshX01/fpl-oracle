"""
FastAPI Main Application Entry Point.
Runs FPL Oracle backend server and serves single-page dashboard at http://localhost:8000.
Features global structured error handling, rotating file logging, and graceful lifecycle shutdown.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.config import BASE_DIR
from fpl_oracle.server.jobs import start_scheduler, stop_scheduler
from fpl_oracle.server.routes.api import router as api_router
from fpl_oracle.server.safe_json import SafeJSONResponse, register_fastapi_safe_encoders
from fpl_oracle.utils.logging import setup_logging

# Setup structured rotating file logger
logger = setup_logging()

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

# Register safe JSON encoders globally
register_fastapi_safe_encoders()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background scheduler
    logger.info("Starting FPL Oracle application server and scheduler...")
    start_scheduler()
    yield
    # Shutdown
    logger.info("Shutting down FPL Oracle application server...")
    stop_scheduler()
    await fpl_client.aclose()


app = FastAPI(
    title="FPL Oracle",
    description="Local ML-driven Fantasy Premier League Expert & Optimization Engine (Season 2026/27)",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    lifespan=lifespan,
)


# Global Error Handlers (Never leak raw tracebacks)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "code": exc.status_code,
            "hint": "Check request parameters or verify resource identifiers.",
            "fallback_used": False,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()[:3]}")
    return SafeJSONResponse(
        status_code=422,
        content={
            "error": "Validation error in request parameters",
            "code": 422,
            "hint": [f"{e.get('loc')}: {e.get('msg')}" for e in exc.errors()[:3]],
            "fallback_used": False,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
    return SafeJSONResponse(
        status_code=500,
        content={
            "error": "Internal server processing error",
            "code": 500,
            "hint": "An internal error occurred. Safe fallback engaged.",
            "fallback_used": True,
        },
    )


# CORS: Restrict to local loopback origins for local security
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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


@app.get("/health")
async def health_redirect():
    from fpl_oracle.server.routes.api import get_health
    return await get_health()

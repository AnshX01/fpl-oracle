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

from fpl_oracle.utils.logging import setup_logging

# Setup structured rotating file logger first before importing domain modules (S1-1d)
logger = setup_logging()

from fpl_oracle.api.fpl_client import fpl_client  # noqa: E402
from fpl_oracle.api.read_context import read_context  # noqa: E402
from fpl_oracle.config import BASE_DIR  # noqa: E402
from fpl_oracle.server.jobs import start_scheduler, stop_scheduler  # noqa: E402
from fpl_oracle.server.routes.api import router as api_router  # noqa: E402
from fpl_oracle.server.safe_json import SafeJSONResponse, register_fastapi_safe_encoders  # noqa: E402

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

# Register safe JSON encoders globally
register_fastapi_safe_encoders()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background scheduler
    logger.info("Starting FPL Oracle application server and scheduler...")
    try:
        start_scheduler()
    except Exception as e:
        logger.error(f"Failed to start scheduler on startup: {e}", exc_info=True)

    yield

    # Shutdown: Stop background scheduler
    logger.info("Stopping FPL Oracle background scheduler...")
    try:
        stop_scheduler()
    except Exception as e:
        logger.error(f"Failed to stop scheduler on shutdown: {e}", exc_info=True)


app = FastAPI(
    title="FPL Oracle API & Dashboard",
    description="Unified API and local UI for FPL Oracle",
    version="1.0.0",
    default_response_class=SafeJSONResponse,
    lifespan=lifespan,
)

# CORS Middleware (Local dev & dashboard access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def upstream_snapshot_guard(request: Request, call_next):
    """Pin repeated upstream reads and reject an advice response changed during work."""
    context: dict = {}
    token = read_context.set(context)
    try:
        response = await call_next(request)
        changed = [
            key for key, (_, _, timestamp) in context.items() if fpl_client._cache_timestamps.get(key) != timestamp
        ]
        if (
            changed
            and request.url.path.startswith("/api/")
            and (request.method == "GET" or request.url.path in ("/api/transfers", "/api/optimize"))
        ):
            return SafeJSONResponse(
                status_code=409,
                content={
                    "status": "unavailable",
                    "is_stale": True,
                    "error": "Upstream data changed during analysis. Refresh the coherent snapshot.",
                },
            )
        response.headers["X-FPL-Revision"] = str(fpl_client.revision)
        return response
    finally:
        read_context.reset(token)


# Include API Router (already has /api prefix configured)
app.include_router(api_router)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return SafeJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "detail": exc.detail,
            "code": exc.status_code,
            "hint": "Check request parameters or verify resource identifiers.",
            "fallback_used": False,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return SafeJSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred. Please check application logs."},
    )


# Static File Serving & UI Single-Page Fallback
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>FPL Oracle Web UI not found</h1>", status_code=404)


@app.get("/{full_path:path}", response_class=FileResponse, include_in_schema=False)
async def serve_spa_or_static(full_path: str):
    # Exclude /api paths from SPA fallback
    if full_path.startswith("api/"):
        return SafeJSONResponse(status_code=404, content={"detail": "API endpoint not found"})

    candidate_file = WEB_DIR / full_path
    if candidate_file.is_file():
        return FileResponse(candidate_file)

    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>FPL Oracle Web UI not found</h1>", status_code=404)

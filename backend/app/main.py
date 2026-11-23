import sys
from pathlib import Path

# Ensure backend directory is in sys.path regardless of execution working directory or invocation style
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.images import router as images_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.system import router as system_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# API v1 route mounting
app.include_router(health_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")

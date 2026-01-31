"""
DataChat Analytics Platform - Main FastAPI Application
An ephemeral, session-based data analytics platform with natural language interface.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.api import upload, chat, analytics, export, session
from app.core.config import settings
from app.core.session_manager import session_manager

# Path to frontend static files (set in Docker production build)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 DataChat Analytics Platform Starting...")
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs(settings.EXPORT_DIR, exist_ok=True)
    
    yield
    
    # Shutdown - Clean up all sessions
    print("🧹 Cleaning up all sessions...")
    await session_manager.cleanup_all()
    print("👋 DataChat Analytics Platform Shutdown Complete")


app = FastAPI(
    title="DataChat Analytics Platform",
    description="Natural language data analytics with ephemeral sessions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration (allow all in production when served from same origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production if needed; same-origin when behind one host
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/upload", tags=["Upload & Ingestion"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat & NL Query"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(session.router, prefix="/api/session", tags=["Session"])


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions)
    }


# Serve frontend in production (when static build exists)
if os.path.isdir(STATIC_DIR):
    index_path = os.path.join(STATIC_DIR, "index.html")
    assets_dir = os.path.join(STATIC_DIR, "assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path)

    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve static files or index.html for SPA client-side routing."""
        if full_path.startswith("api/") or full_path == "api":
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(index_path)
else:
    @app.get("/")
    async def root():
        return {
            "message": "DataChat Analytics Platform",
            "version": "1.0.0",
            "docs": "/docs"
        }

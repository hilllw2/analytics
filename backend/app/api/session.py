"""
Session API Routes
Manage ephemeral sessions.
"""

from fastapi import APIRouter, HTTPException, Header, Body
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from app.core.session_manager import session_manager
from app.core.config import settings


router = APIRouter()


@router.post("/create")
async def create_session():
    """
    Create a new ephemeral session.
    """
    session = session_manager.create_session()
    
    return {
        "session_id": session.id,
        "created_at": session.created_at.isoformat(),
        "timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
        "message": "Session created. All data will be wiped when session ends or times out."
    }


@router.get("/info")
async def get_session_info(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Get information about a session.
    """
    info = session_manager.get_session_info(session_id)
    
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return info


class SetActiveDatasetRequest(BaseModel):
    dataset_name: str


@router.post("/set-active-dataset")
async def set_active_dataset(
    session_id: str = Header(..., alias="X-Session-ID"),
    body: SetActiveDatasetRequest = Body(...),
):
    """
    Set the active dataset (sheet/tab) for this session. Profile, chat, and charts use the active dataset.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if body.dataset_name not in session.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_name}' not found")
    session.active_dataset_name = body.dataset_name
    return {
        "success": True,
        "active_dataset_name": body.dataset_name,
    }


@router.post("/touch")
async def touch_session(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Update session activity to prevent timeout.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session.touch()
    
    return {
        "success": True,
        "last_activity": session.last_activity.isoformat(),
        "expires_in_minutes": settings.SESSION_TIMEOUT_MINUTES
    }


@router.delete("/clear")
async def clear_session(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Clear all data in a session but keep the session active.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session.clear()
    
    return {
        "success": True,
        "message": "Session data cleared. Session remains active."
    }


@router.delete("/end")
async def end_session(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    End a session and delete all data.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_manager.delete_session(session_id)
    
    return {
        "success": True,
        "message": "Session ended. All data has been permanently deleted."
    }


@router.get("/datasets")
async def list_datasets(
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    List all datasets in a session.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "datasets": [
            {
                "name": name,
                "filename": ds.original_filename,
                "row_count": ds.row_count,
                "column_count": ds.column_count,
                "upload_time": ds.upload_time.isoformat(),
                "is_active": name == session.active_dataset_name
            }
            for name, ds in session.datasets.items()
        ],
        "active_dataset": session.active_dataset_name
    }


@router.post("/datasets/{dataset_name}/activate")
async def activate_dataset(
    dataset_name: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Set a dataset as the active dataset.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if dataset_name not in session.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    session.active_dataset_name = dataset_name
    
    return {
        "success": True,
        "active_dataset": dataset_name
    }


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(
    dataset_name: str,
    session_id: str = Header(..., alias="X-Session-ID")
):
    """
    Delete a dataset from the session.
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if dataset_name not in session.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    del session.datasets[dataset_name]
    
    # Clear active dataset if it was the deleted one
    if session.active_dataset_name == dataset_name:
        session.active_dataset_name = next(iter(session.datasets.keys()), None)
    
    return {
        "success": True,
        "message": f"Dataset '{dataset_name}' deleted",
        "active_dataset": session.active_dataset_name
    }


@router.get("/status")
async def get_status():
    """
    Get overall status (admin endpoint).
    """
    return {
        "active_sessions": len(session_manager.sessions),
        "max_sessions": settings.MAX_SESSIONS,
        "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB
    }

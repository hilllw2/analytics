"""
Ephemeral Session Manager
Manages in-memory sessions with automatic cleanup.
All data is wiped when session ends.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from uuid import uuid4
import pandas as pd
from dataclasses import dataclass, field

from app.core.config import settings


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    original_filename: str
    df: pd.DataFrame
    inferred_types: Dict[str, str]
    date_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    upload_time: datetime
    row_count: int
    column_count: int
    is_sampled: bool = False
    full_row_count: Optional[int] = None


@dataclass 
class ChartInfo:
    """Information about a generated chart."""
    id: str
    chart_type: str
    title: str
    created_at: datetime
    plotly_json: Dict[str, Any]
    underlying_data: Optional[pd.DataFrame] = None


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PinnedDefinition:
    """User-defined metric or formula."""
    name: str
    formula: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """Ephemeral session containing all user data and state."""
    id: str
    created_at: datetime
    last_activity: datetime
    
    # Data
    datasets: Dict[str, DatasetInfo] = field(default_factory=dict)
    active_dataset_name: Optional[str] = None
    
    # Conversation
    messages: List[ConversationMessage] = field(default_factory=list)
    
    # Charts and results
    charts: Dict[str, ChartInfo] = field(default_factory=dict)
    cached_results: Dict[str, Any] = field(default_factory=dict)
    
    # User definitions
    pinned_definitions: Dict[str, PinnedDefinition] = field(default_factory=dict)
    
    # Current context (filters, groupings applied)
    current_filters: List[Dict[str, Any]] = field(default_factory=list)
    current_transformations: List[str] = field(default_factory=list)
    
    @property
    def active_dataset(self) -> Optional[DatasetInfo]:
        """Get the currently active dataset."""
        if self.active_dataset_name and self.active_dataset_name in self.datasets:
            return self.datasets[self.active_dataset_name]
        return None
    
    @property
    def active_df(self) -> Optional[pd.DataFrame]:
        """Get the currently active DataFrame."""
        dataset = self.active_dataset
        return dataset.df if dataset else None
    
    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
        return datetime.now() - self.last_activity > timeout
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        ))
        self.touch()
    
    def get_conversation_context(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get recent conversation for LLM context."""
        recent = self.messages[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def clear(self):
        """Clear all session data."""
        self.datasets.clear()
        self.active_dataset_name = None
        self.messages.clear()
        self.charts.clear()
        self.cached_results.clear()
        self.pinned_definitions.clear()
        self.current_filters.clear()
        self.current_transformations.clear()


class SessionManager:
    """
    Manages ephemeral sessions.
    No persistence - everything is in-memory and wiped on session end.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def create_session(self) -> Session:
        """Create a new ephemeral session."""
        session_id = str(uuid4())
        now = datetime.now()
        
        session = Session(
            id=session_id,
            created_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        
        # Start cleanup task if not running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, updating activity timestamp."""
        session = self.sessions.get(session_id)
        if session:
            if session.is_expired():
                self.delete_session(session_id)
                return None
            session.touch()
        return session
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.clear()  # Explicit cleanup
            del self.sessions[session_id]
    
    async def cleanup_expired(self):
        """Clean up all expired sessions."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            self.delete_session(sid)
        
        if expired:
            print(f"🧹 Cleaned up {len(expired)} expired sessions")
    
    async def cleanup_all(self):
        """Clean up all sessions (for shutdown)."""
        for session in list(self.sessions.values()):
            session.clear()
        self.sessions.clear()
    
    async def _periodic_cleanup(self):
        """Background task to clean up expired sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self.cleanup_expired()
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info without sensitive data."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "datasets": list(session.datasets.keys()),
            "active_dataset": session.active_dataset_name,
            "message_count": len(session.messages),
            "chart_count": len(session.charts),
            "expires_in_minutes": max(0, settings.SESSION_TIMEOUT_MINUTES - 
                (datetime.now() - session.last_activity).seconds // 60)
        }


# Global session manager instance
session_manager = SessionManager()

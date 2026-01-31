"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    GEMINI_API_KEY: str = ""
    
    # Session Settings (inactivity timeout)
    SESSION_TIMEOUT_MINUTES: int = 180  # 3 hours
    MAX_SESSIONS: int = 100
    
    # File Settings
    MAX_UPLOAD_SIZE_MB: int = 500
    PREVIEW_ROWS: int = 500
    MAX_ROWS_IN_MEMORY: int = 1_000_000
    SAMPLE_SIZE_FOR_LARGE_FILES: int = 10_000
    
    # Directories
    TEMP_DIR: str = "/tmp/datachat"
    EXPORT_DIR: str = "/tmp/datachat/exports"
    
    # LLM Settings
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_MAX_TOKENS: int = 8192
    LLM_TEMPERATURE: float = 0.3
    
    # Analytics Settings
    OUTLIER_IQR_MULTIPLIER: float = 1.5
    OUTLIER_ZSCORE_THRESHOLD: float = 3.0
    CORRELATION_THRESHOLD: float = 0.5
    HIGH_CARDINALITY_THRESHOLD: int = 100
    NEAR_CONSTANT_THRESHOLD: float = 0.95
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

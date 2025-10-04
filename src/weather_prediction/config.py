"""Configuration management for the Weather Prediction System."""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    APP_NAME: str = "Weather Prediction API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Model Settings
    MODEL_DIR: str = "model"
    MODEL_NAME: str = "weather_predictor"
    
    # Data Settings
    DATA_DIR: str = "data"
    TRAIN_DATA_FILE: str = "weatherAUS.csv"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Monitoring
    ENABLE_METRICS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


settings = get_settings()

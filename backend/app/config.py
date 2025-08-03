"""Configuration management for Healthcare AI Predictor."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Healthcare AI Predictor"
    debug: bool = False
    
    # Data paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"
    
    # Model settings
    model_name: str = "healthcare_los_predictor"
    model_version: str = "v1.0"
    
    # API settings
    api_v1_prefix: str = "/api"
    
    class Config:
        env_file = ".env"


settings = Settings()
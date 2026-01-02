"""Configuration management for the agent system."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-storm-8b")
    MODEL_BASE_URL: str = os.getenv("OPENAI_HOST", "http://localhost:11434/v1")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_TITLE: str = "Supervisor Agent System"
    API_VERSION: str = "1.0.0"
    
    # Agent Configuration
    AVAILABLE_AGENTS: List[str] = ["weather_agent"]
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        assert cls.MODEL_BASE_URL, "OPENAI_HOST must be set"
        assert cls.MODEL_NAME, "MODEL_NAME must be set"

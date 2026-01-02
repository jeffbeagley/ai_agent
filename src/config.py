import os

class Config:
    # Model configuration
    MODEL_NAME = "llama-3.1-storm-8b"
    MODEL_BASE_URL = os.getenv("OPENAI_HOST", "http://localhost:11434/v1")
    MODEL_TEMPERATURE = 0.7
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Agent configuration
    AVAILABLE_AGENTS = ["weather_agent"]
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_host: str = "http://ollama:11434"
    model_name: str = "qwen2.5-coder:3b"
    max_tokens: int = 1024
    rate_limit: str = "20/minute"
    log_level: str = "info"
    app_name: str = "Cloud Code Review API"
    version: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()

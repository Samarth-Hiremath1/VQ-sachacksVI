from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/coaching_platform"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # API
    api_v1_prefix: str = "/api/v1"
    
    # File Upload
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    
    class Config:
        env_file = ".env"


settings = Settings()
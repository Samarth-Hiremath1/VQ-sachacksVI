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
    max_video_duration: int = 3600  # 1 hour in seconds
    allowed_video_formats: list[str] = ["mp4", "avi", "mov", "webm"]
    allowed_audio_formats: list[str] = ["mp3", "wav", "m4a", "aac"]
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
    
    # MinIO S3 Configuration
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "access_key"
    s3_secret_key: str = "secret_key"
    s3_bucket_name: str = "coaching-platform"
    s3_secure: bool = False  # Use HTTP for local development
    s3_region: str = "us-east-1"
    
    # File lifecycle management
    temp_file_cleanup_hours: int = 24
    failed_upload_cleanup_hours: int = 6
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()
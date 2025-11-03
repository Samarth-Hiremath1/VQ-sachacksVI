from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import mimetypes


class RecordingUploadRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Title of the recording")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the recording")


class RecordingUploadResponse(BaseModel):
    recording_id: UUID
    upload_url: str
    max_file_size: int
    allowed_formats: list[str]
    chunk_size: int


class FileValidationResult(BaseModel):
    is_valid: bool
    file_type: str
    file_size: int
    duration_seconds: Optional[int] = None
    format: Optional[str] = None
    errors: list[str] = []


class RecordingMetadata(BaseModel):
    filename: str
    content_type: str
    file_size: int
    duration_seconds: Optional[int] = None
    format: Optional[str] = None
    resolution: Optional[str] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if not v or '/' not in v:
            raise ValueError('Invalid content type')
        return v


class RecordingResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str]
    video_s3_key: Optional[str]
    audio_s3_key: Optional[str]
    duration_seconds: Optional[int]
    file_size_bytes: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RecordingListResponse(BaseModel):
    recordings: list[RecordingResponse]
    total: int
    page: int
    per_page: int


class UploadProgressResponse(BaseModel):
    recording_id: UUID
    status: str
    progress_percentage: float
    bytes_uploaded: int
    total_bytes: int
    estimated_time_remaining: Optional[int] = None  # seconds


class ChunkedUploadInitRequest(BaseModel):
    filename: str
    file_size: int
    content_type: str
    chunk_count: int
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v <= 0:
            raise ValueError('File size must be positive')
        return v
    
    @validator('chunk_count')
    def validate_chunk_count(cls, v):
        if v <= 0:
            raise ValueError('Chunk count must be positive')
        return v


class ChunkedUploadInitResponse(BaseModel):
    upload_id: str
    recording_id: UUID
    chunk_urls: list[str]
    expires_at: datetime


class ChunkUploadResponse(BaseModel):
    chunk_number: int
    etag: str
    status: str


class CompleteUploadRequest(BaseModel):
    upload_id: str
    chunks: list[ChunkUploadResponse]


class FileCleanupStats(BaseModel):
    temp_files_cleaned: int
    failed_uploads_cleaned: int
    storage_freed_bytes: int
    cleanup_timestamp: datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from uuid import UUID
import logging

from ...core.database import get_db
from ...core.dependencies import get_current_user
from ...core.minio_client import get_minio_client, MinIOClient
from ...models.user import User
from ...schemas.recording import (
    RecordingUploadRequest,
    RecordingUploadResponse,
    RecordingResponse,
    RecordingListResponse,
    UploadProgressResponse,
    FileCleanupStats
)
from ...services.file_service import FileStorageService
from ...services.recording_service import RecordingService
from ...core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=RecordingResponse)
async def upload_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Upload a recording file (video or audio)"""
    
    # Validate file size
    if hasattr(file, 'size') and file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error_code": "FILE_TOO_LARGE",
                "message": f"File size {file.size} exceeds maximum allowed size {settings.max_file_size}",
                "max_size": settings.max_file_size
            }
        )
    
    # Create upload request
    upload_request = RecordingUploadRequest(title=title, description=description)
    
    # Initialize services
    file_service = FileStorageService(minio_client)
    recording_service = RecordingService(db, minio_client)
    
    try:
        # Upload file and get metadata
        upload_result = file_service.upload_file(file, str(current_user.id))
        
        # Create recording record
        recording = recording_service.create_recording(
            user_id=current_user.id,
            upload_request=upload_request,
            s3_key=upload_result["s3_key"],
            metadata=upload_result["metadata"]
        )
        
        # Schedule background cleanup task
        background_tasks.add_task(
            cleanup_old_files,
            recording_service
        )
        
        logger.info(f"Successfully uploaded recording {recording.id} for user {current_user.id}")
        
        return RecordingResponse.from_orm(recording)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading recording: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "UPLOAD_ERROR",
                "message": "An error occurred during file upload"
            }
        )


@router.get("/", response_model=RecordingListResponse)
async def get_recordings(
    page: int = 1,
    per_page: int = 20,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Get user's recordings with pagination"""
    
    if per_page > 100:
        per_page = 100
    
    recording_service = RecordingService(db, minio_client)
    
    try:
        return recording_service.get_user_recordings(
            user_id=current_user.id,
            page=page,
            per_page=per_page,
            status_filter=status_filter
        )
    except Exception as e:
        logger.error(f"Error fetching recordings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "FETCH_ERROR",
                "message": "An error occurred while fetching recordings"
            }
        )


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Get a specific recording by ID"""
    
    recording_service = RecordingService(db, minio_client)
    
    recording = recording_service.get_recording_by_id(recording_id, current_user.id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "RECORDING_NOT_FOUND",
                "message": "Recording not found"
            }
        )
    
    return RecordingResponse.from_orm(recording)


@router.get("/{recording_id}/download")
async def get_recording_download_url(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Get a presigned URL for downloading the recording"""
    
    recording_service = RecordingService(db, minio_client)
    
    download_url = recording_service.get_recording_download_url(recording_id, current_user.id)
    if not download_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "RECORDING_NOT_FOUND",
                "message": "Recording not found or no file available"
            }
        )
    
    return {
        "download_url": download_url,
        "expires_in_seconds": 3600
    }


@router.delete("/{recording_id}")
async def delete_recording(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Delete a recording and its associated files"""
    
    recording_service = RecordingService(db, minio_client)
    
    success = recording_service.delete_recording(recording_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "RECORDING_NOT_FOUND",
                "message": "Recording not found"
            }
        )
    
    return {"message": "Recording deleted successfully"}


@router.get("/stats/summary")
async def get_recording_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Get user's recording statistics"""
    
    recording_service = RecordingService(db, minio_client)
    
    try:
        stats = recording_service.get_recording_stats(current_user.id)
        return stats
    except Exception as e:
        logger.error(f"Error fetching recording stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "STATS_ERROR",
                "message": "An error occurred while fetching statistics"
            }
        )


@router.post("/cleanup", response_model=FileCleanupStats)
async def cleanup_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Manually trigger file cleanup (admin only for now)"""
    
    # In a production system, this would be restricted to admin users
    from ...services.lifecycle_service import LifecycleManagementService
    
    lifecycle_service = LifecycleManagementService(db, minio_client)
    
    try:
        results = lifecycle_service.apply_storage_policies()
        
        return FileCleanupStats(
            temp_files_cleaned=results["temp_files"]["cleaned_count"],
            failed_uploads_cleaned=results["failed_uploads"]["cleaned_count"],
            storage_freed_bytes=results["total_storage_freed"],
            cleanup_timestamp=results["failed_uploads"]["cleanup_timestamp"]
        )
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "CLEANUP_ERROR",
                "message": "An error occurred during cleanup"
            }
        )


@router.get("/lifecycle/status")
async def get_lifecycle_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Get lifecycle management and background task status"""
    
    from ...services.lifecycle_service import LifecycleManagementService
    from ...services.background_tasks import get_background_service
    
    try:
        lifecycle_service = LifecycleManagementService(db, minio_client)
        background_service = get_background_service()
        
        # Get storage usage stats
        usage_stats = lifecycle_service.get_storage_usage_stats()
        
        # Get background task status
        task_status = background_service.get_task_status()
        
        return {
            "storage_usage": usage_stats,
            "background_tasks": task_status,
            "lifecycle_policies": {
                "failed_upload_cleanup_hours": settings.failed_upload_cleanup_hours,
                "temp_file_cleanup_hours": settings.temp_file_cleanup_hours,
                "max_file_size": settings.max_file_size
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting lifecycle status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "STATUS_ERROR",
                "message": "An error occurred while getting lifecycle status"
            }
        )


@router.post("/lifecycle/integrity-check")
async def run_integrity_check(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Run storage integrity validation"""
    
    from ...services.lifecycle_service import LifecycleManagementService
    
    try:
        lifecycle_service = LifecycleManagementService(db, minio_client)
        results = lifecycle_service.validate_storage_integrity()
        
        return results
        
    except Exception as e:
        logger.error(f"Error running integrity check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTEGRITY_CHECK_ERROR",
                "message": "An error occurred during integrity check"
            }
        )


@router.get("/health/storage")
async def check_storage_health(
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """Check MinIO storage health"""
    
    try:
        is_healthy = minio_client.health_check()
        
        if is_healthy:
            return {
                "status": "healthy",
                "storage": "accessible",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "storage": "inaccessible",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "storage": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# Background task functions
async def cleanup_old_files(recording_service: RecordingService):
    """Background task to clean up old files"""
    try:
        cleaned_count = recording_service.cleanup_failed_uploads(
            hours_old=settings.failed_upload_cleanup_hours
        )
        if cleaned_count > 0:
            logger.info(f"Background cleanup removed {cleaned_count} failed uploads")
    except Exception as e:
        logger.error(f"Background cleanup failed: {e}")


# Import datetime for the cleanup endpoint
from datetime import datetime
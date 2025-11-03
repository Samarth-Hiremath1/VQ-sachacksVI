from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
import logging

from ..models.recording import Recording
from ..models.user import User
from ..schemas.recording import (
    RecordingResponse, 
    RecordingListResponse, 
    RecordingUploadRequest,
    RecordingMetadata
)
from ..core.minio_client import MinIOClient

logger = logging.getLogger(__name__)


class RecordingService:
    def __init__(self, db: Session, minio_client: MinIOClient):
        self.db = db
        self.minio_client = minio_client
    
    def create_recording(
        self, 
        user_id: UUID, 
        upload_request: RecordingUploadRequest,
        s3_key: str,
        metadata: RecordingMetadata
    ) -> Recording:
        """Create a new recording record in the database"""
        try:
            recording = Recording(
                user_id=user_id,
                title=upload_request.title or metadata.filename,
                video_s3_key=s3_key if metadata.content_type.startswith('video/') else None,
                audio_s3_key=s3_key if metadata.content_type.startswith('audio/') else None,
                duration_seconds=metadata.duration_seconds,
                file_size_bytes=metadata.file_size,
                status="uploaded"
            )
            
            self.db.add(recording)
            self.db.commit()
            self.db.refresh(recording)
            
            logger.info(f"Created recording {recording.id} for user {user_id}")
            return recording
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating recording: {e}")
            raise
    
    def get_recording_by_id(self, recording_id: UUID, user_id: Optional[UUID] = None) -> Optional[Recording]:
        """Get a recording by ID, optionally filtered by user"""
        query = self.db.query(Recording).filter(Recording.id == recording_id)
        
        if user_id:
            query = query.filter(Recording.user_id == user_id)
        
        return query.first()
    
    def get_user_recordings(
        self, 
        user_id: UUID, 
        page: int = 1, 
        per_page: int = 20,
        status_filter: Optional[str] = None
    ) -> RecordingListResponse:
        """Get paginated list of user's recordings"""
        query = self.db.query(Recording).filter(Recording.user_id == user_id)
        
        if status_filter:
            query = query.filter(Recording.status == status_filter)
        
        # Get total count
        total = query.count()
        
        # Apply pagination and ordering
        recordings = (
            query
            .order_by(desc(Recording.created_at))
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        
        return RecordingListResponse(
            recordings=[RecordingResponse.from_orm(r) for r in recordings],
            total=total,
            page=page,
            per_page=per_page
        )
    
    def update_recording_status(self, recording_id: UUID, status: str) -> Optional[Recording]:
        """Update recording status"""
        try:
            recording = self.db.query(Recording).filter(Recording.id == recording_id).first()
            if recording:
                recording.status = status
                recording.updated_at = datetime.utcnow()
                self.db.commit()
                self.db.refresh(recording)
                logger.info(f"Updated recording {recording_id} status to {status}")
            return recording
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating recording status: {e}")
            raise
    
    def delete_recording(self, recording_id: UUID, user_id: UUID) -> bool:
        """Delete a recording and its associated files"""
        try:
            recording = self.get_recording_by_id(recording_id, user_id)
            if not recording:
                return False
            
            # Delete files from MinIO
            files_deleted = True
            if recording.video_s3_key:
                files_deleted &= self.minio_client.delete_file(recording.video_s3_key)
            if recording.audio_s3_key:
                files_deleted &= self.minio_client.delete_file(recording.audio_s3_key)
            
            # Delete database record
            self.db.delete(recording)
            self.db.commit()
            
            logger.info(f"Deleted recording {recording_id}")
            return files_deleted
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting recording: {e}")
            raise
    
    def get_recording_download_url(self, recording_id: UUID, user_id: UUID) -> Optional[str]:
        """Get a presigned URL for downloading the recording"""
        recording = self.get_recording_by_id(recording_id, user_id)
        if not recording:
            return None
        
        # Prefer video over audio
        s3_key = recording.video_s3_key or recording.audio_s3_key
        if not s3_key:
            return None
        
        return self.minio_client.get_file_url(s3_key, expires_in_seconds=3600)
    
    def get_recording_stats(self, user_id: UUID) -> Dict[str, Any]:
        """Get user's recording statistics"""
        try:
            stats = (
                self.db.query(
                    func.count(Recording.id).label('total_recordings'),
                    func.sum(Recording.file_size_bytes).label('total_size_bytes'),
                    func.sum(Recording.duration_seconds).label('total_duration_seconds'),
                    func.avg(Recording.duration_seconds).label('avg_duration_seconds')
                )
                .filter(Recording.user_id == user_id)
                .first()
            )
            
            # Get status breakdown
            status_stats = (
                self.db.query(
                    Recording.status,
                    func.count(Recording.id).label('count')
                )
                .filter(Recording.user_id == user_id)
                .group_by(Recording.status)
                .all()
            )
            
            return {
                'total_recordings': stats.total_recordings or 0,
                'total_size_bytes': stats.total_size_bytes or 0,
                'total_duration_seconds': stats.total_duration_seconds or 0,
                'avg_duration_seconds': float(stats.avg_duration_seconds or 0),
                'status_breakdown': {stat.status: stat.count for stat in status_stats}
            }
            
        except Exception as e:
            logger.error(f"Error getting recording stats: {e}")
            return {
                'total_recordings': 0,
                'total_size_bytes': 0,
                'total_duration_seconds': 0,
                'avg_duration_seconds': 0.0,
                'status_breakdown': {}
            }
    
    def cleanup_failed_uploads(self, hours_old: int = 6) -> int:
        """Clean up recordings that failed to upload completely"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
            
            failed_recordings = (
                self.db.query(Recording)
                .filter(
                    Recording.status.in_(['uploading', 'failed']),
                    Recording.created_at < cutoff_time
                )
                .all()
            )
            
            cleaned_count = 0
            for recording in failed_recordings:
                try:
                    # Delete files from MinIO
                    if recording.video_s3_key:
                        self.minio_client.delete_file(recording.video_s3_key)
                    if recording.audio_s3_key:
                        self.minio_client.delete_file(recording.audio_s3_key)
                    
                    # Delete database record
                    self.db.delete(recording)
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning up recording {recording.id}: {e}")
            
            if cleaned_count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {cleaned_count} failed uploads")
            
            return cleaned_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during cleanup: {e}")
            return 0
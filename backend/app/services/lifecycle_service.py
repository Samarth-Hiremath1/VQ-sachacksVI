"""
File lifecycle management service for automated cleanup and policies
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.recording import Recording
from ..core.minio_client import MinIOClient
from ..core.config import settings

logger = logging.getLogger(__name__)


class LifecycleManagementService:
    """Service for managing file lifecycle policies and automated cleanup"""
    
    def __init__(self, db: Session, minio_client: MinIOClient):
        self.db = db
        self.minio_client = minio_client
    
    def cleanup_failed_uploads(self, hours_old: int = None) -> Dict[str, Any]:
        """Clean up recordings that failed to upload completely"""
        if hours_old is None:
            hours_old = settings.failed_upload_cleanup_hours
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
        
        try:
            # Find failed uploads
            failed_recordings = (
                self.db.query(Recording)
                .filter(
                    and_(
                        Recording.status.in_(['uploading', 'failed', 'processing_failed']),
                        Recording.created_at < cutoff_time
                    )
                )
                .all()
            )
            
            cleaned_count = 0
            storage_freed = 0
            errors = []
            
            for recording in failed_recordings:
                try:
                    # Track storage freed
                    if recording.file_size_bytes:
                        storage_freed += recording.file_size_bytes
                    
                    # Delete files from MinIO
                    if recording.video_s3_key:
                        if not self.minio_client.delete_file(recording.video_s3_key):
                            errors.append(f"Failed to delete video file for recording {recording.id}")
                    
                    if recording.audio_s3_key:
                        if not self.minio_client.delete_file(recording.audio_s3_key):
                            errors.append(f"Failed to delete audio file for recording {recording.id}")
                    
                    # Delete database record
                    self.db.delete(recording)
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning up recording {recording.id}: {e}")
                    errors.append(f"Recording {recording.id}: {str(e)}")
            
            if cleaned_count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {cleaned_count} failed uploads, freed {storage_freed} bytes")
            
            return {
                "cleaned_count": cleaned_count,
                "storage_freed_bytes": storage_freed,
                "errors": errors,
                "cleanup_timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during failed upload cleanup: {e}")
            return {
                "cleaned_count": 0,
                "storage_freed_bytes": 0,
                "errors": [str(e)],
                "cleanup_timestamp": datetime.utcnow()
            }
    
    def cleanup_old_temp_files(self, hours_old: int = None) -> Dict[str, Any]:
        """Clean up temporary files older than specified hours"""
        if hours_old is None:
            hours_old = settings.temp_file_cleanup_hours
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
        
        try:
            # Find recordings marked as temporary or in processing state for too long
            temp_recordings = (
                self.db.query(Recording)
                .filter(
                    and_(
                        Recording.status.in_(['temp', 'processing']),
                        Recording.created_at < cutoff_time
                    )
                )
                .all()
            )
            
            cleaned_count = 0
            storage_freed = 0
            errors = []
            
            for recording in temp_recordings:
                try:
                    # Track storage freed
                    if recording.file_size_bytes:
                        storage_freed += recording.file_size_bytes
                    
                    # Delete files from MinIO
                    if recording.video_s3_key:
                        if not self.minio_client.delete_file(recording.video_s3_key):
                            errors.append(f"Failed to delete temp video file for recording {recording.id}")
                    
                    if recording.audio_s3_key:
                        if not self.minio_client.delete_file(recording.audio_s3_key):
                            errors.append(f"Failed to delete temp audio file for recording {recording.id}")
                    
                    # Delete database record
                    self.db.delete(recording)
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning up temp recording {recording.id}: {e}")
                    errors.append(f"Recording {recording.id}: {str(e)}")
            
            if cleaned_count > 0:
                self.db.commit()
                logger.info(f"Cleaned up {cleaned_count} temp files, freed {storage_freed} bytes")
            
            return {
                "cleaned_count": cleaned_count,
                "storage_freed_bytes": storage_freed,
                "errors": errors,
                "cleanup_timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during temp file cleanup: {e}")
            return {
                "cleaned_count": 0,
                "storage_freed_bytes": 0,
                "errors": [str(e)],
                "cleanup_timestamp": datetime.utcnow()
            }
    
    def apply_storage_policies(self) -> Dict[str, Any]:
        """Apply storage lifecycle policies"""
        results = {
            "failed_uploads": self.cleanup_failed_uploads(),
            "temp_files": self.cleanup_old_temp_files(),
            "total_storage_freed": 0,
            "total_files_cleaned": 0
        }
        
        # Calculate totals
        results["total_storage_freed"] = (
            results["failed_uploads"]["storage_freed_bytes"] +
            results["temp_files"]["storage_freed_bytes"]
        )
        results["total_files_cleaned"] = (
            results["failed_uploads"]["cleaned_count"] +
            results["temp_files"]["cleaned_count"]
        )
        
        return results
    
    def get_storage_usage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            from sqlalchemy import func
            
            # Get overall stats
            stats = (
                self.db.query(
                    func.count(Recording.id).label('total_recordings'),
                    func.sum(Recording.file_size_bytes).label('total_size_bytes'),
                    func.avg(Recording.file_size_bytes).label('avg_size_bytes')
                )
                .first()
            )
            
            # Get stats by status
            status_stats = (
                self.db.query(
                    Recording.status,
                    func.count(Recording.id).label('count'),
                    func.sum(Recording.file_size_bytes).label('size_bytes')
                )
                .group_by(Recording.status)
                .all()
            )
            
            # Get stats by age
            now = datetime.utcnow()
            age_ranges = [
                ("last_24h", now - timedelta(hours=24)),
                ("last_week", now - timedelta(days=7)),
                ("last_month", now - timedelta(days=30)),
                ("older", datetime.min)
            ]
            
            age_stats = {}
            for label, cutoff in age_ranges:
                if label == "older":
                    # Older than 30 days
                    count_query = self.db.query(func.count(Recording.id)).filter(
                        Recording.created_at < (now - timedelta(days=30))
                    )
                    size_query = self.db.query(func.sum(Recording.file_size_bytes)).filter(
                        Recording.created_at < (now - timedelta(days=30))
                    )
                else:
                    count_query = self.db.query(func.count(Recording.id)).filter(
                        Recording.created_at >= cutoff
                    )
                    size_query = self.db.query(func.sum(Recording.file_size_bytes)).filter(
                        Recording.created_at >= cutoff
                    )
                
                age_stats[label] = {
                    "count": count_query.scalar() or 0,
                    "size_bytes": size_query.scalar() or 0
                }
            
            return {
                "total_recordings": stats.total_recordings or 0,
                "total_size_bytes": stats.total_size_bytes or 0,
                "avg_size_bytes": float(stats.avg_size_bytes or 0),
                "status_breakdown": {
                    stat.status: {
                        "count": stat.count,
                        "size_bytes": stat.size_bytes or 0
                    }
                    for stat in status_stats
                },
                "age_breakdown": age_stats,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting storage usage stats: {e}")
            return {
                "total_recordings": 0,
                "total_size_bytes": 0,
                "avg_size_bytes": 0.0,
                "status_breakdown": {},
                "age_breakdown": {},
                "error": str(e),
                "generated_at": datetime.utcnow()
            }
    
    def validate_storage_integrity(self) -> Dict[str, Any]:
        """Validate that database records match actual files in storage"""
        try:
            recordings = self.db.query(Recording).filter(
                or_(
                    Recording.video_s3_key.isnot(None),
                    Recording.audio_s3_key.isnot(None)
                )
            ).all()
            
            missing_files = []
            orphaned_records = []
            size_mismatches = []
            
            for recording in recordings:
                # Check video file
                if recording.video_s3_key:
                    if not self.minio_client.file_exists(recording.video_s3_key):
                        missing_files.append({
                            "recording_id": str(recording.id),
                            "file_type": "video",
                            "s3_key": recording.video_s3_key
                        })
                    else:
                        # Check file size
                        file_info = self.minio_client.get_file_info(recording.video_s3_key)
                        if file_info and recording.file_size_bytes:
                            if abs(file_info["size"] - recording.file_size_bytes) > 1024:  # Allow 1KB difference
                                size_mismatches.append({
                                    "recording_id": str(recording.id),
                                    "file_type": "video",
                                    "db_size": recording.file_size_bytes,
                                    "actual_size": file_info["size"]
                                })
                
                # Check audio file
                if recording.audio_s3_key:
                    if not self.minio_client.file_exists(recording.audio_s3_key):
                        missing_files.append({
                            "recording_id": str(recording.id),
                            "file_type": "audio",
                            "s3_key": recording.audio_s3_key
                        })
            
            return {
                "total_checked": len(recordings),
                "missing_files": missing_files,
                "orphaned_records": orphaned_records,
                "size_mismatches": size_mismatches,
                "integrity_score": 1.0 - (len(missing_files) + len(size_mismatches)) / max(len(recordings), 1),
                "checked_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error validating storage integrity: {e}")
            return {
                "total_checked": 0,
                "missing_files": [],
                "orphaned_records": [],
                "size_mismatches": [],
                "integrity_score": 0.0,
                "error": str(e),
                "checked_at": datetime.utcnow()
            }
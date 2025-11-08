"""
Storage utilities for S3/MinIO operations

Handles file uploads, downloads, and management in S3-compatible storage.
Requirements: 3.4, 6.2, 6.6
"""

import os
import logging
from typing import Dict, Any, Optional
from minio import Minio
from minio.error import S3Error
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class S3StorageHandler:
    """Handler for S3-compatible storage operations using MinIO"""
    
    def __init__(self):
        self.endpoint = os.getenv('S3_ENDPOINT', 'minio:9000')
        self.access_key = os.getenv('S3_ACCESS_KEY', 'access_key')
        self.secret_key = os.getenv('S3_SECRET_KEY', 'secret_key')
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'coaching-platform')
        self.secure = os.getenv('S3_SECURE', 'false').lower() == 'true'
        
        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        logger.info(f"Initialized S3StorageHandler with endpoint: {self.endpoint}")
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def upload_recording(self, recording_id: str, file_path: str) -> Dict[str, Any]:
        """
        Upload recording file to S3
        
        Args:
            recording_id: Unique recording identifier
            file_path: Local path to file
            
        Returns:
            Dict with upload details
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Generate S3 key
            file_extension = Path(file_path).suffix
            s3_key = f"recordings/{recording_id}/video{file_extension}"
            
            # Get file metadata
            file_size = os.path.getsize(file_path)
            
            logger.info(f"Uploading {file_path} to s3://{self.bucket_name}/{s3_key}")
            
            # Upload file
            self.client.fput_object(
                self.bucket_name,
                s3_key,
                file_path,
                content_type=self._get_content_type(file_extension)
            )
            
            logger.info(f"Successfully uploaded {s3_key}")
            
            return {
                's3_key': s3_key,
                'video_path': file_path,
                'file_size': file_size,
                'duration': self._get_video_duration(file_path),
                'bucket': self.bucket_name
            }
            
        except S3Error as e:
            logger.error(f"S3 error uploading file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error uploading recording: {e}")
            raise
    
    def upload_audio(self, recording_id: str, audio_path: str) -> Dict[str, Any]:
        """
        Upload audio file to S3
        
        Args:
            recording_id: Unique recording identifier
            audio_path: Local path to audio file
            
        Returns:
            Dict with upload details
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Generate S3 key
            file_extension = Path(audio_path).suffix
            s3_key = f"recordings/{recording_id}/audio{file_extension}"
            
            logger.info(f"Uploading {audio_path} to s3://{self.bucket_name}/{s3_key}")
            
            # Upload file
            self.client.fput_object(
                self.bucket_name,
                s3_key,
                audio_path,
                content_type=self._get_content_type(file_extension)
            )
            
            logger.info(f"Successfully uploaded {s3_key}")
            
            return {
                's3_key': s3_key,
                'audio_path': audio_path,
                'bucket': self.bucket_name
            }
            
        except S3Error as e:
            logger.error(f"S3 error uploading audio: {e}")
            raise
        except Exception as e:
            logger.error(f"Error uploading audio: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str) -> str:
        """
        Download file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.client.fget_object(
                self.bucket_name,
                s3_key,
                local_path
            )
            
            logger.info(f"Successfully downloaded to {local_path}")
            return local_path
            
        except S3Error as e:
            logger.error(f"S3 error downloading file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Deleting s3://{self.bucket_name}/{s3_key}")
            
            self.client.remove_object(self.bucket_name, s3_key)
            
            logger.info(f"Successfully deleted {s3_key}")
            return True
            
        except S3Error as e:
            logger.error(f"S3 error deleting file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def get_file_url(self, s3_key: str, expires_in_seconds: int = 3600) -> str:
        """
        Get presigned URL for file access
        
        Args:
            s3_key: S3 object key
            expires_in_seconds: URL expiration time
            
        Returns:
            Presigned URL
        """
        try:
            from datetime import timedelta
            
            url = self.client.presigned_get_object(
                self.bucket_name,
                s3_key,
                expires=timedelta(seconds=expires_in_seconds)
            )
            
            logger.info(f"Generated presigned URL for {s3_key}")
            return url
            
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise
    
    def _get_content_type(self, file_extension: str) -> str:
        """Get content type based on file extension"""
        content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.webm': 'video/webm',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac'
        }
        return content_types.get(file_extension.lower(), 'application/octet-stream')
    
    def _get_video_duration(self, file_path: str) -> Optional[int]:
        """Get video duration in seconds (placeholder - requires ffmpeg)"""
        # This would require ffmpeg/ffprobe in production
        # For now, return None or estimate
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return int(float(result.stdout.strip()))
        except Exception as e:
            logger.warning(f"Could not determine video duration: {e}")
        
        return None

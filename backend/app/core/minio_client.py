from minio import Minio
from minio.error import S3Error
from typing import Optional, BinaryIO
import logging
from .config import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    def __init__(self):
        self.client = None
        self.bucket_name = settings.s3_bucket_name
        self._initialized = False
    
    def _initialize_client(self):
        """Lazy initialization of MinIO client"""
        if not self._initialized:
            try:
                self.client = Minio(
                    settings.s3_endpoint,
                    access_key=settings.s3_access_key,
                    secret_key=settings.s3_secret_key,
                    secure=settings.s3_secure
                )
                self._ensure_bucket_exists()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {e}")
                self.client = None
                self._initialized = False
                raise
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        if not self.client:
            return
        
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name, location=settings.s3_region)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def upload_file(self, object_name: str, file_data: BinaryIO, length: int, content_type: str = "application/octet-stream") -> bool:
        """Upload a file to MinIO"""
        try:
            if not self._initialized:
                self._initialize_client()
            
            self.client.put_object(
                self.bucket_name,
                object_name,
                file_data,
                length,
                content_type=content_type
            )
            logger.info(f"Successfully uploaded {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error uploading {object_name}: {e}")
            return False
    
    def upload_file_chunked(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> bool:
        """Upload a file using chunked upload for large files"""
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            logger.info(f"Successfully uploaded {object_name} using chunked upload")
            return True
        except S3Error as e:
            logger.error(f"Error uploading {object_name} with chunked upload: {e}")
            return False
    
    def get_file_url(self, object_name: str, expires_in_seconds: int = 3600) -> Optional[str]:
        """Get a presigned URL for file access"""
        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires_in_seconds
            )
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {object_name}: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """Delete a file from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Successfully deleted {object_name}")
            return True
        except S3Error as e:
            logger.error(f"Error deleting {object_name}: {e}")
            return False
    
    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    def get_file_info(self, object_name: str) -> Optional[dict]:
        """Get file metadata"""
        try:
            stat = self.client.stat_object(self.bucket_name, object_name)
            return {
                "size": stat.size,
                "etag": stat.etag,
                "last_modified": stat.last_modified,
                "content_type": stat.content_type
            }
        except S3Error as e:
            logger.error(f"Error getting file info for {object_name}: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if MinIO is accessible"""
        try:
            if not self._initialized:
                self._initialize_client()
            
            return self.client.bucket_exists(self.bucket_name)
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return False


# Global MinIO client instance
minio_client = MinIOClient()


def get_minio_client() -> MinIOClient:
    """Dependency to get MinIO client"""
    return minio_client
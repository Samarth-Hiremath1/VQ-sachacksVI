import os
import tempfile
import magic
import subprocess
import json
import hashlib
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import logging
from datetime import datetime, timedelta
from fastapi import UploadFile, HTTPException, status

from ..core.config import settings
from ..core.minio_client import MinIOClient
from ..schemas.recording import FileValidationResult, RecordingMetadata

logger = logging.getLogger(__name__)


class FileValidationService:
    def __init__(self):
        self.max_file_size = settings.max_file_size
        self.max_video_duration = settings.max_video_duration
        self.allowed_video_formats = settings.allowed_video_formats
        self.allowed_audio_formats = settings.allowed_audio_formats
    
    def validate_file(self, file: UploadFile) -> FileValidationResult:
        """Validate uploaded file for format, size, and content"""
        errors = []
        
        # Check file size
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            errors.append(f"File size {file.size} exceeds maximum allowed size {self.max_file_size}")
        
        # Detect file type using python-magic
        file_content = file.file.read(2048)  # Read first 2KB for magic detection
        file.file.seek(0)  # Reset file pointer
        
        mime_type = magic.from_buffer(file_content, mime=True)
        file_type = self._get_file_type_from_mime(mime_type)
        
        if file_type not in ['video', 'audio']:
            errors.append(f"Unsupported file type: {mime_type}")
        
        # Validate file format
        file_extension = Path(file.filename).suffix.lower().lstrip('.')
        if file_type == 'video' and file_extension not in self.allowed_video_formats:
            errors.append(f"Video format {file_extension} not allowed. Allowed formats: {self.allowed_video_formats}")
        elif file_type == 'audio' and file_extension not in self.allowed_audio_formats:
            errors.append(f"Audio format {file_extension} not allowed. Allowed formats: {self.allowed_audio_formats}")
        
        # Additional validation for video files
        duration_seconds = None
        if file_type == 'video' and not errors:
            try:
                duration_seconds = self._get_video_duration(file)
                if duration_seconds and duration_seconds > self.max_video_duration:
                    errors.append(f"Video duration {duration_seconds}s exceeds maximum allowed duration {self.max_video_duration}s")
            except Exception as e:
                logger.warning(f"Could not determine video duration: {e}")
        
        return FileValidationResult(
            is_valid=len(errors) == 0,
            file_type=file_type,
            file_size=file.size or 0,
            duration_seconds=duration_seconds,
            format=file_extension,
            errors=errors
        )
    
    def _get_file_type_from_mime(self, mime_type: str) -> str:
        """Determine if file is video or audio based on MIME type"""
        if mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        else:
            return 'unknown'
    
    def _get_video_duration(self, file: UploadFile) -> Optional[int]:
        """Get video duration using ffprobe"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                # Copy file content to temp file
                file.file.seek(0)
                temp_file.write(file.file.read())
                temp_file.flush()
                file.file.seek(0)  # Reset file pointer
                
                # Use ffprobe to get duration
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', temp_file.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    duration = float(data.get('format', {}).get('duration', 0))
                    return int(duration)
                
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return None
    
    def extract_metadata(self, file: UploadFile) -> RecordingMetadata:
        """Extract comprehensive metadata from uploaded file"""
        try:
            # Basic metadata
            metadata = RecordingMetadata(
                filename=file.filename,
                content_type=file.content_type,
                file_size=file.size or 0
            )
            
            # Get file format
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            metadata.format = file_extension
            
            # For video files, extract additional metadata
            if file.content_type and file.content_type.startswith('video/'):
                video_metadata = self._extract_video_metadata(file)
                if video_metadata:
                    metadata.duration_seconds = video_metadata.get('duration_seconds')
                    metadata.resolution = video_metadata.get('resolution')
                    metadata.bitrate = video_metadata.get('bitrate')
                    metadata.codec = video_metadata.get('codec')
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return basic metadata even if extraction fails
            return RecordingMetadata(
                filename=file.filename,
                content_type=file.content_type or 'application/octet-stream',
                file_size=file.size or 0,
                format=Path(file.filename).suffix.lower().lstrip('.')
            )
    
    def _extract_video_metadata(self, file: UploadFile) -> Optional[Dict[str, Any]]:
        """Extract detailed video metadata using ffprobe"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                file.file.seek(0)
                temp_file.write(file.file.read())
                temp_file.flush()
                file.file.seek(0)
                
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', '-show_streams', temp_file.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    
                    # Extract format info
                    format_info = data.get('format', {})
                    duration = float(format_info.get('duration', 0))
                    bitrate = int(format_info.get('bit_rate', 0))
                    
                    # Extract video stream info
                    video_stream = None
                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            video_stream = stream
                            break
                    
                    resolution = None
                    codec = None
                    if video_stream:
                        width = video_stream.get('width')
                        height = video_stream.get('height')
                        if width and height:
                            resolution = f"{width}x{height}"
                        codec = video_stream.get('codec_name')
                    
                    return {
                        'duration_seconds': int(duration),
                        'bitrate': bitrate,
                        'resolution': resolution,
                        'codec': codec
                    }
                
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        return None
    
    def scan_for_viruses(self, file_path: str) -> bool:
        """Basic virus scanning (placeholder for actual antivirus integration)"""
        # In a production environment, integrate with ClamAV or similar
        # For now, just check file size and basic patterns
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False
            
            # Check for suspicious file patterns (basic implementation)
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                # Basic check for executable headers
                if header.startswith(b'MZ') or header.startswith(b'\x7fELF'):
                    logger.warning(f"Suspicious executable header detected in {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during virus scan: {e}")
            return False


class FileStorageService:
    def __init__(self, minio_client: MinIOClient):
        self.minio_client = minio_client
        self.validation_service = FileValidationService()
    
    def generate_s3_key(self, user_id: str, filename: str, file_type: str) -> str:
        """Generate S3 key for file storage"""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        file_hash = hashlib.md5(f"{user_id}{filename}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
        extension = Path(filename).suffix
        return f"{file_type}/{timestamp}/{user_id}/{file_hash}{extension}"
    
    def upload_file(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """Upload file to MinIO with validation"""
        # Validate file
        validation_result = self.validation_service.validate_file(file)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "FILE_VALIDATION_FAILED",
                    "message": "File validation failed",
                    "errors": validation_result.errors
                }
            )
        
        # Extract metadata
        metadata = self.validation_service.extract_metadata(file)
        
        # Generate S3 key
        s3_key = self.generate_s3_key(user_id, file.filename, validation_result.file_type)
        
        # Upload to MinIO
        try:
            file.file.seek(0)
            success = self.minio_client.upload_file(
                s3_key,
                file.file,
                file.size,
                file.content_type
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error_code": "UPLOAD_FAILED",
                        "message": "Failed to upload file to storage"
                    }
                )
            
            return {
                "s3_key": s3_key,
                "metadata": metadata,
                "validation_result": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "UPLOAD_ERROR",
                    "message": "An error occurred during file upload"
                }
            )
    
    def cleanup_old_files(self) -> Dict[str, int]:
        """Clean up old temporary and failed upload files"""
        # This would be implemented as a background task
        # For now, return placeholder stats
        return {
            "temp_files_cleaned": 0,
            "failed_uploads_cleaned": 0,
            "storage_freed_bytes": 0
        }
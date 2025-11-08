"""
Video processing utilities

Handles video processing operations like audio extraction.
Requirements: 2.1, 3.5
"""

import os
import logging
import subprocess
from typing import Dict, Any
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        logger.info("Initialized VideoProcessor")
    
    def extract_audio(self, video_path: str, output_format: str = 'wav') -> Dict[str, Any]:
        """
        Extract audio track from video file
        
        Args:
            video_path: Path to video file
            output_format: Audio format (wav, mp3, etc.)
            
        Returns:
            Dict with audio file path and metadata
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Generate output path
            video_name = Path(video_path).stem
            audio_path = os.path.join(self.temp_dir, f"{video_name}_audio.{output_format}")
            
            logger.info(f"Extracting audio from {video_path} to {audio_path}")
            
            # Use ffmpeg to extract audio
            # -vn: no video
            # -acodec: audio codec
            # -ar: audio sample rate
            # -ac: audio channels
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le' if output_format == 'wav' else 'libmp3lame',
                '-ar', '16000',  # 16kHz sample rate for speech analysis
                '-ac', '1',  # Mono channel
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Failed to extract audio: {result.stderr}")
            
            if not os.path.exists(audio_path):
                raise Exception("Audio file was not created")
            
            # Get audio metadata
            duration = self._get_audio_duration(audio_path)
            file_size = os.path.getsize(audio_path)
            
            logger.info(f"Successfully extracted audio: {audio_path} ({file_size} bytes, {duration}s)")
            
            return {
                'audio_path': audio_path,
                'sample_rate': 16000,
                'channels': 1,
                'duration': duration,
                'file_size': file_size,
                'format': output_format
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout extracting audio from {video_path}")
            raise Exception("Audio extraction timeout")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def _get_audio_duration(self, audio_path: str) -> int:
        """Get audio duration in seconds"""
        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return int(float(result.stdout.strip()))
            else:
                logger.warning(f"Could not determine audio duration: {result.stderr}")
                return 0
                
        except Exception as e:
            logger.warning(f"Error getting audio duration: {e}")
            return 0
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video file information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video metadata
        """
        try:
            command = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration,bit_rate,codec_name',
                '-show_entries', 'format=duration,size',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                stream = data.get('streams', [{}])[0]
                format_info = data.get('format', {})
                
                return {
                    'width': stream.get('width'),
                    'height': stream.get('height'),
                    'duration': int(float(format_info.get('duration', 0))),
                    'size': int(format_info.get('size', 0)),
                    'codec': stream.get('codec_name'),
                    'bit_rate': stream.get('bit_rate')
                }
            else:
                logger.warning(f"Could not get video info: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.warning(f"Error getting video info: {e}")
            return {}
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {file_path}: {e}")

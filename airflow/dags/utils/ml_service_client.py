"""
ML Service Clients for PyTorch and TensorFlow services

Handles communication with ML microservices for body language and speech analysis.
Requirements: 2.1, 2.2, 2.3
"""

import requests
import logging
from typing import Dict, Any, Optional
import os
import json

logger = logging.getLogger(__name__)


class MLServiceClient:
    """Base class for ML service clients"""
    
    def __init__(self, service_url: str, timeout: int = 300):
        self.service_url = service_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if the ML service is healthy"""
        try:
            response = self.session.get(
                f"{self.service_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {self.service_url}: {e}")
            return False


class PyTorchBodyLanguageClient(MLServiceClient):
    """Client for PyTorch body language analysis service"""
    
    def __init__(self):
        service_url = os.getenv('PYTORCH_SERVICE_URL', 'http://pytorch-service:8000')
        super().__init__(service_url)
        logger.info(f"Initialized PyTorch client with URL: {service_url}")
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze body language from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict containing body language analysis results
        """
        try:
            logger.info(f"Sending video for body language analysis: {video_path}")
            
            # Open video file
            with open(video_path, 'rb') as video_file:
                files = {'file': video_file}
                
                response = self.session.post(
                    f"{self.service_url}/api/v1/analyze/body-language",
                    files=files,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Body language analysis completed. Score: {result.get('overall_score', 'N/A')}")
                return result
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout analyzing video: {video_path}")
            raise Exception("PyTorch service timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling PyTorch service: {e}")
            raise Exception(f"PyTorch service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in body language analysis: {e}")
            raise
    
    def analyze_pose_landmarks(self, landmarks: list) -> Dict[str, Any]:
        """
        Analyze pre-extracted pose landmarks
        
        Args:
            landmarks: List of pose landmark data
            
        Returns:
            Dict containing body language analysis results
        """
        try:
            logger.info(f"Analyzing {len(landmarks)} pose landmarks")
            
            response = self.session.post(
                f"{self.service_url}/api/v1/analyze/pose-landmarks",
                json={'landmarks': landmarks},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Pose landmark analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing pose landmarks: {e}")
            raise


class TensorFlowSpeechClient(MLServiceClient):
    """Client for TensorFlow speech analysis service"""
    
    def __init__(self):
        service_url = os.getenv('TENSORFLOW_SERVICE_URL', 'http://tensorflow-service:8000')
        super().__init__(service_url)
        logger.info(f"Initialized TensorFlow client with URL: {service_url}")
    
    def analyze_audio(self, audio_path: str, transcript: str = "") -> Dict[str, Any]:
        """
        Analyze speech quality from audio file
        
        Args:
            audio_path: Path to audio file
            transcript: Optional transcript text
            
        Returns:
            Dict containing speech analysis results
        """
        try:
            logger.info(f"Sending audio for speech analysis: {audio_path}")
            
            # Open audio file
            with open(audio_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {'transcript': transcript} if transcript else {}
                
                response = self.session.post(
                    f"{self.service_url}/api/v1/analyze/speech",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Speech analysis completed. Quality: {result.get('overall_quality', 'N/A')}")
                return result
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout analyzing audio: {audio_path}")
            raise Exception("TensorFlow service timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling TensorFlow service: {e}")
            raise Exception(f"TensorFlow service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in speech analysis: {e}")
            raise
    
    def detect_filler_words(self, transcript: str) -> Dict[str, Any]:
        """
        Detect filler words in transcript
        
        Args:
            transcript: Text transcript
            
        Returns:
            Dict containing filler word analysis
        """
        try:
            logger.info(f"Detecting filler words in transcript")
            
            response = self.session.post(
                f"{self.service_url}/api/v1/analyze/filler-words",
                json={'transcript': transcript},
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Filler word detection completed")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting filler words: {e}")
            raise

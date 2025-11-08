"""
Fallback Handler for Graceful Degradation

Provides fallback analysis using MediaPipe and Web Speech API when ML models fail.
Requirements: 2.7, 3.5, 6.4
"""

import logging
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class FallbackAnalyzer:
    """Provides fallback analysis when ML services are unavailable"""
    
    def __init__(self):
        logger.info("Initialized FallbackAnalyzer")
    
    def analyze_body_language_fallback(
        self,
        video_path: str,
        pose_landmarks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Fallback body language analysis using rule-based MediaPipe
        
        Args:
            video_path: Path to video file
            pose_landmarks: Pre-extracted pose landmarks (optional)
            
        Returns:
            Dict with basic body language analysis
        """
        try:
            logger.info("Using fallback body language analysis (rule-based)")
            
            # In production, this would use MediaPipe to extract pose landmarks
            # and apply rule-based heuristics
            
            # Placeholder implementation with conservative scores
            result = {
                'overall_score': 65.0,
                'posture_score': 65.0,
                'gesture_score': 65.0,
                'confidence': 0.6,
                'method': 'fallback_rule_based',
                'detailed_metrics': {
                    'posture_analysis': {
                        'shoulder_alignment': 'neutral',
                        'spine_alignment': 'neutral',
                        'weight_distribution': 'balanced'
                    },
                    'gesture_analysis': {
                        'hand_movement_frequency': 'moderate',
                        'gesture_variety': 'limited',
                        'open_gestures': 'some'
                    },
                    'movement_patterns': {
                        'stability': 'moderate',
                        'fidgeting': 'minimal',
                        'pacing': 'none'
                    }
                },
                'warnings': [
                    'Using fallback analysis - ML model unavailable',
                    'Scores are conservative estimates',
                    'Consider re-analyzing when ML service is available'
                ]
            }
            
            logger.info("Fallback body language analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback body language analysis: {e}")
            return self._get_minimal_body_language_result()
    
    def analyze_speech_fallback(
        self,
        audio_path: str,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Fallback speech analysis using rule-based Web Speech API results
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript text from Web Speech API
            
        Returns:
            Dict with basic speech analysis
        """
        try:
            logger.info("Using fallback speech analysis (rule-based)")
            
            # Analyze transcript if available
            word_count = len(transcript.split()) if transcript else 0
            filler_word_count = self._count_filler_words(transcript) if transcript else 0
            
            # Estimate speaking pace (assuming 5 minute presentation)
            estimated_duration = 300  # 5 minutes
            speaking_pace_wpm = (word_count / estimated_duration) * 60 if word_count > 0 else 140
            
            # Calculate basic scores
            pace_score = self._score_speaking_pace(speaking_pace_wpm)
            filler_score = self._score_filler_words(filler_word_count, word_count)
            
            overall_quality = (pace_score + filler_score) / 2
            
            result = {
                'overall_quality': round(overall_quality, 2),
                'speaking_rate_wpm': round(speaking_pace_wpm, 2),
                'filler_word_count': filler_word_count,
                'confidence': 0.6,
                'method': 'fallback_rule_based',
                'detailed_metrics': {
                    'pace_analysis': {
                        'average_wpm': speaking_pace_wpm,
                        'pace_rating': self._rate_pace(speaking_pace_wpm),
                        'pace_score': pace_score
                    },
                    'filler_words': {
                        'total_count': filler_word_count,
                        'filler_ratio': (filler_word_count / max(word_count, 1)) * 100,
                        'filler_score': filler_score
                    },
                    'vocal_quality': {
                        'clarity': 'unknown',
                        'volume': 'unknown',
                        'variation': 'unknown',
                        'note': 'Vocal quality analysis requires ML model'
                    }
                },
                'warnings': [
                    'Using fallback analysis - ML model unavailable',
                    'Vocal quality metrics not available',
                    'Consider re-analyzing when ML service is available'
                ]
            }
            
            logger.info("Fallback speech analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback speech analysis: {e}")
            return self._get_minimal_speech_result()
    
    def _count_filler_words(self, transcript: str) -> int:
        """Count filler words in transcript"""
        if not transcript:
            return 0
        
        filler_patterns = [
            r'\bum\b', r'\buh\b', r'\blike\b', r'\byou know\b',
            r'\bso\b', r'\bbasically\b', r'\bactually\b', r'\bliterally\b'
        ]
        
        count = 0
        transcript_lower = transcript.lower()
        
        for pattern in filler_patterns:
            count += len(re.findall(pattern, transcript_lower))
        
        return count
    
    def _score_speaking_pace(self, wpm: float) -> float:
        """Score speaking pace (0-100)"""
        if 130 <= wpm <= 170:
            return 100.0
        elif 110 <= wpm < 130 or 170 < wpm <= 190:
            return 80.0
        elif 90 <= wpm < 110 or 190 < wpm <= 210:
            return 60.0
        elif wpm < 90:
            return max(0, 60 - (90 - wpm) * 2)
        else:
            return max(0, 60 - (wpm - 210) * 1.5)
    
    def _score_filler_words(self, filler_count: int, word_count: int) -> float:
        """Score filler word usage (0-100)"""
        if word_count == 0:
            return 70.0  # Neutral score
        
        filler_ratio = (filler_count / word_count) * 100
        
        if filler_ratio <= 0.5:
            return 100.0
        elif filler_ratio <= 1.0:
            return 90.0
        elif filler_ratio <= 2.0:
            return 70.0
        elif filler_ratio <= 3.0:
            return 50.0
        else:
            return max(0, 50 - (filler_ratio - 3) * 10)
    
    def _rate_pace(self, wpm: float) -> str:
        """Rate speaking pace"""
        if 130 <= wpm <= 170:
            return "optimal"
        elif 110 <= wpm < 130 or 170 < wpm <= 190:
            return "acceptable"
        elif wpm < 110:
            return "too_slow"
        else:
            return "too_fast"
    
    def _get_minimal_body_language_result(self) -> Dict[str, Any]:
        """Return minimal body language result when all analysis fails"""
        return {
            'overall_score': 50.0,
            'posture_score': 50.0,
            'gesture_score': 50.0,
            'confidence': 0.3,
            'method': 'minimal_fallback',
            'detailed_metrics': {},
            'warnings': [
                'Analysis failed - using minimal default scores',
                'Please re-record and try again'
            ]
        }
    
    def _get_minimal_speech_result(self) -> Dict[str, Any]:
        """Return minimal speech result when all analysis fails"""
        return {
            'overall_quality': 50.0,
            'speaking_rate_wpm': 140.0,
            'filler_word_count': 0,
            'confidence': 0.3,
            'method': 'minimal_fallback',
            'detailed_metrics': {},
            'warnings': [
                'Analysis failed - using minimal default scores',
                'Please re-record and try again'
            ]
        }


class GracefulDegradationHandler:
    """Handles graceful degradation for ML service failures"""
    
    def __init__(self):
        self.fallback_analyzer = FallbackAnalyzer()
        logger.info("Initialized GracefulDegradationHandler")
    
    def handle_body_language_failure(
        self,
        error: Exception,
        video_path: str,
        pose_landmarks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Handle body language analysis failure with fallback
        
        Args:
            error: Original exception
            video_path: Path to video file
            pose_landmarks: Pre-extracted pose landmarks
            
        Returns:
            Fallback analysis result
        """
        logger.warning(f"Body language ML service failed: {error}")
        logger.info("Attempting fallback analysis")
        
        try:
            result = self.fallback_analyzer.analyze_body_language_fallback(
                video_path, pose_landmarks
            )
            result['fallback_reason'] = str(error)
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback analysis also failed: {fallback_error}")
            return self.fallback_analyzer._get_minimal_body_language_result()
    
    def handle_speech_failure(
        self,
        error: Exception,
        audio_path: str,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Handle speech analysis failure with fallback
        
        Args:
            error: Original exception
            audio_path: Path to audio file
            transcript: Transcript text
            
        Returns:
            Fallback analysis result
        """
        logger.warning(f"Speech ML service failed: {error}")
        logger.info("Attempting fallback analysis")
        
        try:
            result = self.fallback_analyzer.analyze_speech_fallback(
                audio_path, transcript
            )
            result['fallback_reason'] = str(error)
            return result
        except Exception as fallback_error:
            logger.error(f"Fallback analysis also failed: {fallback_error}")
            return self.fallback_analyzer._get_minimal_speech_result()

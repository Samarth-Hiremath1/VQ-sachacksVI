import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime

from ..models.speech_quality_analyzer import SpeechQualityAnalyzer, SpeechQualityMetrics
from ..models.filler_word_detector import FillerWordDetector, FillerWordAnalysis
from .audio_processor import AudioProcessor
from .web_speech_integration import WebSpeechAPIIntegration, EnhancedTranscriptionResult

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveSpeechAnalysis:
    """Complete speech analysis result combining all analysis components."""
    
    # Core analysis results
    speech_quality: SpeechQualityMetrics
    filler_word_analysis: FillerWordAnalysis
    transcription_result: EnhancedTranscriptionResult
    
    # Hybrid analysis metrics
    overall_presentation_score: float
    confidence_level: float
    
    # Recommendations and feedback
    recommendations: List[str]
    real_time_feedback: Dict[str, Any]
    
    # Metadata
    processing_time: float
    analysis_timestamp: str
    model_versions: Dict[str, str]

class SpeechAnalysisService:
    """
    Comprehensive speech analysis service that combines TensorFlow models,
    Web Speech API integration, and provides hybrid analysis results.
    """
    
    def __init__(self):
        # Initialize all analysis components
        self.speech_analyzer = SpeechQualityAnalyzer()
        self.filler_detector = FillerWordDetector()
        self.audio_processor = AudioProcessor()
        self.web_speech_integration = WebSpeechAPIIntegration()
        
        # Analysis configuration
        self.config = {
            'use_ml_enhancement': True,
            'enable_real_time_feedback': True,
            'confidence_threshold': 0.7,
            'quality_weights': {
                'speech_quality': 0.4,
                'transcription_quality': 0.3,
                'filler_word_penalty': 0.3
            }
        }
        
        logger.info("Speech Analysis Service initialized")
    
    async def analyze_audio_comprehensive(self, 
                                        audio_data: np.ndarray,
                                        transcript: Optional[str] = None,
                                        enable_real_time: bool = False) -> ComprehensiveSpeechAnalysis:
        """
        Perform comprehensive speech analysis combining all components.
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_data, 16000)
            
            # Step 2: Run parallel analysis tasks
            analysis_tasks = await self._run_parallel_analysis(processed_audio, transcript)
            
            # Step 3: Combine results into hybrid analysis
            hybrid_result = self._create_hybrid_analysis(analysis_tasks, processed_audio)
            
            # Step 4: Generate comprehensive recommendations
            recommendations = self._generate_comprehensive_recommendations(hybrid_result)
            
            # Step 5: Calculate overall scores
            overall_score, confidence = self._calculate_overall_metrics(hybrid_result)
            
            # Step 6: Real-time feedback (if enabled)
            real_time_feedback = {}
            if enable_real_time:
                real_time_feedback = self.web_speech_integration.get_real_time_feedback(
                    transcript or "", processed_audio[-1600:]  # Last 0.1 seconds
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ComprehensiveSpeechAnalysis(
                speech_quality=hybrid_result['speech_quality'],
                filler_word_analysis=hybrid_result['filler_analysis'],
                transcription_result=hybrid_result['transcription_result'],
                overall_presentation_score=overall_score,
                confidence_level=confidence,
                recommendations=recommendations,
                real_time_feedback=real_time_feedback,
                processing_time=processing_time,
                analysis_timestamp=datetime.now().isoformat(),
                model_versions={
                    'speech_analyzer': '1.0.0',
                    'filler_detector': '1.0.0',
                    'audio_processor': '1.0.0',
                    'web_speech_integration': '1.0.0'
                }
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Return fallback analysis
            return await self._create_fallback_analysis(audio_data, transcript, start_time)
    
    async def _run_parallel_analysis(self, audio_data: np.ndarray, 
                                   transcript: Optional[str]) -> Dict[str, Any]:
        """Run analysis components in parallel for better performance."""
        
        async def run_speech_quality_analysis():
            """Run TensorFlow speech quality analysis."""
            try:
                return self.speech_analyzer.analyze_speech_quality(audio_data, transcript or "")
            except Exception as e:
                logger.error(f"Speech quality analysis failed: {e}")
                return None
        
        async def run_filler_word_analysis():
            """Run filler word detection."""
            try:
                if not transcript:
                    return None
                
                if self.config['use_ml_enhancement']:
                    return self.filler_detector.detect_filler_words_ml_enhanced(transcript)
                else:
                    return self.filler_detector.detect_filler_words_rule_based(transcript)
            except Exception as e:
                logger.error(f"Filler word analysis failed: {e}")
                return None
        
        async def run_transcription_analysis():
            """Run Web Speech API simulation and enhancement."""
            try:
                # Simulate Web Speech API results
                web_speech_results = self.web_speech_integration.simulate_web_speech_api_result(
                    audio_data, 16000
                )
                
                # Enhance with TensorFlow
                enhanced_result = self.web_speech_integration.enhance_transcription_with_tensorflow(
                    web_speech_results, audio_data
                )
                
                return enhanced_result
            except Exception as e:
                logger.error(f"Transcription analysis failed: {e}")
                return None
        
        # Run analyses concurrently
        try:
            speech_quality_task = asyncio.create_task(run_speech_quality_analysis())
            filler_analysis_task = asyncio.create_task(run_filler_word_analysis())
            transcription_task = asyncio.create_task(run_transcription_analysis())
            
            # Wait for all tasks to complete
            speech_quality, filler_analysis, transcription_result = await asyncio.gather(
                speech_quality_task,
                filler_analysis_task,
                transcription_task,
                return_exceptions=True
            )
            
            return {
                'speech_quality': speech_quality,
                'filler_analysis': filler_analysis,
                'transcription_result': transcription_result
            }
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {e}")
            # Fallback to sequential execution
            return {
                'speech_quality': await run_speech_quality_analysis(),
                'filler_analysis': await run_filler_word_analysis(),
                'transcription_result': await run_transcription_analysis()
            }
    
    def _create_hybrid_analysis(self, analysis_results: Dict[str, Any], 
                              audio_data: np.ndarray) -> Dict[str, Any]:
        """Create hybrid analysis combining all results."""
        
        # Extract individual results
        speech_quality = analysis_results.get('speech_quality')
        filler_analysis = analysis_results.get('filler_analysis')
        transcription_result = analysis_results.get('transcription_result')
        
        # Create hybrid result with fallbacks
        hybrid_result = {}
        
        # Speech quality with fallback
        if speech_quality and hasattr(speech_quality, 'overall_quality'):
            hybrid_result['speech_quality'] = speech_quality
        else:
            # Create fallback speech quality metrics
            hybrid_result['speech_quality'] = self._create_fallback_speech_quality()
        
        # Filler analysis with fallback
        if filler_analysis and hasattr(filler_analysis, 'total_filler_words'):
            hybrid_result['filler_analysis'] = filler_analysis
        else:
            # Create fallback filler analysis
            hybrid_result['filler_analysis'] = self._create_fallback_filler_analysis()
        
        # Transcription result with fallback
        if transcription_result and hasattr(transcription_result, 'enhanced_transcript'):
            hybrid_result['transcription_result'] = transcription_result
        else:
            # Create fallback transcription result
            hybrid_result['transcription_result'] = self._create_fallback_transcription_result()
        
        # Create combined analysis using Web Speech Integration
        if speech_quality:
            tensorflow_analysis = {
                'clarity_score': speech_quality.clarity_score,
                'volume_variation_score': speech_quality.volume_variation_score,
                'pace_score': speech_quality.pace_score,
                'overall_quality': speech_quality.overall_quality
            }
            
            if transcription_result:
                hybrid_analysis = self.web_speech_integration.create_hybrid_analysis_result(
                    transcription_result, tensorflow_analysis
                )
                hybrid_result['combined_analysis'] = hybrid_analysis
        
        return hybrid_result
    
    def _generate_comprehensive_recommendations(self, hybrid_result: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []
        
        try:
            # Speech quality recommendations
            speech_quality = hybrid_result.get('speech_quality')
            if speech_quality:
                pace_recommendations = self.speech_analyzer.get_pace_recommendations(
                    speech_quality.speaking_rate_wpm
                )
                recommendations.extend(pace_recommendations)
                
                # Volume and clarity recommendations
                if speech_quality.volume_consistency < 0.6:
                    recommendations.append("Try to maintain more consistent volume throughout your speech")
                
                if speech_quality.clarity_score < 0.7:
                    recommendations.append("Focus on speaking more clearly and articulating words")
                
                if speech_quality.pitch_variation < 0.4:
                    recommendations.append("Add more vocal variety to keep your audience engaged")
                elif speech_quality.pitch_variation > 0.8:
                    recommendations.append("Try to moderate your pitch variation for better clarity")
            
            # Filler word recommendations
            filler_analysis = hybrid_result.get('filler_analysis')
            if filler_analysis:
                filler_recommendations = self.filler_detector.get_filler_word_recommendations(
                    filler_analysis
                )
                recommendations.extend(filler_recommendations)
            
            # Transcription quality recommendations
            transcription_result = hybrid_result.get('transcription_result')
            if transcription_result and hasattr(transcription_result, 'quality_metrics'):
                quality_metrics = transcription_result.quality_metrics
                
                if quality_metrics.get('average_confidence', 1.0) < 0.8:
                    recommendations.append("Speak more clearly to improve speech recognition accuracy")
                
                if quality_metrics.get('speech_coverage', 1.0) < 0.7:
                    recommendations.append("Reduce long pauses to maintain better flow")
            
            # Combined analysis recommendations
            combined_analysis = hybrid_result.get('combined_analysis', {})
            hybrid_metrics = combined_analysis.get('hybrid_metrics', {})
            
            overall_assessment = hybrid_metrics.get('overall_assessment', 0.5)
            if overall_assessment < 0.6:
                recommendations.append("Consider practicing your presentation to improve overall delivery")
            elif overall_assessment > 0.8:
                recommendations.append("Great job! Your presentation skills are strong")
            
            # Remove duplicates and limit recommendations
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:8]  # Limit to top 8 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Continue practicing to improve your presentation skills"]
    
    def _calculate_overall_metrics(self, hybrid_result: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate overall presentation score and confidence level."""
        
        try:
            weights = self.config['quality_weights']
            
            # Speech quality score
            speech_quality = hybrid_result.get('speech_quality')
            speech_score = speech_quality.overall_quality if speech_quality else 0.5
            
            # Transcription quality score
            transcription_result = hybrid_result.get('transcription_result')
            transcription_score = 0.5
            if transcription_result and hasattr(transcription_result, 'quality_metrics'):
                transcription_score = transcription_result.quality_metrics.get('overall_quality', 0.5)
            
            # Filler word penalty
            filler_analysis = hybrid_result.get('filler_analysis')
            filler_penalty = 0.0
            if filler_analysis:
                # Penalty based on filler word rate (fillers per minute)
                filler_rate = filler_analysis.filler_word_rate
                if filler_rate > 10:
                    filler_penalty = 0.3
                elif filler_rate > 5:
                    filler_penalty = 0.15
                elif filler_rate > 2:
                    filler_penalty = 0.05
            
            # Calculate weighted overall score
            overall_score = (
                speech_score * weights['speech_quality'] +
                transcription_score * weights['transcription_quality'] -
                filler_penalty * weights['filler_word_penalty']
            )
            
            # Ensure score is between 0 and 1
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Calculate confidence level based on individual component confidences
            confidences = []
            
            if speech_quality:
                # Use average of individual scores as confidence proxy
                speech_confidence = (speech_quality.clarity_score + 
                                   speech_quality.volume_consistency + 
                                   speech_quality.pace_score) / 3
                confidences.append(speech_confidence)
            
            if transcription_result and hasattr(transcription_result, 'confidence_score'):
                confidences.append(transcription_result.confidence_score)
            
            if filler_analysis and filler_analysis.confidence_scores:
                avg_filler_confidence = np.mean(filler_analysis.confidence_scores)
                confidences.append(avg_filler_confidence)
            
            # Calculate overall confidence
            confidence_level = np.mean(confidences) if confidences else 0.5
            
            return overall_score, confidence_level
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return 0.5, 0.5
    
    def _create_fallback_speech_quality(self) -> SpeechQualityMetrics:
        """Create fallback speech quality metrics."""
        return SpeechQualityMetrics(
            clarity_score=0.5,
            volume_variation_score=0.5,
            pace_score=0.5,
            overall_quality=0.5,
            speaking_rate_wpm=120.0,
            volume_consistency=0.5,
            pitch_variation=0.5
        )
    
    def _create_fallback_filler_analysis(self) -> FillerWordAnalysis:
        """Create fallback filler word analysis."""
        return FillerWordAnalysis(
            total_filler_words=0,
            filler_word_rate=0.0,
            filler_word_types={},
            filler_word_positions=[],
            confidence_scores=[]
        )
    
    def _create_fallback_transcription_result(self) -> EnhancedTranscriptionResult:
        """Create fallback transcription result."""
        return EnhancedTranscriptionResult(
            original_transcript="",
            enhanced_transcript="",
            confidence_score=0.5,
            word_timestamps=[],
            speech_segments=[],
            quality_metrics={'overall_quality': 0.5}
        )
    
    async def _create_fallback_analysis(self, audio_data: np.ndarray, 
                                      transcript: Optional[str],
                                      start_time: datetime) -> ComprehensiveSpeechAnalysis:
        """Create fallback analysis when main analysis fails."""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ComprehensiveSpeechAnalysis(
            speech_quality=self._create_fallback_speech_quality(),
            filler_word_analysis=self._create_fallback_filler_analysis(),
            transcription_result=self._create_fallback_transcription_result(),
            overall_presentation_score=0.5,
            confidence_level=0.3,
            recommendations=["Analysis encountered issues. Please try again with clearer audio."],
            real_time_feedback={},
            processing_time=processing_time,
            analysis_timestamp=datetime.now().isoformat(),
            model_versions={
                'speech_analyzer': '1.0.0-fallback',
                'filler_detector': '1.0.0-fallback',
                'audio_processor': '1.0.0-fallback',
                'web_speech_integration': '1.0.0-fallback'
            }
        )
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update service configuration."""
        self.config.update(new_config)
        logger.info(f"Configuration updated: {self.config}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and health."""
        try:
            # Test each component
            test_audio = np.random.random(1600)  # 0.1 second test
            
            status = {
                'service_name': 'SpeechAnalysisService',
                'version': '1.0.0',
                'status': 'healthy',
                'components': {},
                'configuration': self.config,
                'last_check': datetime.now().isoformat()
            }
            
            # Test speech analyzer
            try:
                _ = self.speech_analyzer.analyze_speech_quality(test_audio)
                status['components']['speech_analyzer'] = 'healthy'
            except Exception as e:
                status['components']['speech_analyzer'] = f'error: {str(e)}'
                status['status'] = 'degraded'
            
            # Test filler detector
            try:
                _ = self.filler_detector.detect_filler_words_rule_based("test transcript")
                status['components']['filler_detector'] = 'healthy'
            except Exception as e:
                status['components']['filler_detector'] = f'error: {str(e)}'
                status['status'] = 'degraded'
            
            # Test audio processor
            try:
                _ = self.audio_processor.preprocess_audio(test_audio, 16000)
                status['components']['audio_processor'] = 'healthy'
            except Exception as e:
                status['components']['audio_processor'] = f'error: {str(e)}'
                status['status'] = 'degraded'
            
            # Test web speech integration
            try:
                _ = self.web_speech_integration.simulate_web_speech_api_result(test_audio)
                status['components']['web_speech_integration'] = 'healthy'
            except Exception as e:
                status['components']['web_speech_integration'] = f'error: {str(e)}'
                status['status'] = 'degraded'
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking service status: {e}")
            return {
                'service_name': 'SpeechAnalysisService',
                'version': '1.0.0',
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
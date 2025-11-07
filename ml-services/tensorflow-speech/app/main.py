from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import librosa
import tempfile
import os
import logging
from datetime import datetime
import mlflow
import mlflow.tensorflow

from .models.speech_quality_analyzer import SpeechQualityAnalyzer, SpeechQualityMetrics
from .models.filler_word_detector import FillerWordDetector, FillerWordAnalysis
from .services.audio_processor import AudioProcessor
from .services.web_speech_integration import WebSpeechAPIIntegration, EnhancedTranscriptionResult
from .services.speech_analysis_service import SpeechAnalysisService, ComprehensiveSpeechAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TensorFlow Speech Analysis Service",
    description="AI-powered speech analysis for presentation coaching with hybrid Web Speech API integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
speech_analysis_service = SpeechAnalysisService()
# Keep individual services for backward compatibility
speech_analyzer = speech_analysis_service.speech_analyzer
filler_detector = speech_analysis_service.filler_detector
audio_processor = speech_analysis_service.audio_processor
web_speech_integration = speech_analysis_service.web_speech_integration

# Pydantic models for API
class SpeechAnalysisRequest(BaseModel):
    transcript: Optional[str] = ""
    use_ml_enhancement: bool = True
    include_filler_analysis: bool = True

class SpeechAnalysisResponse(BaseModel):
    speech_quality: Dict[str, Any]
    filler_word_analysis: Dict[str, Any]
    transcription_result: Optional[Dict[str, Any]] = None
    processing_time: float
    model_versions: Dict[str, str]

class RealTimeFeedbackRequest(BaseModel):
    current_transcript: str
    audio_chunk_base64: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "TensorFlow Speech Analysis Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/analyze-speech",
            "/analyze-audio-file", 
            "/real-time-feedback",
            "/health",
            "/docs"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test model availability
        models_status = {
            "speech_quality_analyzer": speech_analyzer is not None,
            "filler_word_detector": filler_detector is not None,
            "audio_processor": audio_processor is not None,
            "web_speech_integration": web_speech_integration is not None
        }
        
        # Test a simple prediction to ensure models work
        test_audio = np.random.random(16000)  # 1 second of random audio
        try:
            _ = speech_analyzer.analyze_speech_quality(test_audio)
            models_status["speech_models_functional"] = True
        except Exception as e:
            logger.warning(f"Speech models test failed: {e}")
            models_status["speech_models_functional"] = False
        
        status = "healthy" if all(models_status.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_status,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/analyze-audio-file", response_model=SpeechAnalysisResponse)
async def analyze_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcript: Optional[str] = "",
    use_ml_enhancement: bool = True,
    include_filler_analysis: bool = True
):
    """Analyze uploaded audio file for speech quality and filler words."""
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process audio file
            audio_data, sample_rate = audio_processor.load_and_preprocess_audio(temp_file_path)
            
            # Analyze speech quality using TensorFlow models
            speech_quality = speech_analyzer.analyze_speech_quality(audio_data, transcript)
            
            # Analyze filler words if requested
            filler_analysis = None
            if include_filler_analysis:
                if use_ml_enhancement:
                    filler_analysis = filler_detector.detect_filler_words_ml_enhanced(transcript)
                else:
                    filler_analysis = filler_detector.detect_filler_words_rule_based(transcript)
            
            # Simulate Web Speech API and enhance with TensorFlow
            transcription_result = None
            if use_ml_enhancement:
                web_speech_results = web_speech_integration.simulate_web_speech_api_result(audio_data, sample_rate)
                enhanced_transcription = web_speech_integration.enhance_transcription_with_tensorflow(
                    web_speech_results, audio_data
                )
                
                # Create hybrid analysis
                tensorflow_analysis = {
                    'clarity_score': speech_quality.clarity_score,
                    'volume_variation_score': speech_quality.volume_variation_score,
                    'pace_score': speech_quality.pace_score,
                    'overall_quality': speech_quality.overall_quality
                }
                
                transcription_result = web_speech_integration.create_hybrid_analysis_result(
                    enhanced_transcription, tensorflow_analysis
                )
            
            # Log to MLflow
            background_tasks.add_task(log_analysis_to_mlflow, {
                'speech_quality': speech_quality,
                'filler_analysis': filler_analysis,
                'file_name': file.filename,
                'processing_time': (datetime.now() - start_time).total_seconds()
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SpeechAnalysisResponse(
                speech_quality={
                    'clarity_score': speech_quality.clarity_score,
                    'volume_variation_score': speech_quality.volume_variation_score,
                    'pace_score': speech_quality.pace_score,
                    'overall_quality': speech_quality.overall_quality,
                    'speaking_rate_wpm': speech_quality.speaking_rate_wpm,
                    'volume_consistency': speech_quality.volume_consistency,
                    'pitch_variation': speech_quality.pitch_variation,
                    'recommendations': speech_analyzer.get_pace_recommendations(speech_quality.speaking_rate_wpm)
                },
                filler_word_analysis={
                    'total_filler_words': filler_analysis.total_filler_words if filler_analysis else 0,
                    'filler_word_rate': filler_analysis.filler_word_rate if filler_analysis else 0.0,
                    'filler_word_types': filler_analysis.filler_word_types if filler_analysis else {},
                    'filler_word_positions': filler_analysis.filler_word_positions if filler_analysis else [],
                    'confidence_scores': filler_analysis.confidence_scores if filler_analysis else [],
                    'recommendations': filler_detector.get_filler_word_recommendations(filler_analysis) if filler_analysis else []
                } if include_filler_analysis else {},
                transcription_result=transcription_result,
                processing_time=processing_time,
                model_versions={
                    'speech_analyzer': '1.0.0',
                    'filler_detector': '1.0.0',
                    'audio_processor': '1.0.0'
                }
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-speech", response_model=SpeechAnalysisResponse)
async def analyze_speech(
    request: SpeechAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze speech from transcript and optional audio data."""
    start_time = datetime.now()
    
    try:
        # Generate synthetic audio for demonstration if no audio provided
        # In production, this would work with actual audio data
        duration = max(10, len(request.transcript.split()) * 0.5)  # Estimate duration
        synthetic_audio = np.random.random(int(16000 * duration)) * 0.1  # Low amplitude noise
        
        # Analyze speech quality
        speech_quality = speech_analyzer.analyze_speech_quality(synthetic_audio, request.transcript)
        
        # Analyze filler words if requested
        filler_analysis = None
        if request.include_filler_analysis:
            if request.use_ml_enhancement:
                filler_analysis = filler_detector.detect_filler_words_ml_enhanced(request.transcript)
            else:
                filler_analysis = filler_detector.detect_filler_words_rule_based(request.transcript)
        
        # Enhanced transcription analysis
        transcription_result = None
        if request.use_ml_enhancement and request.transcript:
            # Simulate Web Speech API results from transcript
            web_speech_results = web_speech_integration.simulate_web_speech_api_result(synthetic_audio)
            enhanced_transcription = web_speech_integration.enhance_transcription_with_tensorflow(
                web_speech_results, synthetic_audio
            )
            
            tensorflow_analysis = {
                'clarity_score': speech_quality.clarity_score,
                'volume_variation_score': speech_quality.volume_variation_score,
                'pace_score': speech_quality.pace_score,
                'overall_quality': speech_quality.overall_quality
            }
            
            transcription_result = web_speech_integration.create_hybrid_analysis_result(
                enhanced_transcription, tensorflow_analysis
            )
        
        # Log to MLflow
        background_tasks.add_task(log_analysis_to_mlflow, {
            'speech_quality': speech_quality,
            'filler_analysis': filler_analysis,
            'transcript_length': len(request.transcript),
            'processing_time': (datetime.now() - start_time).total_seconds()
        })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SpeechAnalysisResponse(
            speech_quality={
                'clarity_score': speech_quality.clarity_score,
                'volume_variation_score': speech_quality.volume_variation_score,
                'pace_score': speech_quality.pace_score,
                'overall_quality': speech_quality.overall_quality,
                'speaking_rate_wpm': speech_quality.speaking_rate_wpm,
                'volume_consistency': speech_quality.volume_consistency,
                'pitch_variation': speech_quality.pitch_variation,
                'recommendations': speech_analyzer.get_pace_recommendations(speech_quality.speaking_rate_wpm)
            },
            filler_word_analysis={
                'total_filler_words': filler_analysis.total_filler_words if filler_analysis else 0,
                'filler_word_rate': filler_analysis.filler_word_rate if filler_analysis else 0.0,
                'filler_word_types': filler_analysis.filler_word_types if filler_analysis else {},
                'filler_word_positions': filler_analysis.filler_word_positions if filler_analysis else [],
                'confidence_scores': filler_analysis.confidence_scores if filler_analysis else [],
                'recommendations': filler_detector.get_filler_word_recommendations(filler_analysis) if filler_analysis else []
            } if request.include_filler_analysis else {},
            transcription_result=transcription_result,
            processing_time=processing_time,
            model_versions={
                'speech_analyzer': '1.0.0',
                'filler_detector': '1.0.0',
                'audio_processor': '1.0.0'
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing speech: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-comprehensive")
async def analyze_comprehensive(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcript: Optional[str] = "",
    enable_real_time: bool = False
):
    """Comprehensive speech analysis using hybrid TensorFlow + Web Speech API approach."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Load and preprocess audio
            audio_data, sample_rate = audio_processor.load_and_preprocess_audio(temp_file_path)
            
            # Run comprehensive analysis
            analysis_result = await speech_analysis_service.analyze_audio_comprehensive(
                audio_data, transcript, enable_real_time
            )
            
            # Log to MLflow
            background_tasks.add_task(log_comprehensive_analysis_to_mlflow, analysis_result)
            
            # Convert to dict for JSON response
            return {
                'comprehensive_analysis': {
                    'speech_quality': {
                        'clarity_score': analysis_result.speech_quality.clarity_score,
                        'volume_variation_score': analysis_result.speech_quality.volume_variation_score,
                        'pace_score': analysis_result.speech_quality.pace_score,
                        'overall_quality': analysis_result.speech_quality.overall_quality,
                        'speaking_rate_wpm': analysis_result.speech_quality.speaking_rate_wpm,
                        'volume_consistency': analysis_result.speech_quality.volume_consistency,
                        'pitch_variation': analysis_result.speech_quality.pitch_variation
                    },
                    'filler_word_analysis': {
                        'total_filler_words': analysis_result.filler_word_analysis.total_filler_words,
                        'filler_word_rate': analysis_result.filler_word_analysis.filler_word_rate,
                        'filler_word_types': analysis_result.filler_word_analysis.filler_word_types,
                        'filler_word_positions': analysis_result.filler_word_analysis.filler_word_positions,
                        'confidence_scores': analysis_result.filler_word_analysis.confidence_scores
                    },
                    'transcription_result': {
                        'original_transcript': analysis_result.transcription_result.original_transcript,
                        'enhanced_transcript': analysis_result.transcription_result.enhanced_transcript,
                        'confidence_score': analysis_result.transcription_result.confidence_score,
                        'word_timestamps': analysis_result.transcription_result.word_timestamps,
                        'quality_metrics': analysis_result.transcription_result.quality_metrics
                    },
                    'overall_metrics': {
                        'presentation_score': analysis_result.overall_presentation_score,
                        'confidence_level': analysis_result.confidence_level
                    },
                    'recommendations': analysis_result.recommendations,
                    'real_time_feedback': analysis_result.real_time_feedback,
                    'metadata': {
                        'processing_time': analysis_result.processing_time,
                        'analysis_timestamp': analysis_result.analysis_timestamp,
                        'model_versions': analysis_result.model_versions
                    }
                }
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@app.post("/real-time-feedback")
async def get_real_time_feedback(request: RealTimeFeedbackRequest):
    """Get real-time feedback during speech recording."""
    try:
        # Generate synthetic audio chunk for demonstration
        audio_chunk = np.random.random(1600)  # 0.1 second at 16kHz
        
        # Get real-time feedback
        feedback = web_speech_integration.get_real_time_feedback(
            request.current_transcript, 
            audio_chunk
        )
        
        return feedback
        
    except Exception as e:
        logger.error(f"Error generating real-time feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback generation failed: {str(e)}")

@app.get("/service/status")
async def get_service_status():
    """Get comprehensive service status including all components."""
    try:
        status = speech_analysis_service.get_service_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=f"Service status check failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    try:
        return {
            "speech_analysis_service": {
                "type": "Comprehensive Hybrid Analysis",
                "version": "1.0.0",
                "capabilities": [
                    "tensorflow_speech_quality_analysis",
                    "hybrid_filler_word_detection", 
                    "web_speech_api_integration",
                    "real_time_feedback",
                    "comprehensive_recommendations"
                ]
            },
            "speech_quality_analyzer": {
                "type": "TensorFlow Sequential Models",
                "components": ["clarity_model", "volume_model", "pace_model"],
                "input_features": ["MFCC", "RMS Energy", "Pace Features"],
                "output_metrics": ["clarity_score", "volume_consistency", "pace_score"]
            },
            "filler_word_detector": {
                "type": "Hybrid Rule-based + TensorFlow",
                "approaches": ["pattern_matching", "context_aware_ml"],
                "supported_fillers": list(filler_detector.filler_words),
                "confidence_scoring": True
            },
            "audio_processor": {
                "sample_rate": audio_processor.target_sr,
                "features": ["MFCC", "spectral", "prosodic"],
                "preprocessing": ["normalization", "noise_reduction", "pre_emphasis"]
            },
            "web_speech_integration": {
                "capabilities": ["transcription_enhancement", "real_time_feedback", "hybrid_analysis"],
                "supported_languages": ["en-US"],
                "real_time_features": True,
                "fallback_support": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

async def log_analysis_to_mlflow(analysis_data: Dict[str, Any]):
    """Log analysis results to MLflow for experiment tracking."""
    try:
        with mlflow.start_run():
            # Log metrics
            if 'speech_quality' in analysis_data:
                sq = analysis_data['speech_quality']
                mlflow.log_metric("clarity_score", sq.clarity_score)
                mlflow.log_metric("volume_variation_score", sq.volume_variation_score)
                mlflow.log_metric("pace_score", sq.pace_score)
                mlflow.log_metric("overall_quality", sq.overall_quality)
                mlflow.log_metric("speaking_rate_wpm", sq.speaking_rate_wpm)
            
            if 'filler_analysis' in analysis_data and analysis_data['filler_analysis']:
                fa = analysis_data['filler_analysis']
                mlflow.log_metric("total_filler_words", fa.total_filler_words)
                mlflow.log_metric("filler_word_rate", fa.filler_word_rate)
            
            # Log parameters
            mlflow.log_param("model_version", "1.0.0")
            mlflow.log_param("processing_time", analysis_data.get('processing_time', 0))
            
            if 'file_name' in analysis_data:
                mlflow.log_param("file_name", analysis_data['file_name'])
            
            if 'transcript_length' in analysis_data:
                mlflow.log_param("transcript_length", analysis_data['transcript_length'])
            
    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")

async def log_comprehensive_analysis_to_mlflow(analysis_result: ComprehensiveSpeechAnalysis):
    """Log comprehensive analysis results to MLflow for experiment tracking."""
    try:
        with mlflow.start_run():
            # Log speech quality metrics
            mlflow.log_metric("clarity_score", analysis_result.speech_quality.clarity_score)
            mlflow.log_metric("volume_variation_score", analysis_result.speech_quality.volume_variation_score)
            mlflow.log_metric("pace_score", analysis_result.speech_quality.pace_score)
            mlflow.log_metric("speech_overall_quality", analysis_result.speech_quality.overall_quality)
            mlflow.log_metric("speaking_rate_wpm", analysis_result.speech_quality.speaking_rate_wpm)
            mlflow.log_metric("volume_consistency", analysis_result.speech_quality.volume_consistency)
            mlflow.log_metric("pitch_variation", analysis_result.speech_quality.pitch_variation)
            
            # Log filler word metrics
            mlflow.log_metric("total_filler_words", analysis_result.filler_word_analysis.total_filler_words)
            mlflow.log_metric("filler_word_rate", analysis_result.filler_word_analysis.filler_word_rate)
            
            # Log transcription metrics
            mlflow.log_metric("transcription_confidence", analysis_result.transcription_result.confidence_score)
            
            # Log overall metrics
            mlflow.log_metric("overall_presentation_score", analysis_result.overall_presentation_score)
            mlflow.log_metric("confidence_level", analysis_result.confidence_level)
            
            # Log parameters
            mlflow.log_param("analysis_type", "comprehensive_hybrid")
            mlflow.log_param("processing_time", analysis_result.processing_time)
            mlflow.log_param("model_versions", str(analysis_result.model_versions))
            mlflow.log_param("num_recommendations", len(analysis_result.recommendations))
            
            # Log transcript length if available
            if analysis_result.transcription_result.enhanced_transcript:
                mlflow.log_param("transcript_length", len(analysis_result.transcription_result.enhanced_transcript))
            
            # Log filler word types
            if analysis_result.filler_word_analysis.filler_word_types:
                for filler_type, count in analysis_result.filler_word_analysis.filler_word_types.items():
                    mlflow.log_metric(f"filler_{filler_type}_count", count)
            
    except Exception as e:
        logger.error(f"Error logging comprehensive analysis to MLflow: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
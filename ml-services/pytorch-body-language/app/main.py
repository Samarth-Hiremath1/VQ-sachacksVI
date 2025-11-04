from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import io
import os
from PIL import Image
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel

from app.models.body_language_classifier import BodyLanguageAnalyzer
from app.services.pose_detector import MediaPipePoseDetector, PoseSequenceProcessor
from app.services.feature_extractor import PoseFeatureExtractor
from app.services.mlflow_tracker import MLflowTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PyTorch Body Language Analysis Service",
    description="Microservice for body language analysis using PyTorch and MediaPipe",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pose_detector = None
body_language_analyzer = None
feature_extractor = None
sequence_processor = None
mlflow_tracker = None


class PoseAnalysisRequest(BaseModel):
    """Request model for pose analysis"""
    landmarks: List[List[float]]  # 33 landmarks x 3 coordinates
    confidence_threshold: Optional[float] = 0.5


class SequenceAnalysisRequest(BaseModel):
    """Request model for sequence analysis"""
    landmarks_sequence: List[List[List[float]]]  # sequence_length x 33 landmarks x 3 coordinates
    confidence_threshold: Optional[float] = 0.5


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    posture_score: float
    posture_class: int
    posture_confidence: float
    gesture_name: Optional[str] = None
    gesture_confidence: Optional[float] = None
    movement_intensity: Optional[float] = None
    overall_body_language_score: float
    detailed_metrics: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global pose_detector, body_language_analyzer, feature_extractor, sequence_processor, mlflow_tracker
    
    logger.info("Initializing PyTorch Body Language Analysis Service...")
    
    try:
        # Initialize MediaPipe pose detector
        pose_detector = MediaPipePoseDetector(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize body language analyzer
        body_language_analyzer = BodyLanguageAnalyzer(device='cpu')
        
        # Initialize feature extractor
        feature_extractor = PoseFeatureExtractor()
        
        # Initialize sequence processor
        sequence_processor = PoseSequenceProcessor(sequence_length=30, overlap=0.5)
        
        # Initialize MLflow tracker
        mlflow_tracker = MLflowTracker(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name="body-language-inference"
        )
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pose_detector
    
    if pose_detector:
        pose_detector.close()
    
    logger.info("Services shut down successfully")


def get_pose_detector():
    """Dependency to get pose detector instance"""
    if pose_detector is None:
        raise HTTPException(status_code=500, detail="Pose detector not initialized")
    return pose_detector


def get_body_language_analyzer():
    """Dependency to get body language analyzer instance"""
    if body_language_analyzer is None:
        raise HTTPException(status_code=500, detail="Body language analyzer not initialized")
    return body_language_analyzer


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pytorch-body-language",
        "version": "1.0.0"
    }


@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    detector: MediaPipePoseDetector = Depends(get_pose_detector),
    analyzer: BodyLanguageAnalyzer = Depends(get_body_language_analyzer)
):
    """
    Analyze body language from a single image
    
    Args:
        file: Uploaded image file
        detector: MediaPipe pose detector
        analyzer: Body language analyzer
        
    Returns:
        Analysis results
    """
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect pose
        pose_result = detector.detect_pose(image_array)
        
        if pose_result is None:
            raise HTTPException(status_code=400, detail="No pose detected in image")
        
        # Analyze body language
        analysis_result = analyzer.analyze_pose_frame(pose_result['landmarks_array'])
        
        return AnalysisResponse(
            posture_score=analysis_result['posture_score'],
            posture_class=analysis_result['posture_class'],
            posture_confidence=analysis_result['posture_confidence'],
            overall_body_language_score=analysis_result['posture_score'],
            detailed_metrics=pose_result['pose_metrics']
        )
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/landmarks", response_model=AnalysisResponse)
async def analyze_landmarks(
    request: PoseAnalysisRequest,
    analyzer: BodyLanguageAnalyzer = Depends(get_body_language_analyzer)
):
    """
    Analyze body language from pose landmarks
    
    Args:
        request: Pose analysis request with landmarks
        analyzer: Body language analyzer
        
    Returns:
        Analysis results
    """
    import time
    start_time = time.time()
    
    try:
        # Convert landmarks to numpy array
        landmarks_array = np.array(request.landmarks, dtype=np.float32)
        
        if landmarks_array.shape != (33, 3):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected landmarks shape (33, 3), got {landmarks_array.shape}"
            )
        
        # Analyze body language
        analysis_result = analyzer.analyze_pose_frame(landmarks_array)
        
        # Log inference metrics to MLflow if available
        if mlflow_tracker:
            try:
                inference_time = time.time() - start_time
                mlflow_tracker.log_inference_metrics(
                    model_name="posture_classifier",
                    inference_time=inference_time,
                    batch_size=1,
                    input_shape=landmarks_array.shape,
                    confidence_scores=[analysis_result['posture_confidence']]
                )
            except Exception as mlflow_error:
                logger.warning(f"Failed to log MLflow metrics: {mlflow_error}")
        
        return AnalysisResponse(
            posture_score=analysis_result['posture_score'],
            posture_class=analysis_result['posture_class'],
            posture_confidence=analysis_result['posture_confidence'],
            overall_body_language_score=analysis_result['posture_score'],
            detailed_metrics={}
        )
        
    except Exception as e:
        logger.error(f"Error analyzing landmarks: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/sequence", response_model=AnalysisResponse)
async def analyze_sequence(
    request: SequenceAnalysisRequest,
    analyzer: BodyLanguageAnalyzer = Depends(get_body_language_analyzer)
):
    """
    Analyze body language from pose sequence for gesture recognition
    
    Args:
        request: Sequence analysis request
        analyzer: Body language analyzer
        
    Returns:
        Analysis results including gesture recognition
    """
    try:
        # Convert sequence to numpy array
        landmarks_sequence = np.array(request.landmarks_sequence, dtype=np.float32)
        
        if len(landmarks_sequence.shape) != 3 or landmarks_sequence.shape[1:] != (33, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Expected sequence shape (seq_len, 33, 3), got {landmarks_sequence.shape}"
            )
        
        # Use the last frame for posture analysis
        last_frame = landmarks_sequence[-1]
        
        # Get comprehensive analysis
        analysis_result = analyzer.get_comprehensive_analysis(
            pose_landmarks=last_frame,
            pose_sequence=landmarks_sequence
        )
        
        return AnalysisResponse(
            posture_score=analysis_result['posture_score'],
            posture_class=analysis_result['posture_class'],
            posture_confidence=analysis_result['posture_confidence'],
            gesture_name=analysis_result.get('gesture_name'),
            gesture_confidence=analysis_result.get('gesture_confidence'),
            movement_intensity=analysis_result.get('movement_intensity'),
            overall_body_language_score=analysis_result['overall_body_language_score'],
            detailed_metrics={}
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sequence: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/detect/pose")
async def detect_pose_in_image(
    file: UploadFile = File(...),
    detector: MediaPipePoseDetector = Depends(get_pose_detector)
):
    """
    Detect pose landmarks in an image
    
    Args:
        file: Uploaded image file
        detector: MediaPipe pose detector
        
    Returns:
        Pose detection results with landmarks
    """
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect pose
        pose_result = detector.detect_pose(image_array)
        
        if pose_result is None:
            raise HTTPException(status_code=400, detail="No pose detected in image")
        
        # Convert landmarks to serializable format
        landmarks_list = []
        for landmark in pose_result['landmarks']:
            landmarks_list.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return {
            "landmarks": landmarks_list,
            "pose_metrics": pose_result['pose_metrics'],
            "detection_confidence": pose_result['detection_confidence']
        }
        
    except Exception as e:
        logger.error(f"Error detecting pose: {e}")
        raise HTTPException(status_code=500, detail=f"Pose detection failed: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "posture_classifier": {
            "input_dim": 99,
            "hidden_dim": 128,
            "num_classes": 3,
            "classes": ["poor", "fair", "good"]
        },
        "gesture_recognizer": {
            "sequence_length": 30,
            "input_dim": 99,
            "hidden_dim": 128,
            "gesture_classes": ["neutral", "pointing", "open_palm", "crossed_arms", "hands_on_hips"]
        },
        "feature_extractor": {
            "basic_features": "geometric, angular, distance",
            "temporal_features": "velocity, acceleration, stability, rhythm"
        },
        "mlflow_tracking": {
            "enabled": mlflow_tracker is not None,
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            "experiment_name": "body-language-inference"
        }
    }


@app.get("/mlflow/experiments")
async def get_mlflow_experiments():
    """Get MLflow experiments"""
    if not mlflow_tracker:
        raise HTTPException(status_code=503, detail="MLflow tracking not available")
    
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": exp.creation_time
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching MLflow experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiments: {str(e)}")


@app.get("/mlflow/runs")
async def get_mlflow_runs(experiment_id: Optional[str] = None, limit: int = 10):
    """Get MLflow runs"""
    if not mlflow_tracker:
        raise HTTPException(status_code=503, detail="MLflow tracking not available")
    
    try:
        import mlflow
        if experiment_id:
            runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results=limit)
        else:
            runs = mlflow.search_runs(max_results=limit)
        
        return {
            "runs": runs.to_dict('records') if not runs.empty else []
        }
    except Exception as e:
        logger.error(f"Error fetching MLflow runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch runs: {str(e)}")


@app.post("/mlflow/log_custom_metric")
async def log_custom_metric(metric_name: str, metric_value: float, step: Optional[int] = None):
    """Log a custom metric to MLflow"""
    if not mlflow_tracker:
        raise HTTPException(status_code=503, detail="MLflow tracking not available")
    
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric(metric_name, metric_value, step=step)
            return {"status": "success", "message": f"Logged metric {metric_name}: {metric_value}"}
        else:
            return {"status": "warning", "message": "No active MLflow run"}
    except Exception as e:
        logger.error(f"Error logging custom metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log metric: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
"""
Presentation Analysis Pipeline DAG

This DAG orchestrates the end-to-end analysis workflow for presentation recordings:
1. Upload recording to S3 storage
2. Extract audio from video
3. Parallel analysis: Body language (PyTorch) and Speech (TensorFlow)
4. Aggregate results and calculate overall scores
5. Store results in database

Requirements: 2.1, 2.5, 3.5, 3.2, 4.4
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Configure logging
logger = logging.getLogger(__name__)


def task_failure_callback(context):
    """
    Callback function for task failures
    Logs detailed error information and updates recording status
    """
    from airflow.dags.utils.database_handler import DatabaseHandler
    
    task_instance = context['task_instance']
    exception = context.get('exception')
    recording_id = context['dag_run'].conf.get('recording_id')
    
    logger.error(
        f"Task {task_instance.task_id} failed for recording {recording_id}. "
        f"Exception: {exception}"
    )
    
    # Update recording status to failed
    try:
        db_handler = DatabaseHandler()
        db_handler.update_recording_status(recording_id, 'failed')
        logger.info(f"Updated recording {recording_id} status to 'failed'")
    except Exception as e:
        logger.error(f"Could not update recording status: {e}")


def dag_failure_callback(context):
    """
    Callback function for DAG failures
    Performs cleanup and final error reporting
    """
    recording_id = context['dag_run'].conf.get('recording_id')
    logger.error(f"DAG failed for recording {recording_id}")
    
    # Additional cleanup or notification logic can be added here


# Default arguments for the DAG
default_args = {
    'owner': 'coaching-platform',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=10),
    'on_failure_callback': task_failure_callback,
}

# DAG definition
dag = DAG(
    'presentation_analysis_pipeline',
    default_args=default_args,
    description='End-to-end presentation analysis workflow with PyTorch and TensorFlow',
    schedule_interval=None,  # Triggered by API
    catchup=False,
    max_active_runs=10,
    tags=['presentation', 'analysis', 'ml', 'pytorch', 'tensorflow'],
    on_failure_callback=dag_failure_callback,
)


def upload_recording_to_s3(**context) -> Dict[str, Any]:
    """
    Upload recording to S3 storage and validate file
    
    Returns:
        Dict with s3_key, video_path, file_size, duration
    """
    from airflow.dags.utils.storage_utils import S3StorageHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    file_path = context['dag_run'].conf.get('file_path')
    
    if not recording_id:
        raise ValueError("recording_id is required")
    if not file_path:
        raise ValueError("file_path is required")
    
    logger.info(f"Uploading recording {recording_id} to S3 from {file_path}")
    
    try:
        storage_handler = S3StorageHandler()
        result = storage_handler.upload_recording(recording_id, file_path)
        
        # Push to XCom for downstream tasks
        context['task_instance'].xcom_push(key='s3_key', value=result['s3_key'])
        context['task_instance'].xcom_push(key='video_path', value=result['video_path'])
        context['task_instance'].xcom_push(key='file_size', value=result['file_size'])
        context['task_instance'].xcom_push(key='duration', value=result['duration'])
        
        logger.info(f"Successfully uploaded to S3: {result['s3_key']}")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to upload recording to S3: {e}", exc_info=True)
        raise


def extract_audio_from_video(**context) -> Dict[str, Any]:
    """
    Extract audio track from video file
    
    Returns:
        Dict with audio_path, audio_s3_key, sample_rate
    """
    from airflow.dags.utils.video_processor import VideoProcessor
    from airflow.dags.utils.storage_utils import S3StorageHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    video_path = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='video_path')
    
    if not video_path:
        raise ValueError("video_path not found in XCom")
    
    logger.info(f"Extracting audio from video for recording {recording_id}")
    
    try:
        video_processor = VideoProcessor()
        audio_result = video_processor.extract_audio(video_path)
        
        logger.info(f"Audio extracted successfully: {audio_result['audio_path']}")
        
        # Upload audio to S3
        storage_handler = S3StorageHandler()
        audio_s3_result = storage_handler.upload_audio(recording_id, audio_result['audio_path'])
        
        result = {
            'audio_path': audio_result['audio_path'],
            'audio_s3_key': audio_s3_result['s3_key'],
            'sample_rate': audio_result['sample_rate'],
            'duration': audio_result['duration']
        }
        
        # Push to XCom
        context['task_instance'].xcom_push(key='audio_path', value=result['audio_path'])
        context['task_instance'].xcom_push(key='audio_s3_key', value=result['audio_s3_key'])
        context['task_instance'].xcom_push(key='sample_rate', value=result['sample_rate'])
        
        logger.info(f"Successfully extracted and uploaded audio: {result['audio_s3_key']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract audio: {e}", exc_info=True)
        raise


def analyze_body_language_pytorch(**context) -> Dict[str, Any]:
    """
    Analyze body language using PyTorch service with circuit breaker and fallback
    
    Returns:
        Dict with body language analysis results
    """
    from airflow.dags.utils.ml_service_client import PyTorchBodyLanguageClient
    from airflow.dags.utils.circuit_breaker import MLServiceCircuitBreakers, CircuitBreakerOpenError
    from airflow.dags.utils.fallback_handler import GracefulDegradationHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    video_path = context['task_instance'].xcom_pull(task_ids='upload_to_s3', key='video_path')
    
    logger.info(f"Analyzing body language for recording {recording_id}")
    
    # Initialize circuit breaker and fallback handler
    circuit_breakers = MLServiceCircuitBreakers()
    fallback_handler = GracefulDegradationHandler()
    
    try:
        # Try ML service with circuit breaker protection
        pytorch_client = PyTorchBodyLanguageClient()
        
        def call_pytorch():
            return pytorch_client.analyze_video(video_path)
        
        analysis_result = circuit_breakers.call_pytorch_service(call_pytorch)
        
        logger.info(f"Body language analysis complete (ML). Score: {analysis_result['overall_score']}")
        
    except (CircuitBreakerOpenError, Exception) as e:
        # Use fallback analysis
        logger.warning(f"PyTorch service unavailable, using fallback: {e}")
        analysis_result = fallback_handler.handle_body_language_failure(e, video_path)
        logger.info(f"Body language analysis complete (fallback). Score: {analysis_result['overall_score']}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='body_language_score', value=analysis_result['overall_score'])
    context['task_instance'].xcom_push(key='posture_score', value=analysis_result['posture_score'])
    context['task_instance'].xcom_push(key='gesture_score', value=analysis_result['gesture_score'])
    context['task_instance'].xcom_push(key='body_language_metrics', value=analysis_result['detailed_metrics'])
    context['task_instance'].xcom_push(key='body_language_method', value=analysis_result.get('method', 'ml'))
    
    return analysis_result


def analyze_speech_tensorflow(**context) -> Dict[str, Any]:
    """
    Analyze speech using TensorFlow service with circuit breaker and fallback
    
    Returns:
        Dict with speech analysis results
    """
    from airflow.dags.utils.ml_service_client import TensorFlowSpeechClient
    from airflow.dags.utils.circuit_breaker import MLServiceCircuitBreakers, CircuitBreakerOpenError
    from airflow.dags.utils.fallback_handler import GracefulDegradationHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    audio_path = context['task_instance'].xcom_pull(task_ids='extract_audio', key='audio_path')
    transcript = context['dag_run'].conf.get('transcript', '')
    
    logger.info(f"Analyzing speech for recording {recording_id}")
    
    # Initialize circuit breaker and fallback handler
    circuit_breakers = MLServiceCircuitBreakers()
    fallback_handler = GracefulDegradationHandler()
    
    try:
        # Try ML service with circuit breaker protection
        tensorflow_client = TensorFlowSpeechClient()
        
        def call_tensorflow():
            return tensorflow_client.analyze_audio(audio_path, transcript)
        
        analysis_result = circuit_breakers.call_tensorflow_service(call_tensorflow)
        
        logger.info(f"Speech analysis complete (ML). Quality: {analysis_result['overall_quality']}")
        
    except (CircuitBreakerOpenError, Exception) as e:
        # Use fallback analysis
        logger.warning(f"TensorFlow service unavailable, using fallback: {e}")
        analysis_result = fallback_handler.handle_speech_failure(e, audio_path, transcript)
        logger.info(f"Speech analysis complete (fallback). Quality: {analysis_result['overall_quality']}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='speech_quality_score', value=analysis_result['overall_quality'])
    context['task_instance'].xcom_push(key='speaking_pace_wpm', value=analysis_result['speaking_rate_wpm'])
    context['task_instance'].xcom_push(key='filler_word_count', value=analysis_result['filler_word_count'])
    context['task_instance'].xcom_push(key='speech_metrics', value=analysis_result['detailed_metrics'])
    context['task_instance'].xcom_push(key='speech_method', value=analysis_result.get('method', 'ml'))
    
    return analysis_result


def aggregate_analysis_results(**context) -> Dict[str, Any]:
    """
    Aggregate body language and speech analysis results
    Calculate overall presentation score with weighted metrics
    
    Returns:
        Dict with aggregated results and overall score
    """
    from airflow.dags.utils.result_aggregator import ResultAggregator
    
    recording_id = context['dag_run'].conf.get('recording_id')
    
    logger.info(f"Aggregating analysis results for recording {recording_id}")
    
    try:
        # Pull results from parallel tasks
        body_language_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='body_language_score'
        )
        posture_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='posture_score'
        )
        gesture_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='gesture_score'
        )
        body_language_metrics = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='body_language_metrics'
        )
        body_language_method = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='body_language_method'
        )
        
        speech_quality_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speech_quality_score'
        )
        speaking_pace_wpm = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speaking_pace_wpm'
        )
        filler_word_count = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='filler_word_count'
        )
        speech_metrics = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speech_metrics'
        )
        speech_method = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speech_method'
        )
        
        # Validate required data
        if body_language_score is None or speech_quality_score is None:
            raise ValueError("Missing required analysis scores")
        
        # Log if fallback methods were used
        if body_language_method and 'fallback' in body_language_method:
            logger.warning(f"Body language analysis used fallback method: {body_language_method}")
        if speech_method and 'fallback' in speech_method:
            logger.warning(f"Speech analysis used fallback method: {speech_method}")
        
        # Aggregate results
        aggregator = ResultAggregator()
        aggregated_result = aggregator.aggregate_results(
            body_language_score=body_language_score,
            posture_score=posture_score or 50.0,
            gesture_score=gesture_score or 50.0,
            body_language_metrics=body_language_metrics or {},
            speech_quality_score=speech_quality_score,
            speaking_pace_wpm=speaking_pace_wpm or 140.0,
            filler_word_count=filler_word_count or 0,
            speech_metrics=speech_metrics or {}
        )
        
        # Add method information to detailed metrics
        aggregated_result['detailed_metrics']['analysis_methods'] = {
            'body_language': body_language_method or 'ml',
            'speech': speech_method or 'ml'
        }
        
        # Push aggregated results to XCom
        context['task_instance'].xcom_push(key='overall_score', value=aggregated_result['overall_score'])
        context['task_instance'].xcom_push(key='recommendations', value=aggregated_result['recommendations'])
        context['task_instance'].xcom_push(key='detailed_metrics', value=aggregated_result['detailed_metrics'])
        
        logger.info(f"Results aggregated. Overall score: {aggregated_result['overall_score']}")
        return aggregated_result
        
    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}", exc_info=True)
        raise


def store_results_in_database(**context) -> Dict[str, Any]:
    """
    Store analysis results in PostgreSQL database
    
    Returns:
        Dict with database record ID
    """
    from airflow.dags.utils.database_handler import DatabaseHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    
    logger.info(f"Storing results in database for recording {recording_id}")
    
    try:
        # Pull all results from XCom
        overall_score = context['task_instance'].xcom_pull(
            task_ids='aggregate_results', key='overall_score'
        )
        recommendations = context['task_instance'].xcom_pull(
            task_ids='aggregate_results', key='recommendations'
        )
        detailed_metrics = context['task_instance'].xcom_pull(
            task_ids='aggregate_results', key='detailed_metrics'
        )
        
        body_language_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='body_language_score'
        )
        posture_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='posture_score'
        )
        gesture_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_body_language', key='gesture_score'
        )
        
        speech_quality_score = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speech_quality_score'
        )
        speaking_pace_wpm = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='speaking_pace_wpm'
        )
        filler_word_count = context['task_instance'].xcom_pull(
            task_ids='ml_analysis.analyze_speech', key='filler_word_count'
        )
        
        # Validate required data
        if overall_score is None:
            raise ValueError("overall_score is required")
        if body_language_score is None or speech_quality_score is None:
            raise ValueError("Missing required analysis scores")
        
        # Store in database
        db_handler = DatabaseHandler()
        result = db_handler.store_analysis_result(
            recording_id=recording_id,
            body_language_score=body_language_score,
            speech_quality_score=speech_quality_score,
            overall_score=overall_score,
            filler_word_count=filler_word_count or 0,
            speaking_pace_wpm=speaking_pace_wpm or 140.0,
            posture_score=posture_score or 50.0,
            gesture_score=gesture_score or 50.0,
            recommendations=recommendations or [],
            detailed_metrics=detailed_metrics or {}
        )
        
        logger.info(f"Results stored in database. Analysis ID: {result['analysis_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to store results in database: {e}", exc_info=True)
        raise


def update_recording_status(**context) -> None:
    """
    Update recording status to 'analyzed' in database
    """
    from airflow.dags.utils.database_handler import DatabaseHandler
    
    recording_id = context['dag_run'].conf.get('recording_id')
    
    logger.info(f"Updating recording status for {recording_id}")
    
    try:
        db_handler = DatabaseHandler()
        db_handler.update_recording_status(recording_id, 'analyzed')
        
        logger.info(f"Recording status updated to 'analyzed'")
        
    except Exception as e:
        logger.error(f"Failed to update recording status: {e}", exc_info=True)
        # Don't raise - this is a final cleanup step
        logger.warning("Continuing despite status update failure")


# Define tasks
upload_to_s3 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_recording_to_s3,
    provide_context=True,
    dag=dag,
)

extract_audio = PythonOperator(
    task_id='extract_audio',
    python_callable=extract_audio_from_video,
    provide_context=True,
    dag=dag,
)

# Parallel ML analysis tasks
with TaskGroup('ml_analysis', tooltip='Parallel ML analysis tasks', dag=dag) as ml_analysis_group:
    analyze_body_language = PythonOperator(
        task_id='analyze_body_language',
        python_callable=analyze_body_language_pytorch,
        provide_context=True,
    )
    
    analyze_speech = PythonOperator(
        task_id='analyze_speech',
        python_callable=analyze_speech_tensorflow,
        provide_context=True,
    )

aggregate_results = PythonOperator(
    task_id='aggregate_results',
    python_callable=aggregate_analysis_results,
    provide_context=True,
    dag=dag,
)

store_results = PythonOperator(
    task_id='store_results',
    python_callable=store_results_in_database,
    provide_context=True,
    dag=dag,
)

update_status = PythonOperator(
    task_id='update_status',
    python_callable=update_recording_status,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
# Upload -> Extract audio
upload_to_s3 >> extract_audio

# Upload -> Body language analysis (parallel)
upload_to_s3 >> ml_analysis_group

# Extract audio -> Speech analysis (parallel)
extract_audio >> ml_analysis_group

# Both ML analyses -> Aggregate results
ml_analysis_group >> aggregate_results

# Aggregate -> Store -> Update status
aggregate_results >> store_results >> update_status

# Airflow DAG - Presentation Analysis Pipeline

## Overview

This directory contains the Airflow DAG implementation for orchestrating the end-to-end presentation analysis workflow. The pipeline processes video recordings through PyTorch body language analysis and TensorFlow speech analysis services, with comprehensive error handling and graceful degradation.

## Architecture

### Main DAG: `presentation_analysis_pipeline.py`

The main DAG orchestrates the following workflow:

1. **Upload to S3**: Upload recording to MinIO S3-compatible storage
2. **Extract Audio**: Extract audio track from video file using FFmpeg
3. **Parallel ML Analysis**:
   - **Body Language Analysis**: PyTorch-based pose and gesture analysis
   - **Speech Analysis**: TensorFlow-based speech quality and filler word detection
4. **Aggregate Results**: Combine analysis results with weighted scoring
5. **Store Results**: Save analysis results to PostgreSQL database
6. **Update Status**: Mark recording as analyzed

### Utility Modules

#### `utils/ml_service_client.py`
- **PyTorchBodyLanguageClient**: Client for PyTorch body language service
- **TensorFlowSpeechClient**: Client for TensorFlow speech analysis service
- Handles HTTP communication with ML microservices
- Implements timeout and error handling

#### `utils/result_aggregator.py`
- **ResultAggregator**: Aggregates ML analysis results
- Calculates weighted overall presentation score:
  - Body Language: 40%
  - Speech Quality: 35%
  - Content Delivery: 25%
- Generates personalized recommendations based on scores
- Provides detailed metrics for frontend visualization

#### `utils/circuit_breaker.py`
- **CircuitBreaker**: Implements circuit breaker pattern
- **MLServiceCircuitBreakers**: Manages circuit breakers for ML services
- Protects against cascading failures
- States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- Configurable failure threshold and recovery timeout

#### `utils/fallback_handler.py`
- **FallbackAnalyzer**: Provides rule-based fallback analysis
- **GracefulDegradationHandler**: Manages fallback when ML services fail
- Uses MediaPipe and Web Speech API heuristics
- Returns conservative scores with warnings

#### `utils/storage_utils.py`
- **S3StorageHandler**: Handles MinIO S3 operations
- Upload/download recordings and audio files
- Generate presigned URLs for file access
- Automatic bucket management

#### `utils/video_processor.py`
- **VideoProcessor**: Handles video processing operations
- Extract audio from video using FFmpeg
- Get video metadata (duration, resolution, codec)
- Temporary file management

#### `utils/database_handler.py`
- **DatabaseHandler**: Handles database operations
- Store analysis results in PostgreSQL
- Update recording status
- Health check functionality

## Error Handling & Resilience

### Circuit Breaker Pattern

The pipeline implements circuit breakers for ML services to prevent cascading failures:

```python
# Circuit breaker states
CLOSED -> Normal operation
OPEN -> Service failing, reject requests
HALF_OPEN -> Testing recovery
```

**Configuration**:
- Failure threshold: 3 failures
- Recovery timeout: 60 seconds
- Automatic state transitions

### Graceful Degradation

When ML services fail, the pipeline automatically falls back to rule-based analysis:

1. **Body Language Fallback**:
   - Uses MediaPipe pose detection heuristics
   - Conservative scoring (65/100)
   - Provides basic posture and gesture analysis

2. **Speech Analysis Fallback**:
   - Rule-based filler word detection
   - Speaking pace calculation from transcript
   - Conservative quality scores

### Retry Logic

- **Retries**: 3 attempts per task
- **Retry delay**: 2 minutes (exponential backoff)
- **Max retry delay**: 10 minutes
- **Timeout**: 5 minutes per ML inference

### Error Callbacks

- **Task failure callback**: Logs errors and updates recording status to 'failed'
- **DAG failure callback**: Performs cleanup and final error reporting

## Configuration

### Environment Variables

```bash
# ML Service URLs
PYTORCH_SERVICE_URL=http://pytorch-service:8000
TENSORFLOW_SERVICE_URL=http://tensorflow-service:8000

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/coaching_platform

# S3/MinIO
S3_ENDPOINT=minio:9000
S3_ACCESS_KEY=access_key
S3_SECRET_KEY=secret_key
S3_BUCKET_NAME=coaching-platform
S3_SECURE=false
```

### DAG Configuration

```python
default_args = {
    'owner': 'coaching-platform',
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=10),
    'on_failure_callback': task_failure_callback,
}
```

## Triggering the DAG

The DAG is triggered via API with the following configuration:

```python
dag_run_conf = {
    'recording_id': 'uuid-string',
    'file_path': '/path/to/video.mp4',
    'transcript': 'optional transcript text'
}
```

Example using Airflow REST API:

```bash
curl -X POST \
  http://localhost:8080/api/v1/dags/presentation_analysis_pipeline/dagRuns \
  -H 'Content-Type: application/json' \
  -d '{
    "conf": {
      "recording_id": "123e4567-e89b-12d3-a456-426614174000",
      "file_path": "/tmp/recording.mp4",
      "transcript": "Hello everyone..."
    }
  }'
```

## Monitoring

### Task Status

Monitor task execution in Airflow UI:
- http://localhost:8080/dags/presentation_analysis_pipeline

### Logs

Each task logs detailed information:
- Upload progress and S3 keys
- Audio extraction status
- ML service responses or fallback usage
- Aggregation calculations
- Database storage confirmation

### Circuit Breaker Stats

Check circuit breaker status:

```python
from airflow.dags.utils.circuit_breaker import MLServiceCircuitBreakers

breakers = MLServiceCircuitBreakers()
stats = breakers.get_all_stats()
# Returns: {'pytorch': {...}, 'tensorflow': {...}}
```

## Requirements

### System Dependencies
- FFmpeg (for audio extraction)
- FFprobe (for video metadata)

### Python Dependencies
- apache-airflow
- requests
- minio
- sqlalchemy
- psycopg2-binary

## Testing

### Manual Testing

1. Place a test video file in accessible location
2. Trigger DAG with test configuration
3. Monitor execution in Airflow UI
4. Verify results in database

### Fallback Testing

To test fallback behavior:
1. Stop ML services
2. Trigger DAG
3. Verify fallback analysis is used
4. Check warnings in results

## Troubleshooting

### Common Issues

**Issue**: Audio extraction fails
- **Solution**: Ensure FFmpeg is installed and video file is valid

**Issue**: ML service timeout
- **Solution**: Check service health, increase timeout if needed

**Issue**: Circuit breaker stuck OPEN
- **Solution**: Manually reset: `breakers.reset_all()`

**Issue**: Database connection fails
- **Solution**: Verify DATABASE_URL and PostgreSQL is running

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('airflow.dags').setLevel(logging.DEBUG)
```

## Performance

### Expected Timings

- Upload to S3: 5-30 seconds (depends on file size)
- Audio extraction: 10-60 seconds
- Body language analysis: 30-120 seconds
- Speech analysis: 20-90 seconds
- Aggregation: < 1 second
- Database storage: < 1 second

**Total**: 1-5 minutes per recording

### Optimization

- Parallel ML analysis reduces total time by ~50%
- Circuit breakers prevent wasted time on failing services
- Fallback analysis completes in < 5 seconds

## Future Enhancements

1. Add eye contact analysis
2. Implement content quality scoring
3. Add real-time progress updates via WebSocket
4. Implement distributed task execution
5. Add A/B testing for model versions
6. Implement data drift detection

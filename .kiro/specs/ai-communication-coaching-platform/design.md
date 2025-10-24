# Design Document

## Overview

The AI Communication Coaching Platform is designed as a microservices architecture leveraging containerized services to provide real-time presentation analysis and coaching. The system combines a React/TypeScript frontend with a FastAPI backend, utilizing a hybrid ML approach with PyTorch for body language analysis and TensorFlow for speech processing. The platform simulates cloud-native patterns using Docker containers, local S3-compatible storage, and comprehensive monitoring.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[React/TypeScript/Vite App]
        B[MediaPipe Pose Detection]
        C[Web Speech API]
    end
    
    subgraph "API Gateway"
        D[FastAPI Backend]
        E[Authentication Middleware]
        F[Rate Limiting]
    end
    
    subgraph "Processing Layer"
        G[Airflow DAGs]
        H[PyTorch Body Language Service]
        I[TensorFlow Speech Service]
        J[Analysis Orchestrator]
    end
    
    subgraph "Data Layer"
        K[Simulated S3 Storage]
        L[PostgreSQL Database]
        M[Redis Cache]
    end
    
    subgraph "ML Operations"
        N[MLflow Tracking Server]
        O[Model Registry]
        P[Experiment Tracking]
    end
    
    subgraph "Monitoring"
        Q[Prometheus Metrics]
        R[Grafana Dashboards]
        S[Container Health Checks]
    end
    
    A --> D
    B --> D
    C --> D
    D --> G
    G --> H
    G --> I
    H --> J
    I --> J
    J --> K
    J --> L
    D --> M
    H --> N
    I --> N
    Q --> R
    D --> Q
    H --> Q
    I --> Q
```

### Technology Stack

**Frontend:**
- React 18 with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- MediaPipe for real-time pose detection
- Web Speech API for speech recognition
- Recharts for analytics visualization

**Backend:**
- FastAPI with Python 3.11
- Pydantic for data validation
- SQLAlchemy for database ORM
- Alembic for database migrations
- Redis for caching and session management

**Machine Learning:**
- PyTorch 2.0+ for body language classification models
- TensorFlow 2.13+ for speech analysis models
- MLflow for experiment tracking and model registry
- Scikit-learn for preprocessing and evaluation metrics

**Infrastructure:**
- Docker and Docker Compose for containerization
- MinIO for S3-compatible object storage simulation
- PostgreSQL for relational data storage
- Apache Airflow for workflow orchestration
- Prometheus for metrics collection
- Grafana for monitoring dashboards

## Components and Interfaces

### Frontend Components

#### Core Components
- **RecordingInterface**: Manages webcam, microphone, and MediaPipe integration
- **AnalyticsDashboard**: Displays comprehensive presentation analysis
- **CoachingRecommendations**: Shows personalized improvement suggestions
- **ProgressTracking**: Visualizes user improvement over time

#### MediaPipe Integration
```typescript
interface PoseAnalysis {
  landmarks: PoseLandmark[];
  confidence: number;
  timestamp: number;
  bodyLanguageMetrics: {
    posture: PostureScore;
    gestures: GestureAnalysis[];
    movement: MovementPattern;
  };
}
```

#### Speech Recognition Integration
```typescript
interface SpeechAnalysis {
  transcript: string;
  confidence: number;
  fillerWords: FillerWordCount[];
  speakingPace: number; // words per minute
  volumeVariation: number[];
  pausePatterns: PauseAnalysis[];
}
```

### Backend API Interfaces

#### FastAPI Endpoints
```python
# Core Analysis Endpoints
POST /api/v1/recordings/upload
POST /api/v1/recordings/{id}/analyze
GET /api/v1/recordings/{id}/results
GET /api/v1/users/{id}/analytics

# ML Model Endpoints
POST /api/v1/models/body-language/predict
POST /api/v1/models/speech/predict
GET /api/v1/models/health

# User Management
POST /api/v1/auth/login
POST /api/v1/auth/register
GET /api/v1/users/profile
```

#### Data Models
```python
class PresentationRecording(BaseModel):
    id: UUID
    user_id: UUID
    video_url: str
    audio_url: str
    metadata: Dict[str, Any]
    created_at: datetime
    analysis_status: AnalysisStatus

class AnalysisResult(BaseModel):
    recording_id: UUID
    body_language_score: float
    speech_quality_score: float
    overall_score: float
    recommendations: List[Recommendation]
    detailed_metrics: DetailedMetrics
    processed_at: datetime
```

### Machine Learning Services

#### PyTorch Body Language Service
```python
class BodyLanguageClassifier:
    def __init__(self):
        self.model = self.load_pytorch_model()
        self.pose_processor = MediaPipeProcessor()
    
    def analyze_pose_sequence(self, pose_landmarks: List[PoseLandmark]) -> BodyLanguageAnalysis:
        # Process pose landmarks through PyTorch model
        # Return classification results for posture, gestures, confidence
        pass
    
    def extract_features(self, landmarks: List[PoseLandmark]) -> torch.Tensor:
        # Feature extraction for body language analysis
        pass
```

#### TensorFlow Speech Service
```python
class SpeechQualityAnalyzer:
    def __init__(self):
        self.model = self.load_tensorflow_model()
        self.audio_processor = AudioProcessor()
    
    def analyze_speech_quality(self, audio_data: np.ndarray) -> SpeechAnalysis:
        # Process audio through TensorFlow model
        # Return speech quality metrics, pace analysis, vocal patterns
        pass
    
    def extract_audio_features(self, audio: np.ndarray) -> tf.Tensor:
        # Feature extraction for speech analysis
        pass
```

### Airflow DAG Architecture

#### Processing Workflow
```python
# Main processing DAG
dag = DAG(
    'presentation_analysis_pipeline',
    default_args=default_args,
    schedule_interval=None,  # Triggered by API
    catchup=False
)

# Task definitions
upload_to_s3 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_recording_to_s3
)

extract_audio = PythonOperator(
    task_id='extract_audio',
    python_callable=extract_audio_from_video
)

analyze_body_language = PythonOperator(
    task_id='analyze_body_language',
    python_callable=run_pytorch_analysis
)

analyze_speech = PythonOperator(
    task_id='analyze_speech',
    python_callable=run_tensorflow_analysis
)

combine_results = PythonOperator(
    task_id='combine_results',
    python_callable=aggregate_analysis_results
)

# Task dependencies
upload_to_s3 >> extract_audio
upload_to_s3 >> analyze_body_language
extract_audio >> analyze_speech
[analyze_body_language, analyze_speech] >> combine_results
```

## Data Models

### Database Schema

#### Users Table
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Recordings Table
```sql
CREATE TABLE recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title VARCHAR(255),
    video_s3_key VARCHAR(500),
    audio_s3_key VARCHAR(500),
    duration_seconds INTEGER,
    file_size_bytes BIGINT,
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Analysis Results Table
```sql
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recording_id UUID REFERENCES recordings(id),
    body_language_score DECIMAL(5,2),
    speech_quality_score DECIMAL(5,2),
    overall_score DECIMAL(5,2),
    filler_word_count INTEGER,
    speaking_pace_wpm DECIMAL(6,2),
    posture_score DECIMAL(5,2),
    gesture_score DECIMAL(5,2),
    eye_contact_score DECIMAL(5,2),
    recommendations JSONB,
    detailed_metrics JSONB,
    processed_at TIMESTAMP DEFAULT NOW()
);
```

### MLflow Model Registry Schema

#### Model Metadata
```python
# PyTorch Body Language Model
{
    "name": "body-language-classifier-v1",
    "version": "1.0.0",
    "framework": "pytorch",
    "input_schema": {
        "pose_landmarks": "List[List[float]]",  # 33 landmarks x 4 coordinates
        "sequence_length": "int"
    },
    "output_schema": {
        "posture_score": "float",
        "gesture_classification": "str",
        "confidence": "float"
    },
    "metrics": {
        "accuracy": 0.89,
        "f1_score": 0.87,
        "precision": 0.91
    }
}

# TensorFlow Speech Model
{
    "name": "speech-quality-analyzer-v1",
    "version": "1.0.0",
    "framework": "tensorflow",
    "input_schema": {
        "audio_features": "np.ndarray",  # MFCC features
        "sample_rate": "int"
    },
    "output_schema": {
        "quality_score": "float",
        "pace_analysis": "dict",
        "vocal_patterns": "dict"
    },
    "metrics": {
        "mse": 0.12,
        "mae": 0.08,
        "r2_score": 0.84
    }
}
```

## Error Handling

### API Error Responses
```python
class APIError(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

# Common error scenarios
UPLOAD_TOO_LARGE = APIError(
    error_code="UPLOAD_TOO_LARGE",
    message="File size exceeds maximum limit of 500MB"
)

MODEL_INFERENCE_FAILED = APIError(
    error_code="MODEL_INFERENCE_FAILED",
    message="ML model inference failed, falling back to rule-based analysis"
)

PROCESSING_TIMEOUT = APIError(
    error_code="PROCESSING_TIMEOUT",
    message="Analysis processing exceeded timeout limit"
)
```

### Graceful Degradation Strategy
1. **ML Model Failures**: Fall back to MediaPipe + Web Speech API analysis
2. **Storage Failures**: Use local temporary storage with cleanup
3. **Processing Overload**: Queue requests and provide estimated wait times
4. **Network Issues**: Implement retry logic with exponential backoff

### Circuit Breaker Pattern
```python
class MLServiceCircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call_service(self, service_func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")
        
        try:
            result = service_func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e
```

## Testing Strategy

### Frontend Testing
- **Unit Tests**: Jest + React Testing Library for component testing
- **Integration Tests**: Cypress for end-to-end user workflows
- **Performance Tests**: Lighthouse CI for performance regression testing
- **Accessibility Tests**: axe-core for WCAG compliance

### Backend Testing
- **Unit Tests**: pytest for FastAPI endpoints and business logic
- **Integration Tests**: TestContainers for database and external service testing
- **Load Tests**: Locust for API performance under load
- **Contract Tests**: Pact for API contract validation

### ML Model Testing
- **Model Validation**: Cross-validation and holdout test sets
- **Performance Benchmarks**: Inference time and accuracy metrics
- **Data Drift Detection**: Statistical tests for input data distribution changes
- **A/B Testing**: MLflow integration for model comparison

### Infrastructure Testing
- **Container Tests**: Docker health checks and startup validation
- **Monitoring Tests**: Prometheus metrics collection validation
- **Disaster Recovery**: Backup and restore procedure testing

### Testing Environments
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-db:
    image: postgres:15
    environment:
      POSTGRES_DB: test_coaching_platform
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
  
  test-redis:
    image: redis:7-alpine
  
  test-minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: test_access_key
      MINIO_ROOT_PASSWORD: test_secret_key
  
  test-api:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://test_user:test_password@test-db:5432/test_coaching_platform
      REDIS_URL: redis://test-redis:6379
      S3_ENDPOINT: http://test-minio:9000
    depends_on:
      - test-db
      - test-redis
      - test-minio
```

## Deployment Architecture

### Docker Compose Services
```yaml
version: '3.8'
services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      VITE_API_URL: http://localhost:8000
  
  # Backend API
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/coaching_platform
      REDIS_URL: redis://redis:6379
      S3_ENDPOINT: http://minio:9000
    depends_on:
      - postgres
      - redis
      - minio
  
  # ML Services
  pytorch-service:
    build: ./ml-services/pytorch-body-language
    ports:
      - "8001:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
  
  tensorflow-service:
    build: ./ml-services/tensorflow-speech
    ports:
      - "8002:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
  
  # Data Services
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: coaching_platform
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
  
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: access_key
      MINIO_ROOT_PASSWORD: secret_key
    volumes:
      - minio_data:/data
  
  # Workflow Orchestration
  airflow-webserver:
    image: apache/airflow:2.7.0
    command: webserver
    ports:
      - "8080:8080"
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
    depends_on:
      - postgres
  
  airflow-scheduler:
    image: apache/airflow:2.7.0
    command: scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
    depends_on:
      - postgres
  
  # ML Operations
  mlflow:
    image: python:3.11-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
               --default-artifact-root s3://mlflow-artifacts/
               --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    environment:
      AWS_ACCESS_KEY_ID: access_key
      AWS_SECRET_ACCESS_KEY: secret_key
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    depends_on:
      - postgres
      - minio
  
  # Monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
  
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

This design provides a comprehensive, scalable architecture that meets all the requirements while using the specified technology stack. The system is designed to be production-ready with proper error handling, monitoring, and testing strategies.
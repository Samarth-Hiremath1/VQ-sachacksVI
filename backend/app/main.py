from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import logging
from .core.config import settings
from .core.database import check_database_health
from .core.redis import check_redis_health
from .core.minio_client import get_minio_client
from .api.v1.router import api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Communication Coaching Platform",
    description="Backend API for AI-driven presentation coaching platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security (disabled for testing)
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
# )


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred"
        }
    )


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status"""
    minio_client = get_minio_client()
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": check_database_health(),
            "redis": check_redis_health(),
            "minio": minio_client.health_check()
        }
    }
    
    # Check if any service is unhealthy
    if not all(health_status["services"].values()):
        health_status["status"] = "unhealthy"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return health_status


# Include API routers
app.include_router(api_router, prefix=settings.api_v1_prefix)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Communication Coaching Platform API")
    logger.info(f"API documentation available at: /docs")
    
    # Start background task scheduler
    try:
        from .services.background_tasks import get_background_service
        background_service = get_background_service()
        await background_service.start_scheduler()
        logger.info("Background task scheduler started")
    except Exception as e:
        logger.error(f"Failed to start background task scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Communication Coaching Platform API")
    
    # Stop background task scheduler
    try:
        from .services.background_tasks import get_background_service
        background_service = get_background_service()
        await background_service.stop_scheduler()
        logger.info("Background task scheduler stopped")
    except Exception as e:
        logger.error(f"Failed to stop background task scheduler: {e}")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Communication Coaching Platform API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }
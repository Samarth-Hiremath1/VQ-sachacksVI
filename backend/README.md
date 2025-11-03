# AI Communication Coaching Platform - Backend

## Overview

FastAPI backend for the AI Communication Coaching Platform with comprehensive authentication, database models, and API structure.

## Features Implemented

### ✅ Task 2: Initialize FastAPI backend with core structure
- FastAPI application with proper project structure
- Database connection with SQLAlchemy and Alembic migrations
- Redis connection for caching and session management
- CORS, authentication middleware, and request validation

### ✅ Task 2.1: Create database models and migrations
- SQLAlchemy models for users, recordings, and analysis_results tables
- Alembic migration scripts for database schema
- Database connection pooling and health checks

### ✅ Task 2.2: Implement user authentication and authorization
- JWT-based authentication system with FastAPI security
- User registration, login, and profile management endpoints
- Role-based access control middleware

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   └── v1/
│   │       ├── auth.py         # Authentication endpoints
│   │       ├── users.py        # User management endpoints
│   │       └── router.py       # API router configuration
│   ├── core/
│   │   ├── config.py           # Application configuration
│   │   ├── database.py         # Database connection and setup
│   │   ├── redis.py            # Redis connection
│   │   ├── security.py         # JWT and password utilities
│   │   └── dependencies.py     # FastAPI dependencies
│   ├── models/
│   │   ├── user.py             # User model
│   │   ├── recording.py        # Recording model
│   │   └── analysis_result.py  # Analysis result model
│   ├── schemas/
│   │   ├── user.py             # User Pydantic schemas
│   │   └── token.py            # Token schemas
│   └── services/
│       └── user_service.py     # User business logic
├── alembic/                    # Database migrations
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── .env.example               # Environment variables template
└── test_*.py                  # Test files
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login

### Users
- `GET /api/v1/users/profile` - Get user profile
- `PUT /api/v1/users/profile` - Update user profile
- `DELETE /api/v1/users/profile` - Deactivate user account

### Health & Documentation
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health check with dependencies
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation

## Database Models

### User
- UUID primary key
- Email (unique)
- Password hash
- First/last name
- Active status and superuser flag
- Created/updated timestamps

### Recording
- UUID primary key
- User foreign key
- Title, S3 keys for video/audio
- Duration, file size, status
- Created/updated timestamps

### AnalysisResult
- UUID primary key
- Recording foreign key
- Various analysis scores (body language, speech quality, etc.)
- Recommendations and detailed metrics (JSONB)
- Processing timestamp

## Configuration

Environment variables (see `.env.example`):
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - JWT secret key
- `ALLOWED_ORIGINS` - CORS allowed origins

## Testing

Run tests:
```bash
python test_basic.py    # Basic structure tests
python test_api.py      # API endpoint tests
python test_auth.py     # Authentication tests
```

## Next Steps

The backend is ready for:
1. File upload handling (Task 3)
2. ML service integration (Tasks 4-5)
3. Airflow pipeline integration (Task 6)
4. Analytics and recommendations (Task 7)

All core infrastructure is in place with proper error handling, security, and monitoring foundations.
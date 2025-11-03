from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine with connection pooling (lazy initialization)
engine = None

def get_engine():
    global engine
    if engine is None:
        # StaticPool doesn't support pool_size and max_overflow parameters
        engine = create_engine(
            settings.database_url,
            poolclass=StaticPool,
            pool_pre_ping=True,
            echo=False
        )
    return engine

# Create SessionLocal class
def get_session_local():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

# Create Base class for models
Base = declarative_base()


# Database dependency
def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# Health check function
def check_database_health():
    """Check if database connection is healthy"""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Event listener for connection pool events
def setup_engine_events():
    """Set up engine event listeners"""
    engine = get_engine()
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set database connection parameters if needed"""
        pass
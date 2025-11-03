"""
Background task service for automated cleanup and maintenance
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.minio_client import get_minio_client
from .lifecycle_service import LifecycleManagementService
from ..core.config import settings

logger = logging.getLogger(__name__)


class BackgroundTaskService:
    """Service for managing background tasks and scheduled operations"""
    
    def __init__(self):
        self.running = False
        self.tasks = {}
    
    async def start_scheduler(self):
        """Start the background task scheduler"""
        self.running = True
        logger.info("Starting background task scheduler")
        
        # Schedule periodic cleanup tasks
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._periodic_health_check())
    
    async def stop_scheduler(self):
        """Stop the background task scheduler"""
        self.running = False
        logger.info("Stopping background task scheduler")
    
    async def _periodic_cleanup(self):
        """Run periodic cleanup tasks"""
        while self.running:
            try:
                # Run cleanup every 6 hours
                await asyncio.sleep(6 * 3600)  # 6 hours
                
                if not self.running:
                    break
                
                logger.info("Running periodic cleanup tasks")
                await self._run_cleanup_task()
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _periodic_health_check(self):
        """Run periodic health checks"""
        while self.running:
            try:
                # Run health check every hour
                await asyncio.sleep(3600)  # 1 hour
                
                if not self.running:
                    break
                
                logger.info("Running periodic health check")
                await self._run_health_check()
                
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _run_cleanup_task(self):
        """Execute cleanup task"""
        try:
            # Get database session
            db = next(get_db())
            minio_client = get_minio_client()
            
            lifecycle_service = LifecycleManagementService(db, minio_client)
            
            # Run cleanup policies
            results = lifecycle_service.apply_storage_policies()
            
            logger.info(f"Cleanup completed: {results['total_files_cleaned']} files cleaned, "
                       f"{results['total_storage_freed']} bytes freed")
            
            # Store task result
            self.tasks['last_cleanup'] = {
                'timestamp': datetime.utcnow(),
                'results': results,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
            self.tasks['last_cleanup'] = {
                'timestamp': datetime.utcnow(),
                'error': str(e),
                'status': 'failed'
            }
        finally:
            if 'db' in locals():
                db.close()
    
    async def _run_health_check(self):
        """Execute health check task"""
        try:
            # Get database session
            db = next(get_db())
            minio_client = get_minio_client()
            
            lifecycle_service = LifecycleManagementService(db, minio_client)
            
            # Run storage integrity check
            integrity_results = lifecycle_service.validate_storage_integrity()
            
            # Get storage usage stats
            usage_stats = lifecycle_service.get_storage_usage_stats()
            
            logger.info(f"Health check completed: integrity score {integrity_results['integrity_score']:.2f}, "
                       f"total storage {usage_stats['total_size_bytes']} bytes")
            
            # Store task result
            self.tasks['last_health_check'] = {
                'timestamp': datetime.utcnow(),
                'integrity': integrity_results,
                'usage': usage_stats,
                'status': 'completed'
            }
            
            # Alert if integrity is low
            if integrity_results['integrity_score'] < 0.95:
                logger.warning(f"Storage integrity is low: {integrity_results['integrity_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Health check task failed: {e}")
            self.tasks['last_health_check'] = {
                'timestamp': datetime.utcnow(),
                'error': str(e),
                'status': 'failed'
            }
        finally:
            if 'db' in locals():
                db.close()
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of background tasks"""
        return {
            'scheduler_running': self.running,
            'tasks': self.tasks,
            'next_cleanup': self._get_next_task_time('last_cleanup', 6 * 3600),
            'next_health_check': self._get_next_task_time('last_health_check', 3600)
        }
    
    def _get_next_task_time(self, task_name: str, interval_seconds: int) -> str:
        """Calculate next task execution time"""
        if task_name in self.tasks:
            last_run = self.tasks[task_name]['timestamp']
            next_run = last_run + timedelta(seconds=interval_seconds)
            return next_run.isoformat()
        else:
            return "Not scheduled"


# Global background task service instance
background_service = BackgroundTaskService()


def get_background_service() -> BackgroundTaskService:
    """Get background task service instance"""
    return background_service
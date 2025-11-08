"""
Database handler for Airflow DAG operations

Handles database operations for storing analysis results.
Requirements: 3.7, 7.4
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from uuid import UUID
import json

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handles database operations for analysis results"""
    
    def __init__(self):
        self.database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://user:password@postgres:5432/coaching_platform'
        )
        
        # Create engine and session
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("Initialized DatabaseHandler")
    
    def store_analysis_result(
        self,
        recording_id: str,
        body_language_score: float,
        speech_quality_score: float,
        overall_score: float,
        filler_word_count: int,
        speaking_pace_wpm: float,
        posture_score: float,
        gesture_score: float,
        recommendations: list,
        detailed_metrics: dict
    ) -> Dict[str, Any]:
        """
        Store analysis results in database
        
        Args:
            recording_id: Recording UUID
            body_language_score: Body language score (0-100)
            speech_quality_score: Speech quality score (0-100)
            overall_score: Overall presentation score (0-100)
            filler_word_count: Number of filler words detected
            speaking_pace_wpm: Speaking pace in words per minute
            posture_score: Posture score (0-100)
            gesture_score: Gesture score (0-100)
            recommendations: List of recommendation objects
            detailed_metrics: Detailed metrics dictionary
            
        Returns:
            Dict with analysis result ID
        """
        session = self.SessionLocal()
        
        try:
            logger.info(f"Storing analysis results for recording {recording_id}")
            
            # Insert analysis result
            query = text("""
                INSERT INTO analysis_results (
                    recording_id,
                    body_language_score,
                    speech_quality_score,
                    overall_score,
                    filler_word_count,
                    speaking_pace_wpm,
                    posture_score,
                    gesture_score,
                    eye_contact_score,
                    recommendations,
                    detailed_metrics,
                    processed_at
                ) VALUES (
                    :recording_id,
                    :body_language_score,
                    :speech_quality_score,
                    :overall_score,
                    :filler_word_count,
                    :speaking_pace_wpm,
                    :posture_score,
                    :gesture_score,
                    :eye_contact_score,
                    :recommendations,
                    :detailed_metrics,
                    :processed_at
                )
                RETURNING id
            """)
            
            result = session.execute(query, {
                'recording_id': recording_id,
                'body_language_score': body_language_score,
                'speech_quality_score': speech_quality_score,
                'overall_score': overall_score,
                'filler_word_count': filler_word_count,
                'speaking_pace_wpm': speaking_pace_wpm,
                'posture_score': posture_score,
                'gesture_score': gesture_score,
                'eye_contact_score': None,  # Not yet implemented
                'recommendations': json.dumps(recommendations),
                'detailed_metrics': json.dumps(detailed_metrics),
                'processed_at': datetime.utcnow()
            })
            
            analysis_id = result.fetchone()[0]
            session.commit()
            
            logger.info(f"Stored analysis result with ID: {analysis_id}")
            
            return {
                'analysis_id': str(analysis_id),
                'recording_id': recording_id,
                'stored_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing analysis results: {e}")
            raise
        finally:
            session.close()
    
    def update_recording_status(self, recording_id: str, status: str) -> None:
        """
        Update recording status
        
        Args:
            recording_id: Recording UUID
            status: New status value
        """
        session = self.SessionLocal()
        
        try:
            logger.info(f"Updating recording {recording_id} status to {status}")
            
            query = text("""
                UPDATE recordings
                SET status = :status, updated_at = :updated_at
                WHERE id = :recording_id
            """)
            
            session.execute(query, {
                'recording_id': recording_id,
                'status': status,
                'updated_at': datetime.utcnow()
            })
            
            session.commit()
            
            logger.info(f"Updated recording status successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating recording status: {e}")
            raise
        finally:
            session.close()
    
    def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recording information
        
        Args:
            recording_id: Recording UUID
            
        Returns:
            Dict with recording info or None
        """
        session = self.SessionLocal()
        
        try:
            query = text("""
                SELECT id, user_id, title, status, video_s3_key, audio_s3_key,
                       duration_seconds, file_size_bytes, created_at
                FROM recordings
                WHERE id = :recording_id
            """)
            
            result = session.execute(query, {'recording_id': recording_id})
            row = result.fetchone()
            
            if row:
                return {
                    'id': str(row[0]),
                    'user_id': str(row[1]),
                    'title': row[2],
                    'status': row[3],
                    'video_s3_key': row[4],
                    'audio_s3_key': row[5],
                    'duration_seconds': row[6],
                    'file_size_bytes': row[7],
                    'created_at': row[8].isoformat() if row[8] else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recording info: {e}")
            return None
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connection health"""
        session = self.SessionLocal()
        
        try:
            session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
        finally:
            session.close()

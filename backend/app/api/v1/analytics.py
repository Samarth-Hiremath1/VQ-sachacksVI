from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import logging

from ...core.database import get_db
from ...core.dependencies import get_current_user
from ...models.user import User
from ...schemas.analytics import (
    DetailedAnalyticsResponse,
    MetricsSummary,
    ComprehensiveMetrics,
    ProgressAnalysis
)
from ...services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{recording_id}/detailed", response_model=DetailedAnalyticsResponse)
async def get_detailed_analytics(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive analytics for a recording including:
    - Detailed metrics for all aspects (posture, gestures, speech, etc.)
    - Historical comparison and progress analysis
    - Visualization data for charts and graphs
    """
    
    analytics_service = AnalyticsService(db)
    
    try:
        analytics = analytics_service.get_detailed_analytics(
            recording_id=recording_id,
            user_id=current_user.id
        )
        
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "ANALYTICS_NOT_FOUND",
                    "message": "Analytics data not found for this recording"
                }
            )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching detailed analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "ANALYTICS_ERROR",
                "message": "An error occurred while generating analytics"
            }
        )


@router.get("/{recording_id}/summary", response_model=MetricsSummary)
async def get_metrics_summary(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a quick summary of key metrics for a recording.
    Useful for dashboard displays and quick overviews.
    """
    
    analytics_service = AnalyticsService(db)
    
    try:
        summary = analytics_service.get_metrics_summary(recording_id)
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "SUMMARY_NOT_FOUND",
                    "message": "Metrics summary not found for this recording"
                }
            )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SUMMARY_ERROR",
                "message": "An error occurred while generating summary"
            }
        )


@router.get("/{recording_id}/metrics", response_model=ComprehensiveMetrics)
async def get_comprehensive_metrics(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive metrics for a recording without historical comparison.
    Includes detailed breakdowns of all analysis aspects.
    """
    
    analytics_service = AnalyticsService(db)
    
    try:
        metrics = analytics_service.calculate_comprehensive_metrics(recording_id)
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "METRICS_NOT_FOUND",
                    "message": "Metrics not found for this recording"
                }
            )
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comprehensive metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "METRICS_ERROR",
                "message": "An error occurred while calculating metrics"
            }
        )


@router.get("/progress", response_model=ProgressAnalysis)
async def get_progress_analysis(
    current_recording_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's progress analysis and trends over time.
    Shows improvement patterns and areas needing attention.
    """
    
    analytics_service = AnalyticsService(db)
    
    try:
        progress = analytics_service.calculate_progress_analysis(
            user_id=current_user.id,
            current_recording_id=current_recording_id
        )
        
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "PROGRESS_NOT_FOUND",
                    "message": "No presentation history found for progress analysis"
                }
            )
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating progress analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "PROGRESS_ERROR",
                "message": "An error occurred while analyzing progress"
            }
        )


@router.get("/dashboard/overview")
async def get_dashboard_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get overview data for the analytics dashboard.
    Includes recent presentations, overall stats, and quick insights.
    """
    
    from ...services.recording_service import RecordingService
    from ...core.minio_client import get_minio_client
    
    analytics_service = AnalyticsService(db)
    
    try:
        # Get user's recording stats
        minio_client = get_minio_client()
        recording_service = RecordingService(db, minio_client)
        recording_stats = recording_service.get_recording_stats(current_user.id)
        
        # Get recent recordings with analysis
        recent_recordings = recording_service.get_user_recordings(
            user_id=current_user.id,
            page=1,
            per_page=5,
            status_filter="analyzed"
        )
        
        # Get progress analysis
        progress = analytics_service.calculate_progress_analysis(current_user.id)
        
        # Compile overview
        overview = {
            "user_id": str(current_user.id),
            "recording_stats": recording_stats,
            "recent_presentations": [
                {
                    "id": str(rec.id),
                    "title": rec.title,
                    "created_at": rec.created_at.isoformat(),
                    "duration_seconds": rec.duration_seconds
                }
                for rec in recent_recordings.recordings
            ],
            "progress_summary": {
                "total_presentations": progress.total_presentations if progress else 0,
                "overall_trend": progress.overall_score_trend.trend if progress else "stable",
                "improvement_percentage": progress.overall_score_trend.improvement_percentage if progress else 0,
                "most_improved_areas": progress.most_improved_areas if progress else [],
                "needs_attention_areas": progress.needs_attention_areas if progress else []
            } if progress else None
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"Error fetching dashboard overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "OVERVIEW_ERROR",
                "message": "An error occurred while fetching dashboard overview"
            }
        )



@router.get("/{recording_id}/recommendations")
async def get_recommendations(
    recording_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized coaching recommendations for a recording.
    Includes practice exercises, improvement goals, and actionable feedback.
    """
    
    from ...services.recommendations_service import RecommendationsService
    
    recommendations_service = RecommendationsService(db)
    
    try:
        recommendations = recommendations_service.generate_recommendations(
            recording_id=recording_id,
            user_id=current_user.id
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "RECOMMENDATIONS_NOT_FOUND",
                    "message": "Unable to generate recommendations for this recording"
                }
            )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "RECOMMENDATIONS_ERROR",
                "message": "An error occurred while generating recommendations"
            }
        )

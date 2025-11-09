from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import statistics

from ..models.analysis_result import AnalysisResult
from ..models.recording import Recording
from ..schemas.analytics import (
    PostureMetrics,
    GestureMetrics,
    SpeechQualityMetrics,
    FillerWordMetrics,
    EyeContactMetrics,
    ComprehensiveMetrics,
    HistoricalComparison,
    ProgressAnalysis,
    VisualizationData,
    DetailedAnalyticsResponse,
    MetricsSummary
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for calculating comprehensive analytics and metrics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_comprehensive_metrics(
        self, 
        recording_id: UUID
    ) -> Optional[ComprehensiveMetrics]:
        """Calculate detailed metrics from analysis results"""
        
        # Get analysis result
        analysis = self.db.query(AnalysisResult).filter(
            AnalysisResult.recording_id == recording_id
        ).first()
        
        if not analysis:
            logger.warning(f"No analysis result found for recording {recording_id}")
            return None
        
        # Get recording for duration
        recording = self.db.query(Recording).filter(
            Recording.id == recording_id
        ).first()
        
        if not recording:
            logger.warning(f"Recording {recording_id} not found")
            return None
        
        # Extract detailed metrics from JSONB
        detailed_metrics = analysis.detailed_metrics or {}
        
        # Calculate posture metrics
        posture_metrics = self._calculate_posture_metrics(
            analysis.posture_score,
            detailed_metrics.get('posture', {})
        )
        
        # Calculate gesture metrics
        gesture_metrics = self._calculate_gesture_metrics(
            analysis.gesture_score,
            detailed_metrics.get('gestures', {})
        )
        
        # Calculate speech quality metrics
        speech_metrics = self._calculate_speech_quality_metrics(
            analysis.speech_quality_score,
            analysis.speaking_pace_wpm,
            detailed_metrics.get('speech', {})
        )
        
        # Calculate filler word metrics
        filler_metrics = self._calculate_filler_word_metrics(
            analysis.filler_word_count,
            recording.duration_seconds or 0,
            detailed_metrics.get('filler_words', {})
        )
        
        # Calculate eye contact metrics
        eye_contact_metrics = self._calculate_eye_contact_metrics(
            analysis.eye_contact_score,
            detailed_metrics.get('eye_contact', {})
        )
        
        # Calculate overall confidence
        confidence_level = self._calculate_confidence_level(detailed_metrics)
        
        return ComprehensiveMetrics(
            recording_id=recording_id,
            overall_score=float(analysis.overall_score or 0),
            body_language_score=float(analysis.body_language_score or 0),
            speech_quality_score=float(analysis.speech_quality_score or 0),
            posture_metrics=posture_metrics,
            gesture_metrics=gesture_metrics,
            speech_quality_metrics=speech_metrics,
            filler_word_metrics=filler_metrics,
            eye_contact_metrics=eye_contact_metrics,
            duration_seconds=recording.duration_seconds or 0,
            processed_at=analysis.processed_at,
            confidence_level=confidence_level
        )
    
    def _calculate_posture_metrics(
        self, 
        posture_score: Optional[Decimal],
        posture_data: Dict[str, Any]
    ) -> PostureMetrics:
        """Calculate detailed posture metrics"""
        
        overall_score = float(posture_score or 0)
        
        # Extract or calculate sub-scores
        head_position = posture_data.get('head_position_score', overall_score * 0.95)
        shoulder_alignment = posture_data.get('shoulder_alignment_score', overall_score * 0.98)
        spine_alignment = posture_data.get('spine_alignment_score', overall_score * 0.92)
        stability = posture_data.get('stability_score', overall_score * 0.90)
        confidence = posture_data.get('confidence', 0.85)
        
        # Extract timeline data
        timeline = posture_data.get('timeline', [])
        
        # Identify problem areas
        problem_areas = []
        if head_position < 70:
            problem_areas.append("Head position needs improvement")
        if shoulder_alignment < 70:
            problem_areas.append("Shoulder alignment could be better")
        if spine_alignment < 70:
            problem_areas.append("Spine alignment needs attention")
        if stability < 70:
            problem_areas.append("Body stability could be improved")
        
        return PostureMetrics(
            overall_score=overall_score,
            head_position_score=head_position,
            shoulder_alignment_score=shoulder_alignment,
            spine_alignment_score=spine_alignment,
            stability_score=stability,
            confidence_level=confidence,
            posture_timeline=timeline,
            problem_areas=problem_areas
        )
    
    def _calculate_gesture_metrics(
        self,
        gesture_score: Optional[Decimal],
        gesture_data: Dict[str, Any]
    ) -> GestureMetrics:
        """Calculate detailed gesture metrics"""
        
        overall_score = float(gesture_score or 0)
        
        # Extract or calculate sub-scores
        hand_movement = gesture_data.get('hand_movement_score', overall_score * 0.95)
        variety = gesture_data.get('variety_score', overall_score * 0.90)
        timing = gesture_data.get('timing_score', overall_score * 0.93)
        open_gestures = gesture_data.get('open_gestures_percentage', 65.0)
        
        # Extract gesture counts
        total_gestures = gesture_data.get('total_count', 0)
        gesture_types = gesture_data.get('types', {})
        timeline = gesture_data.get('timeline', [])
        
        return GestureMetrics(
            overall_score=overall_score,
            hand_movement_score=hand_movement,
            gesture_variety_score=variety,
            gesture_timing_score=timing,
            open_gestures_percentage=open_gestures,
            total_gestures=total_gestures,
            gesture_types=gesture_types,
            gesture_timeline=timeline
        )
    
    def _calculate_speech_quality_metrics(
        self,
        speech_score: Optional[Decimal],
        pace_wpm: Optional[Decimal],
        speech_data: Dict[str, Any]
    ) -> SpeechQualityMetrics:
        """Calculate detailed speech quality metrics"""
        
        overall_score = float(speech_score or 0)
        speaking_pace = float(pace_wpm or 0)
        
        # Extract or calculate sub-scores
        clarity = speech_data.get('clarity_score', overall_score * 0.95)
        volume = speech_data.get('volume_score', overall_score * 0.90)
        pace_score = speech_data.get('pace_score', overall_score * 0.92)
        energy = speech_data.get('energy_score', overall_score * 0.88)
        
        # Optimal pace range (typically 140-160 WPM)
        optimal_range = {
            'min': 140.0,
            'max': 160.0,
            'ideal': 150.0
        }
        
        # Volume variation
        volume_variation = speech_data.get('volume_variation', 25.0)
        
        # Pause patterns
        pause_patterns = speech_data.get('pause_patterns', {
            'total_pauses': 0,
            'avg_pause_duration': 0.0,
            'effective_pauses': 0
        })
        
        # Timeline data
        pace_timeline = speech_data.get('pace_timeline', [])
        volume_timeline = speech_data.get('volume_timeline', [])
        
        return SpeechQualityMetrics(
            overall_score=overall_score,
            clarity_score=clarity,
            volume_score=volume,
            pace_score=pace_score,
            energy_score=energy,
            speaking_pace_wpm=speaking_pace,
            optimal_pace_range=optimal_range,
            volume_variation=volume_variation,
            pause_patterns=pause_patterns,
            pace_timeline=pace_timeline,
            volume_timeline=volume_timeline
        )
    
    def _calculate_filler_word_metrics(
        self,
        filler_count: Optional[int],
        duration_seconds: int,
        filler_data: Dict[str, Any]
    ) -> FillerWordMetrics:
        """Calculate filler word metrics"""
        
        total_count = filler_count or 0
        
        # Calculate rate per minute
        duration_minutes = max(duration_seconds / 60.0, 0.1)  # Avoid division by zero
        filler_per_minute = total_count / duration_minutes
        
        # Calculate percentage (assuming ~150 words per minute average)
        estimated_total_words = duration_minutes * 150
        filler_percentage = (total_count / max(estimated_total_words, 1)) * 100
        
        # Extract breakdown
        breakdown = filler_data.get('breakdown', {})
        most_common = max(breakdown.items(), key=lambda x: x[1])[0] if breakdown else None
        
        # Timeline
        timeline = filler_data.get('timeline', [])
        
        return FillerWordMetrics(
            total_count=total_count,
            filler_words_per_minute=filler_per_minute,
            filler_word_percentage=min(filler_percentage, 100.0),
            filler_word_breakdown=breakdown,
            most_common_filler=most_common,
            filler_word_timeline=timeline
        )
    
    def _calculate_eye_contact_metrics(
        self,
        eye_contact_score: Optional[Decimal],
        eye_contact_data: Dict[str, Any]
    ) -> EyeContactMetrics:
        """Calculate eye contact and engagement metrics"""
        
        overall_score = float(eye_contact_score or 0)
        
        # Extract metrics
        direct_gaze = eye_contact_data.get('direct_gaze_percentage', overall_score * 0.9)
        stability = eye_contact_data.get('stability_score', overall_score * 0.95)
        engagement = eye_contact_data.get('engagement_score', overall_score * 0.92)
        
        # Patterns
        gaze_patterns = eye_contact_data.get('patterns', {})
        distraction_count = eye_contact_data.get('distraction_count', 0)
        
        return EyeContactMetrics(
            overall_score=overall_score,
            direct_gaze_percentage=direct_gaze,
            gaze_stability_score=stability,
            engagement_score=engagement,
            gaze_patterns=gaze_patterns,
            distraction_count=distraction_count
        )
    
    def _calculate_confidence_level(self, detailed_metrics: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis"""
        
        confidence_scores = []
        
        # Collect confidence scores from different analysis components
        if 'posture' in detailed_metrics:
            confidence_scores.append(detailed_metrics['posture'].get('confidence', 0.8))
        if 'gestures' in detailed_metrics:
            confidence_scores.append(detailed_metrics['gestures'].get('confidence', 0.8))
        if 'speech' in detailed_metrics:
            confidence_scores.append(detailed_metrics['speech'].get('confidence', 0.8))
        
        # Return average confidence or default
        return statistics.mean(confidence_scores) if confidence_scores else 0.75

    
    def calculate_progress_analysis(
        self,
        user_id: UUID,
        current_recording_id: Optional[UUID] = None
    ) -> Optional[ProgressAnalysis]:
        """Calculate user's progress and trends over time"""
        
        # Get all user's analysis results ordered by date
        results = (
            self.db.query(AnalysisResult, Recording)
            .join(Recording, AnalysisResult.recording_id == Recording.id)
            .filter(Recording.user_id == user_id)
            .order_by(AnalysisResult.processed_at)
            .all()
        )
        
        if not results:
            logger.info(f"No analysis results found for user {user_id}")
            return None
        
        total_presentations = len(results)
        
        # Extract historical data
        historical_data = self._extract_historical_data(results)
        
        # Calculate trends for each metric
        overall_trend = self._calculate_metric_trend(
            'overall_score',
            historical_data,
            current_recording_id
        )
        
        body_language_trend = self._calculate_metric_trend(
            'body_language_score',
            historical_data,
            current_recording_id
        )
        
        speech_quality_trend = self._calculate_metric_trend(
            'speech_quality_score',
            historical_data,
            current_recording_id
        )
        
        posture_trend = self._calculate_metric_trend(
            'posture_score',
            historical_data,
            current_recording_id
        )
        
        gesture_trend = self._calculate_metric_trend(
            'gesture_score',
            historical_data,
            current_recording_id
        )
        
        pace_trend = self._calculate_metric_trend(
            'speaking_pace_wpm',
            historical_data,
            current_recording_id,
            optimal_value=150.0
        )
        
        filler_word_trend = self._calculate_metric_trend(
            'filler_word_count',
            historical_data,
            current_recording_id,
            lower_is_better=True
        )
        
        # Identify most improved and needs attention areas
        all_trends = [
            ('Overall Performance', overall_trend),
            ('Body Language', body_language_trend),
            ('Speech Quality', speech_quality_trend),
            ('Posture', posture_trend),
            ('Gestures', gesture_trend),
            ('Speaking Pace', pace_trend),
            ('Filler Words', filler_word_trend)
        ]
        
        most_improved = sorted(
            [{'area': name, 'improvement': trend.improvement_percentage} 
             for name, trend in all_trends if trend.improvement_percentage > 0],
            key=lambda x: x['improvement'],
            reverse=True
        )[:3]
        
        needs_attention = sorted(
            [{'area': name, 'improvement': trend.improvement_percentage}
             for name, trend in all_trends if trend.improvement_percentage < -5],
            key=lambda x: x['improvement']
        )[:3]
        
        # Calculate time-based performance
        recent_performance = self._calculate_time_based_performance(
            historical_data,
            days=30
        )
        
        long_term_performance = self._calculate_time_based_performance(
            historical_data,
            days=None  # All time
        )
        
        return ProgressAnalysis(
            user_id=user_id,
            total_presentations=total_presentations,
            overall_score_trend=overall_trend,
            body_language_trend=body_language_trend,
            speech_quality_trend=speech_quality_trend,
            posture_trend=posture_trend,
            gesture_trend=gesture_trend,
            pace_trend=pace_trend,
            filler_word_trend=filler_word_trend,
            most_improved_areas=most_improved,
            needs_attention_areas=needs_attention,
            recent_performance=recent_performance,
            long_term_performance=long_term_performance
        )
    
    def _extract_historical_data(
        self,
        results: List[Tuple[AnalysisResult, Recording]]
    ) -> List[Dict[str, Any]]:
        """Extract historical data from analysis results"""
        
        historical_data = []
        
        for analysis, recording in results:
            historical_data.append({
                'recording_id': recording.id,
                'processed_at': analysis.processed_at,
                'overall_score': float(analysis.overall_score or 0),
                'body_language_score': float(analysis.body_language_score or 0),
                'speech_quality_score': float(analysis.speech_quality_score or 0),
                'posture_score': float(analysis.posture_score or 0),
                'gesture_score': float(analysis.gesture_score or 0),
                'eye_contact_score': float(analysis.eye_contact_score or 0),
                'speaking_pace_wpm': float(analysis.speaking_pace_wpm or 0),
                'filler_word_count': analysis.filler_word_count or 0
            })
        
        return historical_data
    
    def _calculate_metric_trend(
        self,
        metric_name: str,
        historical_data: List[Dict[str, Any]],
        current_recording_id: Optional[UUID] = None,
        optimal_value: Optional[float] = None,
        lower_is_better: bool = False
    ) -> HistoricalComparison:
        """Calculate trend for a specific metric"""
        
        values = [d[metric_name] for d in historical_data if d[metric_name] is not None]
        
        if not values:
            # Return default comparison
            return HistoricalComparison(
                metric_name=metric_name,
                current_value=0.0,
                historical_average=0.0,
                improvement_percentage=0.0,
                trend='stable',
                best_score=0.0,
                worst_score=0.0,
                historical_data=[]
            )
        
        # Get current value
        current_value = values[-1] if not current_recording_id else next(
            (d[metric_name] for d in historical_data if d['recording_id'] == current_recording_id),
            values[-1]
        )
        
        # Calculate statistics
        historical_average = statistics.mean(values[:-1]) if len(values) > 1 else values[0]
        
        if lower_is_better:
            best_score = min(values)
            worst_score = max(values)
            improvement_percentage = ((historical_average - current_value) / max(historical_average, 1)) * 100
        else:
            best_score = max(values)
            worst_score = min(values)
            improvement_percentage = ((current_value - historical_average) / max(historical_average, 1)) * 100
        
        # Determine trend
        if improvement_percentage > 5:
            trend = 'improving'
        elif improvement_percentage < -5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Format historical data for visualization
        formatted_history = [
            {
                'date': d['processed_at'].isoformat(),
                'value': d[metric_name],
                'recording_id': str(d['recording_id'])
            }
            for d in historical_data
        ]
        
        return HistoricalComparison(
            metric_name=metric_name,
            current_value=current_value,
            historical_average=historical_average,
            improvement_percentage=improvement_percentage,
            trend=trend,
            best_score=best_score,
            worst_score=worst_score,
            historical_data=formatted_history
        )
    
    def _calculate_time_based_performance(
        self,
        historical_data: List[Dict[str, Any]],
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate performance metrics for a specific time period"""
        
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            filtered_data = [
                d for d in historical_data 
                if d['processed_at'] >= cutoff_date
            ]
        else:
            filtered_data = historical_data
        
        if not filtered_data:
            return {
                'presentation_count': 0,
                'average_overall_score': 0.0,
                'average_body_language_score': 0.0,
                'average_speech_quality_score': 0.0
            }
        
        return {
            'presentation_count': len(filtered_data),
            'average_overall_score': statistics.mean([d['overall_score'] for d in filtered_data]),
            'average_body_language_score': statistics.mean([d['body_language_score'] for d in filtered_data]),
            'average_speech_quality_score': statistics.mean([d['speech_quality_score'] for d in filtered_data]),
            'best_overall_score': max([d['overall_score'] for d in filtered_data]),
            'latest_score': filtered_data[-1]['overall_score']
        }
    
    def generate_visualization_data(
        self,
        comprehensive_metrics: ComprehensiveMetrics,
        progress_analysis: Optional[ProgressAnalysis] = None
    ) -> VisualizationData:
        """Generate structured data for frontend visualizations"""
        
        # Radar chart data (multi-dimensional view)
        radar_chart = {
            'Posture': comprehensive_metrics.posture_metrics.overall_score,
            'Gestures': comprehensive_metrics.gesture_metrics.overall_score,
            'Speech Quality': comprehensive_metrics.speech_quality_metrics.overall_score,
            'Eye Contact': comprehensive_metrics.eye_contact_metrics.overall_score,
            'Pace': comprehensive_metrics.speech_quality_metrics.pace_score,
            'Clarity': comprehensive_metrics.speech_quality_metrics.clarity_score
        }
        
        # Score timeline (from posture and speech timelines)
        score_timeline = []
        
        # Combine timelines from different metrics
        if comprehensive_metrics.posture_metrics.posture_timeline:
            score_timeline.extend(comprehensive_metrics.posture_metrics.posture_timeline)
        
        # Metric comparisons (current vs optimal/average)
        metric_comparisons = [
            {
                'metric': 'Overall Score',
                'current': comprehensive_metrics.overall_score,
                'optimal': 85.0,
                'category': 'overall'
            },
            {
                'metric': 'Body Language',
                'current': comprehensive_metrics.body_language_score,
                'optimal': 80.0,
                'category': 'body_language'
            },
            {
                'metric': 'Speech Quality',
                'current': comprehensive_metrics.speech_quality_score,
                'optimal': 80.0,
                'category': 'speech'
            },
            {
                'metric': 'Speaking Pace (WPM)',
                'current': comprehensive_metrics.speech_quality_metrics.speaking_pace_wpm,
                'optimal': 150.0,
                'category': 'speech'
            }
        ]
        
        # Add historical comparison if available
        if progress_analysis:
            metric_comparisons.append({
                'metric': 'vs. Your Average',
                'current': comprehensive_metrics.overall_score,
                'optimal': progress_analysis.overall_score_trend.historical_average,
                'category': 'comparison'
            })
        
        # Performance heatmap (simplified - would need more granular data)
        performance_heatmap = self._generate_performance_heatmap(comprehensive_metrics)
        
        # Progress indicators
        progress_indicators = {
            'overall': {
                'current': comprehensive_metrics.overall_score,
                'target': 85.0,
                'progress_percentage': (comprehensive_metrics.overall_score / 85.0) * 100
            },
            'body_language': {
                'current': comprehensive_metrics.body_language_score,
                'target': 80.0,
                'progress_percentage': (comprehensive_metrics.body_language_score / 80.0) * 100
            },
            'speech_quality': {
                'current': comprehensive_metrics.speech_quality_score,
                'target': 80.0,
                'progress_percentage': (comprehensive_metrics.speech_quality_score / 80.0) * 100
            }
        }
        
        return VisualizationData(
            radar_chart=radar_chart,
            score_timeline=score_timeline,
            metric_comparisons=metric_comparisons,
            performance_heatmap=performance_heatmap,
            progress_indicators=progress_indicators
        )
    
    def _generate_performance_heatmap(
        self,
        comprehensive_metrics: ComprehensiveMetrics
    ) -> List[List[float]]:
        """Generate a performance heatmap matrix"""
        
        # Create a simple heatmap based on different metric categories
        # Rows: Time segments, Columns: Metric categories
        
        # For now, create a simplified version
        # In production, this would use actual time-series data
        
        heatmap = [
            [
                comprehensive_metrics.posture_metrics.overall_score / 100,
                comprehensive_metrics.gesture_metrics.overall_score / 100,
                comprehensive_metrics.speech_quality_metrics.overall_score / 100,
                comprehensive_metrics.eye_contact_metrics.overall_score / 100
            ]
        ]
        
        return heatmap
    
    def get_detailed_analytics(
        self,
        recording_id: UUID,
        user_id: UUID
    ) -> Optional[DetailedAnalyticsResponse]:
        """Get complete analytics with all metrics and visualizations"""
        
        # Calculate comprehensive metrics
        comprehensive_metrics = self.calculate_comprehensive_metrics(recording_id)
        
        if not comprehensive_metrics:
            return None
        
        # Calculate progress analysis
        progress_analysis = self.calculate_progress_analysis(user_id, recording_id)
        
        # Generate visualization data
        visualization_data = self.generate_visualization_data(
            comprehensive_metrics,
            progress_analysis
        )
        
        return DetailedAnalyticsResponse(
            recording_id=recording_id,
            user_id=user_id,
            comprehensive_metrics=comprehensive_metrics,
            progress_analysis=progress_analysis,
            visualization_data=visualization_data,
            generated_at=datetime.utcnow(),
            analysis_version="1.0.0"
        )
    
    def get_metrics_summary(
        self,
        recording_id: UUID
    ) -> Optional[MetricsSummary]:
        """Get a summary of key metrics for quick display"""
        
        analysis = self.db.query(AnalysisResult).filter(
            AnalysisResult.recording_id == recording_id
        ).first()
        
        if not analysis:
            return None
        
        recording = self.db.query(Recording).filter(
            Recording.id == recording_id
        ).first()
        
        if not recording:
            return None
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        scores = {
            'Posture': float(analysis.posture_score or 0),
            'Gestures': float(analysis.gesture_score or 0),
            'Eye Contact': float(analysis.eye_contact_score or 0),
            'Speech Quality': float(analysis.speech_quality_score or 0)
        }
        
        for metric, score in scores.items():
            if score >= 80:
                strengths.append(f"Excellent {metric.lower()}")
            elif score < 60:
                weaknesses.append(f"{metric} needs improvement")
        
        # Check filler words
        if analysis.filler_word_count and analysis.filler_word_count > 10:
            weaknesses.append("High filler word usage")
        
        # Check speaking pace
        if analysis.speaking_pace_wpm:
            pace = float(analysis.speaking_pace_wpm)
            if 140 <= pace <= 160:
                strengths.append("Optimal speaking pace")
            elif pace < 120 or pace > 180:
                weaknesses.append("Speaking pace needs adjustment")
        
        return MetricsSummary(
            recording_id=recording_id,
            overall_score=float(analysis.overall_score or 0),
            body_language_score=float(analysis.body_language_score or 0),
            speech_quality_score=float(analysis.speech_quality_score or 0),
            strengths=strengths[:3],  # Top 3
            weaknesses=weaknesses[:3],  # Top 3
            duration_seconds=recording.duration_seconds or 0,
            filler_word_count=analysis.filler_word_count or 0,
            speaking_pace_wpm=float(analysis.speaking_pace_wpm or 0),
            processed_at=analysis.processed_at
        )

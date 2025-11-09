from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal


class PostureMetrics(BaseModel):
    """Detailed posture analysis metrics"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall posture score (0-100)")
    head_position_score: float = Field(..., ge=0, le=100, description="Head position and alignment score")
    shoulder_alignment_score: float = Field(..., ge=0, le=100, description="Shoulder alignment score")
    spine_alignment_score: float = Field(..., ge=0, le=100, description="Spine alignment score")
    stability_score: float = Field(..., ge=0, le=100, description="Body stability and balance score")
    confidence_level: float = Field(..., ge=0, le=1, description="Confidence in posture detection")
    
    # Time-series data for visualization
    posture_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Posture scores over time")
    problem_areas: List[str] = Field(default_factory=list, description="Identified posture issues")


class GestureMetrics(BaseModel):
    """Detailed gesture analysis metrics"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall gesture effectiveness score")
    hand_movement_score: float = Field(..., ge=0, le=100, description="Hand movement naturalness score")
    gesture_variety_score: float = Field(..., ge=0, le=100, description="Variety of gestures used")
    gesture_timing_score: float = Field(..., ge=0, le=100, description="Timing and coordination score")
    open_gestures_percentage: float = Field(..., ge=0, le=100, description="Percentage of open/confident gestures")
    
    # Gesture counts and patterns
    total_gestures: int = Field(..., ge=0, description="Total number of gestures detected")
    gesture_types: Dict[str, int] = Field(default_factory=dict, description="Count by gesture type")
    gesture_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Gestures over time")


class SpeechQualityMetrics(BaseModel):
    """Detailed speech quality analysis metrics"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall speech quality score")
    clarity_score: float = Field(..., ge=0, le=100, description="Speech clarity and articulation score")
    volume_score: float = Field(..., ge=0, le=100, description="Volume consistency and appropriateness")
    pace_score: float = Field(..., ge=0, le=100, description="Speaking pace appropriateness")
    energy_score: float = Field(..., ge=0, le=100, description="Vocal energy and engagement")
    
    # Detailed metrics
    speaking_pace_wpm: float = Field(..., ge=0, description="Words per minute")
    optimal_pace_range: Dict[str, float] = Field(default_factory=dict, description="Optimal WPM range")
    volume_variation: float = Field(..., ge=0, le=100, description="Volume variation percentage")
    pause_patterns: Dict[str, Any] = Field(default_factory=dict, description="Pause analysis")
    
    # Time-series data
    pace_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Speaking pace over time")
    volume_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Volume levels over time")


class FillerWordMetrics(BaseModel):
    """Filler word analysis metrics"""
    total_count: int = Field(..., ge=0, description="Total filler words detected")
    filler_words_per_minute: float = Field(..., ge=0, description="Filler words per minute rate")
    filler_word_percentage: float = Field(..., ge=0, le=100, description="Percentage of total words")
    
    # Breakdown by type
    filler_word_breakdown: Dict[str, int] = Field(default_factory=dict, description="Count by filler word type")
    most_common_filler: Optional[str] = Field(None, description="Most frequently used filler word")
    
    # Timeline data
    filler_word_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Filler words over time")


class EyeContactMetrics(BaseModel):
    """Eye contact and engagement metrics"""
    overall_score: float = Field(..., ge=0, le=100, description="Overall eye contact score")
    direct_gaze_percentage: float = Field(..., ge=0, le=100, description="Percentage of time with direct gaze")
    gaze_stability_score: float = Field(..., ge=0, le=100, description="Gaze stability and focus score")
    engagement_score: float = Field(..., ge=0, le=100, description="Overall engagement indicator")
    
    # Patterns
    gaze_patterns: Dict[str, Any] = Field(default_factory=dict, description="Gaze direction patterns")
    distraction_count: int = Field(..., ge=0, description="Number of significant gaze distractions")


class ComprehensiveMetrics(BaseModel):
    """Complete presentation metrics for visualization"""
    recording_id: UUID
    
    # Core scores
    overall_score: float = Field(..., ge=0, le=100, description="Overall presentation score")
    body_language_score: float = Field(..., ge=0, le=100, description="Body language composite score")
    speech_quality_score: float = Field(..., ge=0, le=100, description="Speech quality composite score")
    
    # Detailed breakdowns
    posture_metrics: PostureMetrics
    gesture_metrics: GestureMetrics
    speech_quality_metrics: SpeechQualityMetrics
    filler_word_metrics: FillerWordMetrics
    eye_contact_metrics: EyeContactMetrics
    
    # Metadata
    duration_seconds: int
    processed_at: datetime
    confidence_level: float = Field(..., ge=0, le=1, description="Overall analysis confidence")


class HistoricalComparison(BaseModel):
    """Comparison with user's historical performance"""
    metric_name: str
    current_value: float
    historical_average: float
    improvement_percentage: float
    trend: str = Field(..., description="'improving', 'declining', or 'stable'")
    best_score: float
    worst_score: float
    
    # Visualization data
    historical_data: List[Dict[str, Any]] = Field(default_factory=list, description="Historical values over time")


class ProgressAnalysis(BaseModel):
    """User progress tracking and trends"""
    user_id: UUID
    total_presentations: int
    
    # Overall trends
    overall_score_trend: HistoricalComparison
    body_language_trend: HistoricalComparison
    speech_quality_trend: HistoricalComparison
    
    # Specific metric trends
    posture_trend: HistoricalComparison
    gesture_trend: HistoricalComparison
    pace_trend: HistoricalComparison
    filler_word_trend: HistoricalComparison
    
    # Improvement areas
    most_improved_areas: List[Dict[str, Any]] = Field(default_factory=list, description="Top improving metrics")
    needs_attention_areas: List[Dict[str, Any]] = Field(default_factory=list, description="Areas needing work")
    
    # Time-based analysis
    recent_performance: Dict[str, Any] = Field(default_factory=dict, description="Last 30 days performance")
    long_term_performance: Dict[str, Any] = Field(default_factory=dict, description="All-time performance")


class VisualizationData(BaseModel):
    """Structured data for frontend charts and visualizations"""
    
    # Radar chart data (for overall performance)
    radar_chart: Dict[str, float] = Field(default_factory=dict, description="Multi-dimensional performance view")
    
    # Line charts (trends over time)
    score_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Scores over presentation duration")
    
    # Bar charts (comparisons)
    metric_comparisons: List[Dict[str, Any]] = Field(default_factory=list, description="Metric comparisons")
    
    # Heatmaps (patterns)
    performance_heatmap: List[List[float]] = Field(default_factory=list, description="Performance intensity map")
    
    # Progress indicators
    progress_indicators: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Progress bars data")


class DetailedAnalyticsResponse(BaseModel):
    """Complete analytics response with all metrics and visualizations"""
    recording_id: UUID
    user_id: UUID
    
    # Core metrics
    comprehensive_metrics: ComprehensiveMetrics
    
    # Historical context
    progress_analysis: Optional[ProgressAnalysis] = None
    
    # Visualization data
    visualization_data: VisualizationData
    
    # Metadata
    generated_at: datetime
    analysis_version: str = "1.0.0"


class MetricsSummary(BaseModel):
    """Summary of key metrics for dashboard display"""
    recording_id: UUID
    overall_score: float
    body_language_score: float
    speech_quality_score: float
    
    # Key highlights
    strengths: List[str] = Field(default_factory=list, description="Top performing areas")
    weaknesses: List[str] = Field(default_factory=list, description="Areas needing improvement")
    
    # Quick stats
    duration_seconds: int
    filler_word_count: int
    speaking_pace_wpm: float
    
    processed_at: datetime

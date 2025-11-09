from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from enum import Enum


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationCategory(str, Enum):
    """Categories of recommendations"""
    POSTURE = "posture"
    GESTURES = "gestures"
    SPEECH_QUALITY = "speech_quality"
    FILLER_WORDS = "filler_words"
    EYE_CONTACT = "eye_contact"
    PACING = "pacing"
    ENERGY = "energy"
    OVERALL = "overall"


class PracticeExercise(BaseModel):
    """A specific practice exercise"""
    title: str = Field(..., description="Exercise title")
    description: str = Field(..., description="Detailed exercise description")
    duration_minutes: int = Field(..., ge=1, description="Estimated duration in minutes")
    difficulty: str = Field(..., description="'beginner', 'intermediate', or 'advanced'")
    category: RecommendationCategory
    instructions: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    tips: List[str] = Field(default_factory=list, description="Helpful tips")


class Recommendation(BaseModel):
    """A single coaching recommendation"""
    id: str = Field(..., description="Unique recommendation identifier")
    category: RecommendationCategory
    priority: RecommendationPriority
    title: str = Field(..., description="Short recommendation title")
    description: str = Field(..., description="Detailed recommendation description")
    
    # Context
    current_score: float = Field(..., ge=0, le=100, description="Current performance score")
    target_score: float = Field(..., ge=0, le=100, description="Target performance score")
    improvement_potential: float = Field(..., ge=0, le=100, description="Potential improvement percentage")
    
    # Actionable guidance
    action_items: List[str] = Field(default_factory=list, description="Specific actions to take")
    practice_exercises: List[PracticeExercise] = Field(default_factory=list, description="Recommended exercises")
    
    # Resources
    related_resources: List[Dict[str, str]] = Field(default_factory=list, description="Links to helpful resources")
    
    # Metadata
    estimated_time_to_improve: str = Field(..., description="Estimated time to see improvement")


class ImprovementGoal(BaseModel):
    """A specific improvement goal"""
    goal_id: str
    category: RecommendationCategory
    title: str
    description: str
    target_metric: str
    current_value: float
    target_value: float
    deadline: Optional[datetime] = None
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Progress milestones")
    status: str = Field(default="active", description="'active', 'completed', or 'paused'")


class PersonalizedCoachingPlan(BaseModel):
    """Complete personalized coaching plan"""
    user_id: UUID
    recording_id: UUID
    
    # Overall assessment
    overall_assessment: str = Field(..., description="Summary of current performance")
    key_strengths: List[str] = Field(default_factory=list, description="Top 3-5 strengths")
    key_weaknesses: List[str] = Field(default_factory=list, description="Top 3-5 areas for improvement")
    
    # Recommendations
    high_priority_recommendations: List[Recommendation] = Field(default_factory=list)
    medium_priority_recommendations: List[Recommendation] = Field(default_factory=list)
    low_priority_recommendations: List[Recommendation] = Field(default_factory=list)
    
    # Goals
    suggested_goals: List[ImprovementGoal] = Field(default_factory=list)
    
    # Practice plan
    weekly_practice_plan: Dict[str, List[PracticeExercise]] = Field(
        default_factory=dict,
        description="Organized practice schedule by day"
    )
    
    # Metadata
    generated_at: datetime
    next_review_date: Optional[datetime] = None


class RecommendationFeedback(BaseModel):
    """User feedback on a recommendation"""
    recommendation_id: str
    user_id: UUID
    helpful: bool
    implemented: bool = False
    notes: Optional[str] = None
    submitted_at: datetime


class ProgressMilestone(BaseModel):
    """A milestone in user's improvement journey"""
    milestone_id: str
    title: str
    description: str
    category: RecommendationCategory
    achieved: bool = False
    achieved_at: Optional[datetime] = None
    celebration_message: str = Field(..., description="Congratulatory message when achieved")


class CoachingInsight(BaseModel):
    """AI-generated coaching insight"""
    insight_type: str = Field(..., description="'pattern', 'trend', 'achievement', or 'alert'")
    title: str
    message: str
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    actionable: bool = Field(default=True, description="Whether this insight has actionable recommendations")
    related_recommendations: List[str] = Field(default_factory=list, description="Related recommendation IDs")


class RecommendationsResponse(BaseModel):
    """API response for recommendations"""
    recording_id: UUID
    user_id: UUID
    
    # Coaching plan
    coaching_plan: PersonalizedCoachingPlan
    
    # Additional insights
    insights: List[CoachingInsight] = Field(default_factory=list)
    
    # Milestones
    available_milestones: List[ProgressMilestone] = Field(default_factory=list)
    achieved_milestones: List[ProgressMilestone] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime
    version: str = "1.0.0"

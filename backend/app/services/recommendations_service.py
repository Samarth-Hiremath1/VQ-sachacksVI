from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from ..models.analysis_result import AnalysisResult
from ..models.recording import Recording
from ..schemas.recommendations import (
    Recommendation,
    RecommendationPriority,
    RecommendationCategory,
    PracticeExercise,
    ImprovementGoal,
    PersonalizedCoachingPlan,
    CoachingInsight,
    ProgressMilestone,
    RecommendationsResponse
)
from ..schemas.analytics import ComprehensiveMetrics, ProgressAnalysis
from .analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class RecommendationsService:
    """Service for generating personalized coaching recommendations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.analytics_service = AnalyticsService(db)
    
    def generate_recommendations(
        self,
        recording_id: UUID,
        user_id: UUID
    ) -> Optional[RecommendationsResponse]:
        """Generate comprehensive personalized recommendations"""
        
        # Get comprehensive metrics
        metrics = self.analytics_service.calculate_comprehensive_metrics(recording_id)
        if not metrics:
            logger.warning(f"No metrics found for recording {recording_id}")
            return None
        
        # Get progress analysis
        progress = self.analytics_service.calculate_progress_analysis(user_id, recording_id)
        
        # Generate coaching plan
        coaching_plan = self._generate_coaching_plan(
            recording_id,
            user_id,
            metrics,
            progress
        )
        
        # Generate insights
        insights = self._generate_insights(metrics, progress)
        
        # Generate milestones
        available_milestones, achieved_milestones = self._generate_milestones(
            metrics,
            progress
        )
        
        return RecommendationsResponse(
            recording_id=recording_id,
            user_id=user_id,
            coaching_plan=coaching_plan,
            insights=insights,
            available_milestones=available_milestones,
            achieved_milestones=achieved_milestones,
            generated_at=datetime.utcnow(),
            version="1.0.0"
        )
    
    def _generate_coaching_plan(
        self,
        recording_id: UUID,
        user_id: UUID,
        metrics: ComprehensiveMetrics,
        progress: Optional[ProgressAnalysis]
    ) -> PersonalizedCoachingPlan:
        """Generate a personalized coaching plan"""
        
        # Overall assessment
        overall_assessment = self._generate_overall_assessment(metrics, progress)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(metrics)
        weaknesses = self._identify_weaknesses(metrics)
        
        # Generate recommendations by priority
        all_recommendations = self._generate_all_recommendations(metrics, progress)
        
        high_priority = [r for r in all_recommendations if r.priority == RecommendationPriority.HIGH]
        medium_priority = [r for r in all_recommendations if r.priority == RecommendationPriority.MEDIUM]
        low_priority = [r for r in all_recommendations if r.priority == RecommendationPriority.LOW]
        
        # Generate improvement goals
        suggested_goals = self._generate_improvement_goals(metrics, weaknesses)
        
        # Create weekly practice plan
        weekly_plan = self._create_weekly_practice_plan(all_recommendations)
        
        # Calculate next review date (1 week from now)
        next_review = datetime.utcnow() + timedelta(days=7)
        
        return PersonalizedCoachingPlan(
            user_id=user_id,
            recording_id=recording_id,
            overall_assessment=overall_assessment,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            high_priority_recommendations=high_priority,
            medium_priority_recommendations=medium_priority,
            low_priority_recommendations=low_priority,
            suggested_goals=suggested_goals,
            weekly_practice_plan=weekly_plan,
            generated_at=datetime.utcnow(),
            next_review_date=next_review
        )
    
    def _generate_overall_assessment(
        self,
        metrics: ComprehensiveMetrics,
        progress: Optional[ProgressAnalysis]
    ) -> str:
        """Generate an overall assessment summary"""
        
        score = metrics.overall_score
        
        # Base assessment on score
        if score >= 85:
            base = "Excellent presentation! You demonstrate strong skills across all areas."
        elif score >= 75:
            base = "Very good presentation with solid fundamentals and room for refinement."
        elif score >= 65:
            base = "Good presentation with clear strengths and specific areas to improve."
        elif score >= 50:
            base = "Developing presentation with potential for significant improvement."
        else:
            base = "Early stage presentation skills with many opportunities for growth."
        
        # Add progress context if available
        if progress and progress.total_presentations > 1:
            trend = progress.overall_score_trend.trend
            if trend == "improving":
                base += f" You're showing great progress with {abs(progress.overall_score_trend.improvement_percentage):.1f}% improvement!"
            elif trend == "declining":
                base += " Let's focus on getting back to your previous performance level."
            else:
                base += " Your performance is consistent - let's work on breaking through to the next level."
        
        return base
    
    def _identify_strengths(self, metrics: ComprehensiveMetrics) -> List[str]:
        """Identify top strengths from metrics"""
        
        strengths = []
        
        # Check each metric category
        if metrics.posture_metrics.overall_score >= 80:
            strengths.append(f"Excellent posture (score: {metrics.posture_metrics.overall_score:.0f}/100)")
        
        if metrics.gesture_metrics.overall_score >= 80:
            strengths.append(f"Effective use of gestures (score: {metrics.gesture_metrics.overall_score:.0f}/100)")
        
        if metrics.speech_quality_metrics.overall_score >= 80:
            strengths.append(f"Strong speech quality (score: {metrics.speech_quality_metrics.overall_score:.0f}/100)")
        
        if metrics.eye_contact_metrics.overall_score >= 80:
            strengths.append(f"Great eye contact (score: {metrics.eye_contact_metrics.overall_score:.0f}/100)")
        
        # Check speaking pace
        pace = metrics.speech_quality_metrics.speaking_pace_wpm
        if 140 <= pace <= 160:
            strengths.append(f"Optimal speaking pace ({pace:.0f} WPM)")
        
        # Check filler words
        if metrics.filler_word_metrics.filler_words_per_minute < 2:
            strengths.append("Minimal use of filler words")
        
        # Return top 5 strengths
        return strengths[:5]
    
    def _identify_weaknesses(self, metrics: ComprehensiveMetrics) -> List[str]:
        """Identify areas needing improvement"""
        
        weaknesses = []
        
        # Check each metric category
        if metrics.posture_metrics.overall_score < 70:
            weaknesses.append(f"Posture needs improvement (score: {metrics.posture_metrics.overall_score:.0f}/100)")
        
        if metrics.gesture_metrics.overall_score < 70:
            weaknesses.append(f"Gesture usage could be enhanced (score: {metrics.gesture_metrics.overall_score:.0f}/100)")
        
        if metrics.speech_quality_metrics.overall_score < 70:
            weaknesses.append(f"Speech quality needs work (score: {metrics.speech_quality_metrics.overall_score:.0f}/100)")
        
        if metrics.eye_contact_metrics.overall_score < 70:
            weaknesses.append(f"Eye contact could be improved (score: {metrics.eye_contact_metrics.overall_score:.0f}/100)")
        
        # Check speaking pace
        pace = metrics.speech_quality_metrics.speaking_pace_wpm
        if pace < 120:
            weaknesses.append(f"Speaking pace is too slow ({pace:.0f} WPM)")
        elif pace > 180:
            weaknesses.append(f"Speaking pace is too fast ({pace:.0f} WPM)")
        
        # Check filler words
        if metrics.filler_word_metrics.filler_words_per_minute > 5:
            weaknesses.append(f"High filler word usage ({metrics.filler_word_metrics.filler_words_per_minute:.1f} per minute)")
        
        # Return top 5 weaknesses
        return weaknesses[:5]

    
    def _generate_all_recommendations(
        self,
        metrics: ComprehensiveMetrics,
        progress: Optional[ProgressAnalysis]
    ) -> List[Recommendation]:
        """Generate all recommendations based on metrics"""
        
        recommendations = []
        
        # Posture recommendations
        if metrics.posture_metrics.overall_score < 80:
            recommendations.append(self._create_posture_recommendation(metrics.posture_metrics))
        
        # Gesture recommendations
        if metrics.gesture_metrics.overall_score < 80:
            recommendations.append(self._create_gesture_recommendation(metrics.gesture_metrics))
        
        # Speech quality recommendations
        if metrics.speech_quality_metrics.overall_score < 80:
            recommendations.append(self._create_speech_quality_recommendation(metrics.speech_quality_metrics))
        
        # Filler word recommendations
        if metrics.filler_word_metrics.filler_words_per_minute > 3:
            recommendations.append(self._create_filler_word_recommendation(metrics.filler_word_metrics))
        
        # Eye contact recommendations
        if metrics.eye_contact_metrics.overall_score < 80:
            recommendations.append(self._create_eye_contact_recommendation(metrics.eye_contact_metrics))
        
        # Pacing recommendations
        pace = metrics.speech_quality_metrics.speaking_pace_wpm
        if pace < 130 or pace > 170:
            recommendations.append(self._create_pacing_recommendation(metrics.speech_quality_metrics))
        
        return recommendations
    
    def _create_posture_recommendation(self, posture_metrics) -> Recommendation:
        """Create posture-specific recommendation"""
        
        score = posture_metrics.overall_score
        priority = RecommendationPriority.HIGH if score < 60 else RecommendationPriority.MEDIUM
        
        exercises = [
            PracticeExercise(
                title="Wall Stand Exercise",
                description="Practice standing against a wall to improve posture alignment",
                duration_minutes=5,
                difficulty="beginner",
                category=RecommendationCategory.POSTURE,
                instructions=[
                    "Stand with your back against a wall",
                    "Ensure heels, buttocks, shoulders, and head touch the wall",
                    "Hold for 30 seconds, repeat 5 times",
                    "Practice this daily before presentations"
                ],
                tips=[
                    "Keep your chin parallel to the floor",
                    "Engage your core muscles",
                    "Breathe naturally throughout"
                ]
            ),
            PracticeExercise(
                title="Shoulder Roll Exercise",
                description="Release tension and improve shoulder alignment",
                duration_minutes=3,
                difficulty="beginner",
                category=RecommendationCategory.POSTURE,
                instructions=[
                    "Roll shoulders backward in circular motion 10 times",
                    "Roll shoulders forward 10 times",
                    "Repeat 3 sets"
                ],
                tips=[
                    "Move slowly and deliberately",
                    "Focus on full range of motion"
                ]
            )
        ]
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.POSTURE,
            priority=priority,
            title="Improve Your Posture",
            description="Your posture could be more confident and aligned. Good posture projects confidence and helps with breathing.",
            current_score=score,
            target_score=85.0,
            improvement_potential=85.0 - score,
            action_items=[
                "Practice standing tall with shoulders back",
                "Keep your head level and chin parallel to the floor",
                "Distribute weight evenly on both feet",
                "Avoid slouching or leaning to one side"
            ],
            practice_exercises=exercises,
            related_resources=[
                {"title": "Power Posing Guide", "url": "#"},
                {"title": "Body Language Basics", "url": "#"}
            ],
            estimated_time_to_improve="2-3 weeks with daily practice"
        )
    
    def _create_gesture_recommendation(self, gesture_metrics) -> Recommendation:
        """Create gesture-specific recommendation"""
        
        score = gesture_metrics.overall_score
        priority = RecommendationPriority.MEDIUM if score < 70 else RecommendationPriority.LOW
        
        exercises = [
            PracticeExercise(
                title="Gesture Variety Practice",
                description="Expand your gesture repertoire for more engaging presentations",
                duration_minutes=10,
                difficulty="intermediate",
                category=RecommendationCategory.GESTURES,
                instructions=[
                    "Record yourself explaining a concept",
                    "Use different gestures: pointing, counting, showing size, emphasizing",
                    "Review and identify which gestures felt natural",
                    "Practice incorporating 3-5 different gesture types"
                ],
                tips=[
                    "Keep gestures above waist level",
                    "Use open palm gestures for inclusivity",
                    "Match gesture size to room size"
                ]
            )
        ]
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.GESTURES,
            priority=priority,
            title="Enhance Your Gestures",
            description="More varied and purposeful gestures will make your presentation more engaging and help emphasize key points.",
            current_score=score,
            target_score=85.0,
            improvement_potential=85.0 - score,
            action_items=[
                "Use gestures to emphasize key points",
                "Vary your gestures throughout the presentation",
                "Keep hands visible and above waist level",
                "Avoid repetitive or nervous gestures"
            ],
            practice_exercises=exercises,
            related_resources=[
                {"title": "Effective Gesture Guide", "url": "#"}
            ],
            estimated_time_to_improve="1-2 weeks"
        )
    
    def _create_speech_quality_recommendation(self, speech_metrics) -> Recommendation:
        """Create speech quality recommendation"""
        
        score = speech_metrics.overall_score
        priority = RecommendationPriority.HIGH if score < 60 else RecommendationPriority.MEDIUM
        
        exercises = [
            PracticeExercise(
                title="Vocal Warm-up Routine",
                description="Prepare your voice for clear, confident speaking",
                duration_minutes=5,
                difficulty="beginner",
                category=RecommendationCategory.SPEECH_QUALITY,
                instructions=[
                    "Hum at different pitches for 1 minute",
                    "Practice tongue twisters slowly, then faster",
                    "Read aloud with exaggerated articulation",
                    "Practice projecting your voice"
                ],
                tips=[
                    "Stay hydrated before speaking",
                    "Warm up your voice before presentations"
                ]
            )
        ]
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.SPEECH_QUALITY,
            priority=priority,
            title="Improve Speech Clarity and Quality",
            description="Enhancing your vocal delivery will make your message clearer and more impactful.",
            current_score=score,
            target_score=85.0,
            improvement_potential=85.0 - score,
            action_items=[
                "Articulate words clearly and completely",
                "Vary your volume for emphasis",
                "Use vocal energy to maintain engagement",
                "Practice breathing from your diaphragm"
            ],
            practice_exercises=exercises,
            related_resources=[
                {"title": "Voice Training Basics", "url": "#"}
            ],
            estimated_time_to_improve="3-4 weeks"
        )
    
    def _create_filler_word_recommendation(self, filler_metrics) -> Recommendation:
        """Create filler word reduction recommendation"""
        
        count = filler_metrics.filler_words_per_minute
        priority = RecommendationPriority.HIGH if count > 5 else RecommendationPriority.MEDIUM
        
        exercises = [
            PracticeExercise(
                title="Pause Practice",
                description="Replace filler words with purposeful pauses",
                duration_minutes=10,
                difficulty="intermediate",
                category=RecommendationCategory.FILLER_WORDS,
                instructions=[
                    "Record yourself speaking on a topic for 2 minutes",
                    "Count your filler words",
                    "Re-record, consciously pausing instead of using fillers",
                    "Compare the two recordings"
                ],
                tips=[
                    "Embrace silence - pauses are powerful",
                    "Slow down when you feel a filler word coming",
                    "Practice with a friend who can signal when you use fillers"
                ]
            )
        ]
        
        most_common = filler_metrics.most_common_filler or "filler words"
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.FILLER_WORDS,
            priority=priority,
            title="Reduce Filler Words",
            description=f"You're using {count:.1f} filler words per minute. Your most common is '{most_common}'. Reducing these will make you sound more confident.",
            current_score=max(0, 100 - (count * 10)),
            target_score=85.0,
            improvement_potential=min(50, count * 10),
            action_items=[
                "Become aware of your filler word patterns",
                "Replace filler words with brief pauses",
                "Slow down your speaking pace slightly",
                "Practice speaking with intentional pauses"
            ],
            practice_exercises=exercises,
            related_resources=[
                {"title": "Breaking the Filler Word Habit", "url": "#"}
            ],
            estimated_time_to_improve="4-6 weeks"
        )
    
    def _create_eye_contact_recommendation(self, eye_contact_metrics) -> Recommendation:
        """Create eye contact recommendation"""
        
        score = eye_contact_metrics.overall_score
        priority = RecommendationPriority.MEDIUM
        
        exercises = [
            PracticeExercise(
                title="Eye Contact Practice",
                description="Build comfort with maintaining eye contact",
                duration_minutes=5,
                difficulty="beginner",
                category=RecommendationCategory.EYE_CONTACT,
                instructions=[
                    "Practice with a friend or mirror",
                    "Maintain eye contact for 3-5 seconds before moving",
                    "Use the triangle method: eyes, nose, mouth",
                    "Practice scanning the room naturally"
                ],
                tips=[
                    "Don't stare - blink naturally",
                    "In virtual settings, look at the camera",
                    "With groups, make eye contact with different people"
                ]
            )
        ]
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.EYE_CONTACT,
            priority=priority,
            title="Strengthen Eye Contact",
            description="Better eye contact will help you connect with your audience and project confidence.",
            current_score=score,
            target_score=85.0,
            improvement_potential=85.0 - score,
            action_items=[
                "Maintain eye contact for 3-5 seconds at a time",
                "Scan the room to include everyone",
                "Look at the camera in virtual presentations",
                "Avoid looking down at notes too frequently"
            ],
            practice_exercises=exercises,
            related_resources=[
                {"title": "Eye Contact Techniques", "url": "#"}
            ],
            estimated_time_to_improve="2-3 weeks"
        )
    
    def _create_pacing_recommendation(self, speech_metrics) -> Recommendation:
        """Create speaking pace recommendation"""
        
        pace = speech_metrics.speaking_pace_wpm
        
        if pace < 130:
            title = "Increase Your Speaking Pace"
            description = f"Your pace of {pace:.0f} WPM is slower than optimal (140-160 WPM). A slightly faster pace will maintain audience engagement."
            action_items = [
                "Practice speaking slightly faster",
                "Reduce pause duration between sentences",
                "Maintain energy and enthusiasm",
                "Record and time yourself regularly"
            ]
        else:
            title = "Slow Down Your Speaking Pace"
            description = f"Your pace of {pace:.0f} WPM is faster than optimal (140-160 WPM). Slowing down will improve clarity and comprehension."
            action_items = [
                "Consciously slow down your delivery",
                "Add purposeful pauses between key points",
                "Focus on clear articulation",
                "Practice with a metronome or timer"
            ]
        
        exercises = [
            PracticeExercise(
                title="Pacing Practice",
                description="Develop optimal speaking pace",
                duration_minutes=10,
                difficulty="intermediate",
                category=RecommendationCategory.PACING,
                instructions=[
                    "Read a 150-word passage",
                    "Time yourself - aim for 60 seconds",
                    "Adjust pace and repeat",
                    "Practice until consistent"
                ],
                tips=[
                    "Use a timer to track your pace",
                    "Record yourself to hear the difference",
                    "Vary pace for emphasis"
                ]
            )
        ]
        
        priority = RecommendationPriority.HIGH if abs(pace - 150) > 30 else RecommendationPriority.MEDIUM
        
        return Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.PACING,
            priority=priority,
            title=title,
            description=description,
            current_score=max(0, 100 - abs(pace - 150)),
            target_score=85.0,
            improvement_potential=min(40, abs(pace - 150)),
            action_items=action_items,
            practice_exercises=exercises,
            related_resources=[
                {"title": "Speaking Pace Guide", "url": "#"}
            ],
            estimated_time_to_improve="2-3 weeks"
        )
    
    def _generate_improvement_goals(
        self,
        metrics: ComprehensiveMetrics,
        weaknesses: List[str]
    ) -> List[ImprovementGoal]:
        """Generate specific improvement goals"""
        
        goals = []
        
        # Create goals for top weaknesses
        if metrics.posture_metrics.overall_score < 75:
            goals.append(ImprovementGoal(
                goal_id=str(uuid4()),
                category=RecommendationCategory.POSTURE,
                title="Achieve Excellent Posture",
                description="Improve posture score to 85 or higher",
                target_metric="posture_score",
                current_value=metrics.posture_metrics.overall_score,
                target_value=85.0,
                deadline=datetime.utcnow() + timedelta(days=30),
                milestones=[
                    {"score": 70, "description": "Good posture foundation"},
                    {"score": 80, "description": "Strong posture consistency"},
                    {"score": 85, "description": "Excellent posture mastery"}
                ],
                status="active"
            ))
        
        if metrics.filler_word_metrics.filler_words_per_minute > 3:
            goals.append(ImprovementGoal(
                goal_id=str(uuid4()),
                category=RecommendationCategory.FILLER_WORDS,
                title="Reduce Filler Words",
                description="Reduce filler words to less than 2 per minute",
                target_metric="filler_words_per_minute",
                current_value=metrics.filler_word_metrics.filler_words_per_minute,
                target_value=2.0,
                deadline=datetime.utcnow() + timedelta(days=45),
                milestones=[
                    {"rate": 4, "description": "Noticeable reduction"},
                    {"rate": 3, "description": "Significant improvement"},
                    {"rate": 2, "description": "Minimal filler words"}
                ],
                status="active"
            ))
        
        return goals[:3]  # Return top 3 goals
    
    def _create_weekly_practice_plan(
        self,
        recommendations: List[Recommendation]
    ) -> Dict[str, List[PracticeExercise]]:
        """Create a weekly practice schedule"""
        
        # Collect all exercises
        all_exercises = []
        for rec in recommendations:
            all_exercises.extend(rec.practice_exercises)
        
        # Distribute exercises across the week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_plan = {}
        
        for i, day in enumerate(days):
            # Assign 1-2 exercises per day
            day_exercises = []
            if i < len(all_exercises):
                day_exercises.append(all_exercises[i])
            if i + 7 < len(all_exercises):
                day_exercises.append(all_exercises[i + 7])
            
            weekly_plan[day] = day_exercises
        
        return weekly_plan
    
    def _generate_insights(
        self,
        metrics: ComprehensiveMetrics,
        progress: Optional[ProgressAnalysis]
    ) -> List[CoachingInsight]:
        """Generate AI coaching insights"""
        
        insights = []
        
        # Pattern insights
        if metrics.filler_word_metrics.most_common_filler:
            insights.append(CoachingInsight(
                insight_type="pattern",
                title="Filler Word Pattern Detected",
                message=f"You frequently use '{metrics.filler_word_metrics.most_common_filler}' as a filler word. Being aware of this pattern is the first step to reducing it.",
                supporting_data={
                    "most_common": metrics.filler_word_metrics.most_common_filler,
                    "count": metrics.filler_word_metrics.total_count
                },
                actionable=True,
                related_recommendations=[]
            ))
        
        # Trend insights
        if progress and progress.overall_score_trend.trend == "improving":
            insights.append(CoachingInsight(
                insight_type="trend",
                title="Positive Progress Trend",
                message=f"Great work! Your overall performance has improved by {progress.overall_score_trend.improvement_percentage:.1f}% compared to your average.",
                supporting_data={
                    "improvement": progress.overall_score_trend.improvement_percentage,
                    "current": progress.overall_score_trend.current_value,
                    "average": progress.overall_score_trend.historical_average
                },
                actionable=False,
                related_recommendations=[]
            ))
        
        # Achievement insights
        if metrics.overall_score >= 85:
            insights.append(CoachingInsight(
                insight_type="achievement",
                title="Excellent Performance!",
                message="You've achieved an excellent overall score. Keep up the great work!",
                supporting_data={"score": metrics.overall_score},
                actionable=False,
                related_recommendations=[]
            ))
        
        return insights
    
    def _generate_milestones(
        self,
        metrics: ComprehensiveMetrics,
        progress: Optional[ProgressAnalysis]
    ) -> tuple[List[ProgressMilestone], List[ProgressMilestone]]:
        """Generate available and achieved milestones"""
        
        available = []
        achieved = []
        
        # Define milestones
        milestones_def = [
            {
                "id": "first_presentation",
                "title": "First Presentation",
                "description": "Complete your first analyzed presentation",
                "category": RecommendationCategory.OVERALL,
                "threshold": 1,
                "metric": "total_presentations",
                "celebration": "🎉 Congratulations on your first presentation!"
            },
            {
                "id": "score_70",
                "title": "Good Performance",
                "description": "Achieve an overall score of 70 or higher",
                "category": RecommendationCategory.OVERALL,
                "threshold": 70,
                "metric": "overall_score",
                "celebration": "🌟 Great job reaching a good performance level!"
            },
            {
                "id": "score_85",
                "title": "Excellent Performance",
                "description": "Achieve an overall score of 85 or higher",
                "category": RecommendationCategory.OVERALL,
                "threshold": 85,
                "metric": "overall_score",
                "celebration": "🏆 Outstanding! You've achieved excellence!"
            },
            {
                "id": "low_fillers",
                "title": "Filler Word Master",
                "description": "Reduce filler words to less than 2 per minute",
                "category": RecommendationCategory.FILLER_WORDS,
                "threshold": 2,
                "metric": "filler_words_per_minute",
                "celebration": "💬 Excellent! You've mastered filler word control!"
            }
        ]
        
        # Check each milestone
        for m_def in milestones_def:
            milestone = ProgressMilestone(
                milestone_id=m_def["id"],
                title=m_def["title"],
                description=m_def["description"],
                category=m_def["category"],
                celebration_message=m_def["celebration"]
            )
            
            # Check if achieved
            is_achieved = False
            
            if m_def["metric"] == "total_presentations" and progress:
                is_achieved = progress.total_presentations >= m_def["threshold"]
            elif m_def["metric"] == "overall_score":
                is_achieved = metrics.overall_score >= m_def["threshold"]
            elif m_def["metric"] == "filler_words_per_minute":
                is_achieved = metrics.filler_word_metrics.filler_words_per_minute <= m_def["threshold"]
            
            if is_achieved:
                milestone.achieved = True
                milestone.achieved_at = datetime.utcnow()
                achieved.append(milestone)
            else:
                available.append(milestone)
        
        return available, achieved

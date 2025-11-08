"""
Result Aggregator for combining ML analysis results

Aggregates body language and speech analysis results into overall presentation score.
Requirements: 2.1, 3.2, 4.4
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregates analysis results from multiple ML services"""
    
    # Scoring weights for overall presentation score
    WEIGHTS = {
        'body_language': 0.40,  # 40% weight
        'speech_quality': 0.35,  # 35% weight
        'content_delivery': 0.25  # 25% weight (derived from pace, filler words)
    }
    
    # Thresholds for recommendations
    SCORE_THRESHOLDS = {
        'excellent': 85,
        'good': 70,
        'fair': 55,
        'needs_improvement': 0
    }
    
    def __init__(self):
        logger.info("Initialized ResultAggregator")
    
    def aggregate_results(
        self,
        body_language_score: float,
        posture_score: float,
        gesture_score: float,
        body_language_metrics: Dict[str, Any],
        speech_quality_score: float,
        speaking_pace_wpm: float,
        filler_word_count: int,
        speech_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate all analysis results into comprehensive presentation score
        
        Args:
            body_language_score: Overall body language score (0-100)
            posture_score: Posture quality score (0-100)
            gesture_score: Gesture effectiveness score (0-100)
            body_language_metrics: Detailed body language metrics
            speech_quality_score: Overall speech quality score (0-100)
            speaking_pace_wpm: Speaking rate in words per minute
            filler_word_count: Count of filler words detected
            speech_metrics: Detailed speech metrics
            
        Returns:
            Dict containing aggregated results and recommendations
        """
        try:
            logger.info("Aggregating analysis results")
            
            # Calculate content delivery score based on pace and filler words
            content_delivery_score = self._calculate_content_delivery_score(
                speaking_pace_wpm, filler_word_count, speech_metrics
            )
            
            # Calculate weighted overall score
            overall_score = (
                body_language_score * self.WEIGHTS['body_language'] +
                speech_quality_score * self.WEIGHTS['speech_quality'] +
                content_delivery_score * self.WEIGHTS['content_delivery']
            )
            
            # Round to 2 decimal places
            overall_score = round(overall_score, 2)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_score=overall_score,
                body_language_score=body_language_score,
                posture_score=posture_score,
                gesture_score=gesture_score,
                speech_quality_score=speech_quality_score,
                speaking_pace_wpm=speaking_pace_wpm,
                filler_word_count=filler_word_count,
                body_language_metrics=body_language_metrics,
                speech_metrics=speech_metrics
            )
            
            # Compile detailed metrics
            detailed_metrics = {
                'body_language': {
                    'overall_score': body_language_score,
                    'posture_score': posture_score,
                    'gesture_score': gesture_score,
                    'detailed_metrics': body_language_metrics
                },
                'speech': {
                    'overall_quality': speech_quality_score,
                    'speaking_pace_wpm': speaking_pace_wpm,
                    'filler_word_count': filler_word_count,
                    'detailed_metrics': speech_metrics
                },
                'content_delivery': {
                    'score': content_delivery_score,
                    'pace_rating': self._rate_speaking_pace(speaking_pace_wpm),
                    'filler_word_rating': self._rate_filler_words(filler_word_count)
                },
                'weights_applied': self.WEIGHTS,
                'aggregated_at': datetime.utcnow().isoformat()
            }
            
            result = {
                'overall_score': overall_score,
                'performance_level': self._get_performance_level(overall_score),
                'recommendations': recommendations,
                'detailed_metrics': detailed_metrics,
                'component_scores': {
                    'body_language': body_language_score,
                    'speech_quality': speech_quality_score,
                    'content_delivery': content_delivery_score
                }
            }
            
            logger.info(f"Aggregation complete. Overall score: {overall_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            raise
    
    def _calculate_content_delivery_score(
        self,
        speaking_pace_wpm: float,
        filler_word_count: int,
        speech_metrics: Dict[str, Any]
    ) -> float:
        """Calculate content delivery score based on pace and filler words"""
        
        # Optimal speaking pace is 130-170 WPM
        pace_score = 100
        if speaking_pace_wpm < 100:
            pace_score = max(0, 100 - (100 - speaking_pace_wpm) * 2)
        elif speaking_pace_wpm > 180:
            pace_score = max(0, 100 - (speaking_pace_wpm - 180) * 1.5)
        elif speaking_pace_wpm < 130:
            pace_score = 100 - (130 - speaking_pace_wpm) * 0.5
        elif speaking_pace_wpm > 170:
            pace_score = 100 - (speaking_pace_wpm - 170) * 0.5
        
        # Filler word penalty (assuming ~5 minute presentation = 650 words)
        # More than 1 filler word per 100 words is problematic
        estimated_total_words = speaking_pace_wpm * 5  # Assume 5 min presentation
        filler_ratio = filler_word_count / max(estimated_total_words, 1) * 100
        
        filler_score = 100
        if filler_ratio > 1:
            filler_score = max(0, 100 - (filler_ratio - 1) * 20)
        
        # Weight pace and filler words
        content_delivery_score = (pace_score * 0.6 + filler_score * 0.4)
        
        return round(content_delivery_score, 2)
    
    def _rate_speaking_pace(self, wpm: float) -> str:
        """Rate speaking pace"""
        if 130 <= wpm <= 170:
            return "optimal"
        elif 110 <= wpm < 130 or 170 < wpm <= 190:
            return "acceptable"
        elif wpm < 110:
            return "too_slow"
        else:
            return "too_fast"
    
    def _rate_filler_words(self, count: int) -> str:
        """Rate filler word usage"""
        if count <= 5:
            return "excellent"
        elif count <= 10:
            return "good"
        elif count <= 20:
            return "needs_improvement"
        else:
            return "poor"
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score"""
        if score >= self.SCORE_THRESHOLDS['excellent']:
            return "excellent"
        elif score >= self.SCORE_THRESHOLDS['good']:
            return "good"
        elif score >= self.SCORE_THRESHOLDS['fair']:
            return "fair"
        else:
            return "needs_improvement"
    
    def _generate_recommendations(
        self,
        overall_score: float,
        body_language_score: float,
        posture_score: float,
        gesture_score: float,
        speech_quality_score: float,
        speaking_pace_wpm: float,
        filler_word_count: int,
        body_language_metrics: Dict[str, Any],
        speech_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on analysis"""
        
        recommendations = []
        
        # Body language recommendations
        if body_language_score < 70:
            if posture_score < 65:
                recommendations.append({
                    'category': 'body_language',
                    'priority': 'high',
                    'title': 'Improve Your Posture',
                    'description': 'Your posture needs attention. Stand tall with shoulders back and maintain an open stance.',
                    'actionable_tips': [
                        'Practice standing with your weight evenly distributed',
                        'Keep your shoulders relaxed and back',
                        'Avoid slouching or leaning to one side',
                        'Record yourself to check posture awareness'
                    ]
                })
            
            if gesture_score < 65:
                recommendations.append({
                    'category': 'body_language',
                    'priority': 'high',
                    'title': 'Use More Purposeful Gestures',
                    'description': 'Your gestures could be more effective. Use hand movements to emphasize key points.',
                    'actionable_tips': [
                        'Use open palm gestures to appear more trustworthy',
                        'Match gestures to your words for emphasis',
                        'Avoid repetitive or nervous movements',
                        'Practice gesturing in front of a mirror'
                    ]
                })
        
        # Speech quality recommendations
        if speech_quality_score < 70:
            recommendations.append({
                'category': 'speech',
                'priority': 'high',
                'title': 'Enhance Your Vocal Delivery',
                'description': 'Your speech quality could be improved. Focus on clarity, volume, and vocal variety.',
                'actionable_tips': [
                    'Practice vocal warm-up exercises',
                    'Vary your pitch and tone to maintain interest',
                    'Ensure adequate volume without shouting',
                    'Articulate words clearly, especially endings'
                ]
            })
        
        # Speaking pace recommendations
        pace_rating = self._rate_speaking_pace(speaking_pace_wpm)
        if pace_rating == 'too_slow':
            recommendations.append({
                'category': 'delivery',
                'priority': 'medium',
                'title': 'Increase Your Speaking Pace',
                'description': f'Your speaking pace ({speaking_pace_wpm:.0f} WPM) is slower than optimal. Aim for 130-170 WPM.',
                'actionable_tips': [
                    'Practice with a timer to increase pace naturally',
                    'Reduce unnecessary pauses between sentences',
                    'Stay energized and engaged with your content',
                    'Record and review to find comfortable faster pace'
                ]
            })
        elif pace_rating == 'too_fast':
            recommendations.append({
                'category': 'delivery',
                'priority': 'medium',
                'title': 'Slow Down Your Speaking Pace',
                'description': f'Your speaking pace ({speaking_pace_wpm:.0f} WPM) is faster than optimal. Aim for 130-170 WPM.',
                'actionable_tips': [
                    'Take deliberate pauses between key points',
                    'Practice breathing exercises to control pace',
                    'Focus on articulating each word clearly',
                    'Use pauses for emphasis and audience processing'
                ]
            })
        
        # Filler word recommendations
        filler_rating = self._rate_filler_words(filler_word_count)
        if filler_rating in ['needs_improvement', 'poor']:
            recommendations.append({
                'category': 'delivery',
                'priority': 'high' if filler_rating == 'poor' else 'medium',
                'title': 'Reduce Filler Words',
                'description': f'You used {filler_word_count} filler words. Work on eliminating "um", "uh", "like", and "you know".',
                'actionable_tips': [
                    'Pause silently instead of using filler words',
                    'Practice awareness by recording yourself',
                    'Slow down to give yourself time to think',
                    'Prepare and rehearse your content thoroughly'
                ]
            })
        
        # Positive reinforcement for strong areas
        if overall_score >= 85:
            recommendations.append({
                'category': 'encouragement',
                'priority': 'low',
                'title': 'Excellent Presentation!',
                'description': 'You demonstrated strong presentation skills across all areas.',
                'actionable_tips': [
                    'Continue practicing to maintain this level',
                    'Consider mentoring others in presentation skills',
                    'Challenge yourself with more complex topics',
                    'Experiment with advanced techniques'
                ]
            })
        elif overall_score >= 70:
            recommendations.append({
                'category': 'encouragement',
                'priority': 'low',
                'title': 'Good Progress!',
                'description': 'You have solid presentation fundamentals. Focus on the specific areas above to reach excellence.',
                'actionable_tips': [
                    'Review your detailed metrics to track improvement',
                    'Practice regularly to build confidence',
                    'Focus on one improvement area at a time'
                ]
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations

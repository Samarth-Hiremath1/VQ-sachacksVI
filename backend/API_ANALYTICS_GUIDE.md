# Analytics and Recommendations API Guide

## Base URL
All endpoints are prefixed with `/api/v1/analytics`

## Authentication
All endpoints require authentication via JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### 1. Get Detailed Analytics
**GET** `/api/v1/analytics/{recording_id}/detailed`

Returns comprehensive analytics including metrics, historical comparison, and visualization data.

**Response:**
```json
{
  "recording_id": "uuid",
  "user_id": "uuid",
  "comprehensive_metrics": {
    "overall_score": 78.5,
    "body_language_score": 75.0,
    "speech_quality_score": 82.0,
    "posture_metrics": { ... },
    "gesture_metrics": { ... },
    "speech_quality_metrics": { ... },
    "filler_word_metrics": { ... },
    "eye_contact_metrics": { ... }
  },
  "progress_analysis": {
    "total_presentations": 5,
    "overall_score_trend": { ... },
    "most_improved_areas": [ ... ],
    "needs_attention_areas": [ ... ]
  },
  "visualization_data": {
    "radar_chart": { ... },
    "score_timeline": [ ... ],
    "metric_comparisons": [ ... ]
  }
}
```

### 2. Get Metrics Summary
**GET** `/api/v1/analytics/{recording_id}/summary`

Returns a quick summary of key metrics for dashboard display.

**Response:**
```json
{
  "recording_id": "uuid",
  "overall_score": 78.5,
  "body_language_score": 75.0,
  "speech_quality_score": 82.0,
  "strengths": [
    "Excellent speech quality",
    "Optimal speaking pace"
  ],
  "weaknesses": [
    "Posture needs improvement",
    "High filler word usage"
  ],
  "duration_seconds": 300,
  "filler_word_count": 15,
  "speaking_pace_wpm": 145.0
}
```

### 3. Get Comprehensive Metrics
**GET** `/api/v1/analytics/{recording_id}/metrics`

Returns detailed metrics without historical comparison.

**Response:**
```json
{
  "recording_id": "uuid",
  "overall_score": 78.5,
  "body_language_score": 75.0,
  "speech_quality_score": 82.0,
  "posture_metrics": {
    "overall_score": 72.0,
    "head_position_score": 75.0,
    "shoulder_alignment_score": 70.0,
    "spine_alignment_score": 71.0,
    "stability_score": 73.0,
    "confidence_level": 0.85,
    "problem_areas": ["Shoulder alignment could be better"]
  },
  "gesture_metrics": { ... },
  "speech_quality_metrics": { ... },
  "filler_word_metrics": { ... },
  "eye_contact_metrics": { ... }
}
```

### 4. Get Progress Analysis
**GET** `/api/v1/analytics/progress?current_recording_id={uuid}`

Returns user's progress analysis and trends over time.

**Query Parameters:**
- `current_recording_id` (optional): UUID of current recording for context

**Response:**
```json
{
  "user_id": "uuid",
  "total_presentations": 5,
  "overall_score_trend": {
    "metric_name": "overall_score",
    "current_value": 78.5,
    "historical_average": 72.0,
    "improvement_percentage": 9.0,
    "trend": "improving",
    "best_score": 82.0,
    "worst_score": 65.0
  },
  "most_improved_areas": [
    {"area": "Speech Quality", "improvement": 15.5}
  ],
  "needs_attention_areas": [
    {"area": "Posture", "improvement": -5.2}
  ],
  "recent_performance": { ... },
  "long_term_performance": { ... }
}
```

### 5. Get Dashboard Overview
**GET** `/api/v1/analytics/dashboard/overview`

Returns overview data for the analytics dashboard.

**Response:**
```json
{
  "user_id": "uuid",
  "recording_stats": {
    "total_recordings": 10,
    "total_duration_seconds": 3600,
    "average_duration_seconds": 360
  },
  "recent_presentations": [ ... ],
  "progress_summary": {
    "total_presentations": 5,
    "overall_trend": "improving",
    "improvement_percentage": 9.0,
    "most_improved_areas": [ ... ],
    "needs_attention_areas": [ ... ]
  }
}
```

### 6. Get Recommendations
**GET** `/api/v1/analytics/{recording_id}/recommendations`

Returns personalized coaching recommendations with practice exercises and improvement goals.

**Response:**
```json
{
  "recording_id": "uuid",
  "user_id": "uuid",
  "coaching_plan": {
    "overall_assessment": "Good presentation with clear strengths...",
    "key_strengths": [
      "Excellent speech quality (score: 82/100)",
      "Optimal speaking pace (145 WPM)"
    ],
    "key_weaknesses": [
      "Posture needs improvement (score: 72/100)"
    ],
    "high_priority_recommendations": [
      {
        "id": "uuid",
        "category": "posture",
        "priority": "high",
        "title": "Improve Your Posture",
        "description": "Your posture could be more confident...",
        "current_score": 72.0,
        "target_score": 85.0,
        "improvement_potential": 13.0,
        "action_items": [
          "Practice standing tall with shoulders back",
          "Keep your head level and chin parallel to the floor"
        ],
        "practice_exercises": [
          {
            "title": "Wall Stand Exercise",
            "description": "Practice standing against a wall...",
            "duration_minutes": 5,
            "difficulty": "beginner",
            "instructions": [ ... ],
            "tips": [ ... ]
          }
        ],
        "estimated_time_to_improve": "2-3 weeks with daily practice"
      }
    ],
    "suggested_goals": [
      {
        "goal_id": "uuid",
        "category": "posture",
        "title": "Achieve Excellent Posture",
        "target_metric": "posture_score",
        "current_value": 72.0,
        "target_value": 85.0,
        "milestones": [ ... ]
      }
    ],
    "weekly_practice_plan": {
      "Monday": [ ... ],
      "Tuesday": [ ... ]
    }
  },
  "insights": [
    {
      "insight_type": "pattern",
      "title": "Filler Word Pattern Detected",
      "message": "You frequently use 'um' as a filler word..."
    }
  ],
  "available_milestones": [ ... ],
  "achieved_milestones": [ ... ]
}
```

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": {
    "error_code": "ANALYTICS_NOT_FOUND",
    "message": "Analytics data not found for this recording"
  }
}
```

**Common Error Codes:**
- `ANALYTICS_NOT_FOUND`: No analytics data available
- `METRICS_NOT_FOUND`: Metrics not found for recording
- `SUMMARY_NOT_FOUND`: Summary not available
- `PROGRESS_NOT_FOUND`: No presentation history
- `RECOMMENDATIONS_NOT_FOUND`: Unable to generate recommendations
- `ANALYTICS_ERROR`: General analytics error
- `RECOMMENDATIONS_ERROR`: General recommendations error

## Usage Examples

### JavaScript/TypeScript (Frontend)
```typescript
// Get detailed analytics
const response = await fetch(
  `/api/v1/analytics/${recordingId}/detailed`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);
const analytics = await response.json();

// Get recommendations
const recResponse = await fetch(
  `/api/v1/analytics/${recordingId}/recommendations`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);
const recommendations = await recResponse.json();
```

### Python
```python
import requests

# Get metrics summary
response = requests.get(
    f"/api/v1/analytics/{recording_id}/summary",
    headers={"Authorization": f"Bearer {token}"}
)
summary = response.json()

# Get progress analysis
response = requests.get(
    "/api/v1/analytics/progress",
    headers={"Authorization": f"Bearer {token}"}
)
progress = response.json()
```

## Data Visualization

The `visualization_data` object is optimized for common chart libraries:

### Radar Chart (Chart.js, Recharts)
```javascript
const radarData = analytics.visualization_data.radar_chart;
// { "Posture": 72, "Gestures": 78, "Speech Quality": 82, ... }
```

### Line Chart (Time Series)
```javascript
const timelineData = analytics.visualization_data.score_timeline;
// [{ timestamp: "...", score: 75 }, ...]
```

### Bar Chart (Comparisons)
```javascript
const comparisons = analytics.visualization_data.metric_comparisons;
// [{ metric: "Overall Score", current: 78.5, optimal: 85 }, ...]
```

## Best Practices

1. **Cache Results**: Analytics calculations can be expensive. Cache results on the frontend.
2. **Progressive Loading**: Load summary first, then detailed analytics on demand.
3. **Error Handling**: Always handle cases where analytics aren't available yet.
4. **Polling**: For newly processed recordings, poll the summary endpoint until available.
5. **Visualization**: Use the provided visualization_data structure for consistent charts.

## Rate Limiting

Analytics endpoints are subject to standard API rate limits:
- 100 requests per minute per user
- 1000 requests per hour per user

## Support

For issues or questions about the Analytics API, contact the development team.

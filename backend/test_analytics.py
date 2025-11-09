"""Tests for analytics and recommendations services"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_analytics_imports():
    """Test that analytics modules can be imported"""
    try:
        from app.services.analytics_service import AnalyticsService
        from app.schemas.analytics import (
            ComprehensiveMetrics,
            DetailedAnalyticsResponse,
            MetricsSummary,
            ProgressAnalysis
        )
        print("✓ Analytics imports successful")
        return True
    except Exception as e:
        print(f"✗ Analytics import failed: {e}")
        return False


def test_recommendations_imports():
    """Test that recommendations modules can be imported"""
    try:
        from app.services.recommendations_service import RecommendationsService
        from app.schemas.recommendations import (
            Recommendation,
            PersonalizedCoachingPlan,
            RecommendationsResponse,
            PracticeExercise
        )
        print("✓ Recommendations imports successful")
        return True
    except Exception as e:
        print(f"✗ Recommendations import failed: {e}")
        return False


def test_analytics_api_endpoints():
    """Test that analytics API endpoints are registered"""
    try:
        from app.api.v1.analytics import router
        
        # Check that router has routes
        routes = [route.path for route in router.routes]
        
        expected_routes = [
            "/{recording_id}/detailed",
            "/{recording_id}/summary",
            "/{recording_id}/metrics",
            "/progress",
            "/dashboard/overview",
            "/{recording_id}/recommendations"
        ]
        
        for expected in expected_routes:
            if expected in routes:
                print(f"✓ Route {expected} registered")
            else:
                print(f"✗ Route {expected} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Analytics API test failed: {e}")
        return False


def test_schema_validation():
    """Test that schemas can be instantiated"""
    try:
        from app.schemas.analytics import PostureMetrics, GestureMetrics
        from app.schemas.recommendations import Recommendation, RecommendationPriority, RecommendationCategory
        from uuid import uuid4
        
        # Test PostureMetrics
        posture = PostureMetrics(
            overall_score=75.0,
            head_position_score=80.0,
            shoulder_alignment_score=70.0,
            spine_alignment_score=75.0,
            stability_score=72.0,
            confidence_level=0.85
        )
        assert posture.overall_score == 75.0
        print("✓ PostureMetrics schema validation passed")
        
        # Test Recommendation
        rec = Recommendation(
            id=str(uuid4()),
            category=RecommendationCategory.POSTURE,
            priority=RecommendationPriority.HIGH,
            title="Test Recommendation",
            description="Test description",
            current_score=70.0,
            target_score=85.0,
            improvement_potential=15.0,
            estimated_time_to_improve="2 weeks"
        )
        assert rec.priority == RecommendationPriority.HIGH
        print("✓ Recommendation schema validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False


def test_analytics_service_methods():
    """Test that AnalyticsService has required methods"""
    try:
        from app.services.analytics_service import AnalyticsService
        
        required_methods = [
            'calculate_comprehensive_metrics',
            'calculate_progress_analysis',
            'generate_visualization_data',
            'get_detailed_analytics',
            'get_metrics_summary'
        ]
        
        for method in required_methods:
            if hasattr(AnalyticsService, method):
                print(f"✓ AnalyticsService.{method} exists")
            else:
                print(f"✗ AnalyticsService.{method} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ AnalyticsService methods test failed: {e}")
        return False


def test_recommendations_service_methods():
    """Test that RecommendationsService has required methods"""
    try:
        from app.services.recommendations_service import RecommendationsService
        
        required_methods = [
            'generate_recommendations',
            '_generate_coaching_plan',
            '_generate_all_recommendations',
            '_create_posture_recommendation',
            '_create_gesture_recommendation',
            '_create_speech_quality_recommendation',
            '_create_filler_word_recommendation'
        ]
        
        for method in required_methods:
            if hasattr(RecommendationsService, method):
                print(f"✓ RecommendationsService.{method} exists")
            else:
                print(f"✗ RecommendationsService.{method} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ RecommendationsService methods test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n=== Running Analytics and Recommendations Tests ===\n")
    
    tests = [
        test_analytics_imports,
        test_recommendations_imports,
        test_analytics_api_endpoints,
        test_schema_validation,
        test_analytics_service_methods,
        test_recommendations_service_methods
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)
        print()
    
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)

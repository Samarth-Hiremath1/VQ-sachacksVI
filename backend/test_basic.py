"""Basic test to verify FastAPI application structure"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        from app.main import app
        from app.core.config import settings
        from app.models.user import User
        from app.schemas.user import UserCreate
        from app.services.user_service import UserService
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_app_creation():
    """Test FastAPI app creation"""
    try:
        from app.main import app
        assert app.title == "AI Communication Coaching Platform"
        assert app.version == "1.0.0"
        print("✓ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"✗ App creation failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    try:
        from app.core.config import settings
        assert settings.api_v1_prefix == "/api/v1"
        assert settings.algorithm == "HS256"
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    tests = [test_imports, test_app_creation, test_config]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic structure tests passed!")
    else:
        print("❌ Some tests failed")
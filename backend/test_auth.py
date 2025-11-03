"""Test authentication endpoints"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_register_endpoint_structure():
    """Test register endpoint accepts POST requests"""
    # This should fail with validation error, not 404
    response = client.post("/api/v1/auth/register", json={})
    assert response.status_code == 422  # Validation error, not 404
    print("✓ Register endpoint accessible")


def test_login_endpoint_structure():
    """Test login endpoint accepts POST requests"""
    # This should fail with validation error, not 404
    response = client.post("/api/v1/auth/login")
    assert response.status_code == 422  # Validation error, not 404
    print("✓ Login endpoint accessible")


def test_profile_endpoint_requires_auth():
    """Test profile endpoint requires authentication"""
    response = client.get("/api/v1/users/profile")
    assert response.status_code in [401, 403]  # Unauthorized or Forbidden
    print("✓ Profile endpoint requires authentication")


if __name__ == "__main__":
    tests = [
        test_register_endpoint_structure,
        test_login_endpoint_structure,
        test_profile_endpoint_requires_auth
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All auth tests passed!")
    else:
        print("❌ Some tests failed")
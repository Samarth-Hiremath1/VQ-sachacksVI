"""API endpoint tests"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    print("✓ Root endpoint working")


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health endpoint working")


def test_api_docs():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200
    print("✓ API docs accessible")


def test_openapi_schema():
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    print("✓ OpenAPI schema available")


def test_auth_endpoints_exist():
    """Test that auth endpoints are registered"""
    response = client.get("/openapi.json")
    data = response.json()
    paths = data.get("paths", {})
    
    # Check auth endpoints exist
    assert "/api/v1/auth/register" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/users/profile" in paths
    print("✓ Auth endpoints registered")


if __name__ == "__main__":
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_api_docs,
        test_openapi_schema,
        test_auth_endpoints_exist
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
        print("🎉 All API tests passed!")
    else:
        print("❌ Some tests failed")
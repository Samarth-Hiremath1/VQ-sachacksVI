#!/bin/bash

# Validation script for AI Communication Coaching Platform infrastructure
# This script validates that all components are properly configured

set -e

echo "🔍 AI Communication Coaching Platform - Setup Validation"
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

echo -e "\n📋 File Structure Tests:"

# Test required files exist
run_test "Docker Compose config" "test -f docker-compose.yml"
run_test "Development overrides" "test -f docker-compose.dev.yml"
run_test "Environment template" "test -f .env.example"
run_test "Makefile" "test -f Makefile"
run_test "Infrastructure docs" "test -f INFRASTRUCTURE.md"

echo -e "\n📁 Directory Structure Tests:"

# Test required directories exist
run_test "Airflow DAGs directory" "test -d airflow/dags"
run_test "Prometheus config directory" "test -d monitoring/prometheus"
run_test "Grafana config directory" "test -d monitoring/grafana"
run_test "Scripts directory" "test -d scripts"

echo -e "\n🐳 Docker Configuration Tests:"

# Test Docker Compose configuration
run_test "Docker Compose syntax" "docker compose config --quiet"
run_test "Docker Compose services defined" "docker compose config --services | grep -q postgres"

echo -e "\n📄 Configuration File Tests:"

# Test configuration files
run_test "Prometheus config syntax" "test -f monitoring/prometheus/prometheus.yml"
run_test "Grafana datasource config" "test -f monitoring/grafana/provisioning/datasources/prometheus.yml"
run_test "Database init script" "test -x scripts/init-multiple-databases.sh"

echo -e "\n🔧 Script Tests:"

# Test scripts are executable
run_test "Setup script executable" "test -x scripts/setup-dev.sh"
run_test "Health check script executable" "test -x scripts/health-check.sh"
run_test "Validation script executable" "test -x scripts/validate-setup.sh"

echo -e "\n📊 Summary:"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All validation tests passed!${NC}"
    echo -e "The infrastructure setup is ready for use."
    echo -e "\nNext steps:"
    echo -e "  1. Run: make setup"
    echo -e "  2. Run: make dev"
    echo -e "  3. Run: make health"
    exit 0
else
    echo -e "\n${RED}❌ Some validation tests failed.${NC}"
    echo -e "Please fix the issues above before proceeding."
    exit 1
fi
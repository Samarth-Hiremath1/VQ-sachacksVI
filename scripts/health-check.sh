#!/bin/bash

# Health check script for AI Communication Coaching Platform
# This script checks the health of all services

set -e

echo "🏥 AI Communication Coaching Platform - Health Check"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service_name... "
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓ Healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        return 1
    fi
}

# Function to check port availability
check_port() {
    local service_name=$1
    local host=$2
    local port=$3
    
    echo -n "Checking $service_name port... "
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}✓ Port $port open${NC}"
        return 0
    else
        echo -e "${RED}✗ Port $port closed${NC}"
        return 1
    fi
}

# Check Docker Compose services
echo -e "\n📋 Docker Compose Services:"
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n🔍 Service Health Checks:"

# Database services
check_port "PostgreSQL" "localhost" "5432"
check_port "Redis" "localhost" "6379"

# Storage services
check_service "MinIO API" "http://localhost:9000/minio/health/live"
check_service "MinIO Console" "http://localhost:9001" "200"

# Workflow orchestration
check_service "Airflow Webserver" "http://localhost:8080/health"

# ML Operations
check_service "MLflow" "http://localhost:5000/health"

# Monitoring
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3001/api/health"

echo -e "\n📊 Service URLs:"
echo "  🌐 Airflow:    http://localhost:8080 (admin/admin)"
echo "  🧪 MLflow:     http://localhost:5000"
echo "  📦 MinIO:      http://localhost:9001 (coaching_access_key/coaching_secret_key_12345)"
echo "  📈 Prometheus: http://localhost:9090"
echo "  📊 Grafana:    http://localhost:3001 (admin/admin)"

echo -e "\n✅ Health check completed!"
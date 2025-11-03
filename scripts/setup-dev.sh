#!/bin/bash

# Development environment setup script
# This script initializes the AI Communication Coaching Platform development environment

set -e

echo "🚀 AI Communication Coaching Platform - Development Setup"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

print_success "Docker and Docker Compose are available"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p airflow/logs airflow/plugins
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/dashboards
mkdir -p scripts
print_success "Directory structure created"

# Copy environment file
print_status "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    print_success "Environment file created from template"
else
    print_warning "Environment file already exists, skipping..."
fi

# Make scripts executable
print_status "Setting up scripts..."
chmod +x scripts/*.sh
print_success "Scripts made executable"

# Validate Docker Compose configuration
print_status "Validating Docker Compose configuration..."
if docker compose config --quiet; then
    print_success "Docker Compose configuration is valid"
else
    print_error "Docker Compose configuration is invalid"
    exit 1
fi

# Pull required images
print_status "Pulling Docker images (this may take a while)..."
docker compose pull

print_success "Docker images pulled successfully"

echo ""
echo "🎉 Development environment setup completed!"
echo ""
echo "Next steps:"
echo "  1. Review and customize .env file if needed"
echo "  2. Start services: make up"
echo "  3. Check health: make health"
echo "  4. Access services:"
echo "     - Airflow: http://localhost:8080 (admin/admin)"
echo "     - MLflow: http://localhost:5000"
echo "     - MinIO: http://localhost:9001"
echo "     - Prometheus: http://localhost:9090"
echo "     - Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "For more information, see INFRASTRUCTURE.md"
.PHONY: help build up down restart logs clean status health

# Default target
help:
	@echo "AI Communication Coaching Platform - Docker Management"
	@echo ""
	@echo "Available commands:"
	@echo "  setup     - Run development environment setup script"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services (production mode)"
	@echo "  down      - Stop all services"
	@echo "  dev-up    - Start services in development mode"
	@echo "  dev-down  - Stop development services"
	@echo "  dev       - Quick development setup (init + dev-up)"
	@echo "  restart   - Restart all services"
	@echo "  logs      - Show logs for all services"
	@echo "  clean     - Remove all containers, networks, and volumes"
	@echo "  status    - Show status of all services"
	@echo "  health    - Check health of all services"
	@echo "  init      - Initialize the development environment"

# Build all images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# Restart all services
restart: down up

# Show logs
logs:
	docker compose logs -f

# Clean everything
clean:
	docker compose down -v --remove-orphans
	docker system prune -f

# Show service status
status:
	docker compose ps

# Health check
health:
	@echo "Checking service health..."
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Initialize development environment
init:
	@echo "Initializing AI Communication Coaching Platform..."
	@cp .env.example .env 2>/dev/null || true
	@echo "Environment file created/updated"
	@mkdir -p airflow/logs airflow/plugins
	@mkdir -p monitoring/prometheus/rules
	@mkdir -p monitoring/grafana/dashboards
	@echo "Directory structure created"
	@echo "Run 'make up' to start all services"

# Development environment with overrides
dev-up:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Development environment down
dev-down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# Quick development setup
dev: init dev-up
	@echo "Development environment is starting..."
	@echo "Services will be available at:"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Backend API: http://localhost:8000"
	@echo "  - Airflow: http://localhost:8080 (admin/admin)"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - MinIO Console: http://localhost:9001"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3001 (admin/admin)"

# Setup development environment
setup:
	./scripts/setup-dev.sh
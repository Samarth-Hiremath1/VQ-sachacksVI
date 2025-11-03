# AI Communication Coaching Platform

## Quick Start

```bash
# Setup development environment
make setup

# Start all services
make dev

# Check service health
make health

# Stop services
make dev-down
```

## Services

- Frontend: http://localhost:3000
- Airflow: http://localhost:8080 (admin/admin)
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)
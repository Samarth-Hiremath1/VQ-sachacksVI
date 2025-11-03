# AI Communication Coaching Platform - Infrastructure Setup

This document describes the containerized infrastructure setup for the AI Communication Coaching Platform.

## Architecture Overview

The platform uses a microservices architecture with the following components:

- **PostgreSQL**: Primary database for application data, Airflow metadata, and MLflow tracking
- **Redis**: Caching and session management
- **MinIO**: S3-compatible object storage for recordings and ML artifacts
- **Apache Airflow**: Workflow orchestration for ML pipelines
- **MLflow**: ML experiment tracking and model registry
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting dashboards

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for containers
- Ports 3000, 5000, 6379, 8000, 8080, 9000, 9001, 9090, 3001 available

### Initial Setup

1. **Initialize the environment:**
   ```bash
   make init
   ```

2. **Start all services:**
   ```bash
   make up
   ```

3. **Check service status:**
   ```bash
   make status
   make health
   ```

### Service Access

Once all services are running, you can access:

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow Web UI | http://localhost:8080 | admin/admin |
| MLflow Tracking | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | coaching_access_key/coaching_secret_key_12345 |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3001 | admin/admin |

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

Key configuration options:

- **Database**: PostgreSQL connection settings
- **Storage**: MinIO S3-compatible storage credentials
- **Monitoring**: Prometheus and Grafana ports and credentials
- **Security**: JWT secrets and encryption keys (change for production!)

### Service Configuration

#### PostgreSQL
- **Database**: `coaching_platform` (main application)
- **Additional DBs**: `airflow`, `mlflow` (automatically created)
- **Port**: 5432
- **Volume**: `postgres_data`

#### Redis
- **Port**: 6379
- **Persistence**: AOF enabled
- **Volume**: `redis_data`

#### MinIO
- **API Port**: 9000
- **Console Port**: 9001
- **Buckets**: `recordings`, `mlflow-artifacts`, `processed-data` (auto-created)
- **Volume**: `minio_data`

#### Airflow
- **Webserver Port**: 8080
- **Executor**: LocalExecutor
- **DAGs**: `./airflow/dags`
- **Logs**: `./airflow/logs`

#### MLflow
- **Port**: 5000
- **Backend Store**: PostgreSQL
- **Artifact Store**: MinIO S3
- **Tracking URI**: http://localhost:5000

#### Prometheus
- **Port**: 9090
- **Config**: `./monitoring/prometheus/prometheus.yml`
- **Rules**: `./monitoring/prometheus/rules/`

#### Grafana
- **Port**: 3001
- **Dashboards**: `./monitoring/grafana/dashboards/`
- **Provisioning**: `./monitoring/grafana/provisioning/`

## Development Workflow

### Starting Development

```bash
# Full development setup
make dev

# Or step by step
make init
make up
make logs  # Monitor startup
```

### Managing Services

```bash
# View service status
make status

# View logs
make logs

# Restart services
make restart

# Stop services
make down

# Clean everything (removes volumes!)
make clean
```

### Health Monitoring

The infrastructure includes comprehensive health checks:

```bash
# Check all service health
make health

# Individual service health
docker-compose ps
```

### Troubleshooting

#### Common Issues

1. **Port Conflicts**: Ensure all required ports are available
2. **Memory Issues**: Increase Docker memory allocation to 8GB+
3. **Permission Issues**: Ensure Docker has proper permissions

#### Service Dependencies

Services start in the following order:
1. PostgreSQL, Redis, MinIO
2. MLflow (depends on PostgreSQL, MinIO)
3. Airflow (depends on PostgreSQL)
4. Prometheus, Grafana (depends on other services for metrics)

#### Logs and Debugging

```bash
# View logs for specific service
docker-compose logs -f postgres
docker-compose logs -f airflow-webserver
docker-compose logs -f mlflow

# View all logs
make logs
```

## Production Considerations

### Security

- [ ] Change all default passwords and secrets
- [ ] Use proper SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Enable authentication for all services
- [ ] Use secrets management (Docker Secrets, Kubernetes Secrets)

### Scalability

- [ ] Configure horizontal scaling for ML services
- [ ] Set up database connection pooling
- [ ] Implement load balancing
- [ ] Configure auto-scaling policies

### Monitoring

- [ ] Set up alerting rules in Prometheus
- [ ] Configure notification channels in Grafana
- [ ] Implement log aggregation (ELK stack)
- [ ] Set up distributed tracing

### Backup and Recovery

- [ ] Configure automated database backups
- [ ] Set up MinIO backup policies
- [ ] Implement disaster recovery procedures
- [ ] Test backup restoration

## Network Architecture

The platform uses a custom Docker network (`coaching-network`) with subnet `172.20.0.0/16` for service communication.

### Service Communication

- All services communicate through the internal Docker network
- External access is provided through exposed ports
- Health checks ensure service availability
- Circuit breakers prevent cascade failures

## Volume Management

Persistent data is stored in Docker volumes:

- `postgres_data`: Database files
- `redis_data`: Redis persistence
- `minio_data`: Object storage
- `airflow_data`: Airflow metadata
- `prometheus_data`: Metrics data
- `grafana_data`: Dashboard configurations

### Backup Volumes

```bash
# Backup all volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volume
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

## Next Steps

After the infrastructure is running:

1. **Backend Development**: Implement FastAPI services
2. **ML Services**: Deploy PyTorch and TensorFlow models
3. **Frontend Integration**: Connect React app to backend
4. **Pipeline Development**: Create Airflow DAGs for ML workflows
5. **Monitoring Setup**: Configure custom metrics and alerts

For implementation details, refer to the main project documentation and task specifications.
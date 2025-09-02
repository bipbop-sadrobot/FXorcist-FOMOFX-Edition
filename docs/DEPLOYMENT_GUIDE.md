# FXorcist Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the FXorcist AI Dashboard system in various environments including local development, Docker containers, and cloud platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration Management](#configuration-management)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 50GB free space
- **Python**: 3.8+

#### Recommended Requirements
- **OS**: Ubuntu 20.04+ or macOS 12+
- **RAM**: 32GB+
- **CPU**: 8+ cores with AVX2 support
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with CUDA 11.0+ (optional)

### Software Dependencies

#### Required Packages
```bash
# Python packages
pip install streamlit pandas numpy scikit-learn catboost lightgbm xgboost
pip install psutil plotly matplotlib seaborn
pip install docker kubernetes awscli gcloud azure-cli
```

#### System Packages (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3-dev build-essential git curl wget unzip
sudo apt install docker.io docker-compose
```

#### System Packages (macOS)
```bash
brew install python3 git curl wget unzip
brew install --cask docker
```

## Local Deployment

### Quick Start

1. **Clone Repository**
```bash
git clone https://github.com/your-org/FXorcist-FOMOFX-Edition.git
cd FXorcist-FOMOFX-Edition
```

2. **Setup Environment**
```bash
# Create virtual environment
python3 -m venv fxorcist_env
source fxorcist_env/bin/activate  # Linux/macOS
# fxorcist_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

3. **Configure System**
```bash
# Run setup script
python setup.py

# Or use CLI
python fxorcist_cli.py --command "setup"
```

4. **Start System**
```bash
# Start main dashboard
python fxorcist_cli.py --dashboard

# Or start training dashboard
python fxorcist_cli.py --training-dashboard
```

### Advanced Local Setup

#### Custom Configuration
```bash
# Edit configuration files
nano config/cli_config.json
nano config/pipeline_config.json
nano config/training_config.json
```

#### Environment Variables
```bash
export FXORCIST_ENV=development
export FXORCIST_LOG_LEVEL=DEBUG
export FXORCIST_DATA_DIR=./data
export FXORCIST_MODEL_DIR=./models
export STREAMLIT_SERVER_PORT=8501
```

#### Data Setup
```bash
# Download sample data
python fxorcist_cli.py --command "data download"

# Process data
python fxorcist_cli.py --command "data process"

# Validate data quality
python fxorcist_cli.py --command "data validate"
```

## Docker Deployment

### Single Container Deployment

1. **Build Docker Image**
```bash
# Build image
docker build -t fxorcist:latest .

# Or use build script
./scripts/build_docker.sh
```

2. **Run Container**
```bash
# Basic run
docker run -p 8501:8501 fxorcist:latest

# With volume mounts
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fxorcist:latest
```

3. **Environment Configuration**
```bash
docker run -p 8501:8501 \
  -e FXORCIST_ENV=production \
  -e FXORCIST_LOG_LEVEL=INFO \
  fxorcist:latest
```

### Docker Compose Deployment

1. **Create docker-compose.yml**
```yaml
version: '3.8'
services:
  fxorcist:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - FXORCIST_ENV=production
      - FXORCIST_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8501/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

2. **Deploy with Compose**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Multi-Container Setup

#### docker-compose.prod.yml
```yaml
version: '3.8'
services:
  fxorcist-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - fxorcist_data:/app/data
      - fxorcist_models:/app/models
      - fxorcist_logs:/app/logs
    environment:
      - FXORCIST_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/fxorcist
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=fxorcist
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  fxorcist_data:
  fxorcist_models:
  fxorcist_logs:
  postgres_data:
  redis_data:
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

1. **Launch EC2 Instance**
```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --count 1 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-groups fxorcist-sg
```

2. **Configure Security Group**
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345678 \
  --protocol tcp \
  --port 8501 \
  --cidr 0.0.0.0/0
```

3. **Deploy Application**
```bash
# Connect to instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 git -y

# Clone and setup
git clone https://github.com/your-org/FXorcist-FOMOFX-Edition.git
cd FXorcist-FOMOFX-Edition
python3 -m venv fxorcist_env
source fxorcist_env/bin/activate
pip install -r requirements.txt

# Start application
python fxorcist_cli.py --dashboard &
```

#### ECS Fargate Deployment

1. **Create Task Definition**
```json
{
  "family": "fxorcist-task",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "fxorcist",
      "image": "your-registry/fxorcist:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "hostPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "FXORCIST_ENV", "value": "production"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fxorcist",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

2. **Deploy to ECS**
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster fxorcist-cluster \
  --service-name fxorcist-service \
  --task-definition fxorcist-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-67890]}"
```

### Google Cloud Platform

#### App Engine Deployment

1. **Create app.yaml**
```yaml
runtime: python38
instance_class: F4
automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 1
  max_instances: 10

handlers:
- url: /.*
  script: auto
  secure: always

env_variables:
  FXORCIST_ENV: production
  FXORCIST_LOG_LEVEL: INFO
```

2. **Deploy to App Engine**
```bash
# Deploy
gcloud app deploy

# View logs
gcloud app logs tail -s default
```

#### Cloud Run Deployment

```bash
# Build and push container
gcloud builds submit --tag gcr.io/project-id/fxorcist

# Deploy to Cloud Run
gcloud run deploy fxorcist \
  --image gcr.io/project-id/fxorcist \
  --platform managed \
  --port 8501 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Microsoft Azure

#### Web App Deployment

1. **Create Web App**
```bash
az webapp create \
  --resource-group fxorcist-rg \
  --plan fxorcist-plan \
  --name fxorcist-app \
  --runtime "PYTHON:3.8"
```

2. **Deploy Application**
```bash
# Deploy via ZIP
az webapp deployment source config-zip \
  --resource-group fxorcist-rg \
  --name fxorcist-app \
  --src fxorcist-deployment.zip
```

#### Container Instance Deployment

```bash
az container create \
  --resource-group fxorcist-rg \
  --name fxorcist-container \
  --image your-registry/fxorcist:latest \
  --ports 8501 \
  --cpu 2 \
  --memory 4 \
  --environment-variables FXORCIST_ENV=production
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
```

### Basic Kubernetes Deployment

1. **Create Namespace**
```bash
kubectl create namespace fxorcist
```

2. **Apply Manifests**
```bash
# Deploy application
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# Check deployment
kubectl get pods -n fxorcist
kubectl get services -n fxorcist
```

3. **Access Application**
```bash
# Port forward
kubectl port-forward -n fxorcist svc/fxorcist-service 8501:8501

# Or get external IP
kubectl get svc fxorcist-service -n fxorcist
```

### Helm Chart Deployment

1. **Install via Helm**
```bash
# Add repository
helm repo add fxorcist https://charts.fxorcist.com
helm repo update

# Install chart
helm install fxorcist fxorcist/fxorcist \
  --namespace fxorcist \
  --create-namespace \
  --set image.tag=latest
```

2. **Custom Values**
```yaml
# values.yaml
image:
  repository: your-registry/fxorcist
  tag: latest

service:
  type: LoadBalancer
  port: 8501

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

config:
  env: production
  logLevel: INFO
```

### Advanced Kubernetes Features

#### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fxorcist-hpa
  namespace: fxorcist
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fxorcist
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fxorcist-ingress
  namespace: fxorcist
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: fxorcist.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fxorcist-service
            port:
              number: 8501
```

## Configuration Management

### Environment-Specific Configurations

#### Development Configuration
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "data_dir": "./data",
  "model_dir": "./models",
  "cache_enabled": false,
  "batch_size": 100
}
```

#### Production Configuration
```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "data_dir": "/app/data",
  "model_dir": "/app/models",
  "cache_enabled": true,
  "batch_size": 1000
}
```

### Secrets Management

#### Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: fxorcist-secrets
  namespace: fxorcist
type: Opaque
data:
  database-url: <base64-encoded-url>
  api-key: <base64-encoded-key>
```

#### AWS Secrets Manager
```bash
# Store secret
aws secretsmanager create-secret \
  --name fxorcist/database \
  --secret-string '{"username":"admin","password":"secret"}'

# Retrieve in application
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='fxorcist/database')
```

## Monitoring and Maintenance

### Health Checks

#### Application Health Endpoints
```python
# Health check endpoint
@app.route('/health')
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    }
```

#### System Monitoring
```bash
# Check system resources
python fxorcist_cli.py --command "health system"

# Monitor application
python fxorcist_cli.py --command "health monitor"

# Generate health report
python fxorcist_cli.py --command "health report"
```

### Logging

#### Log Configuration
```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
  },
  "handlers": {
    "file": {
      "class": "logging.FileHandler",
      "filename": "logs/fxorcist.log",
      "formatter": "standard"
    },
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "standard"
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console", "file"]
  }
}
```

#### Log Rotation
```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/fxorcist.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

### Backup and Recovery

#### Automated Backups
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup data
cp -r /app/data $BACKUP_DIR/
cp -r /app/models $BACKUP_DIR/

# Backup configuration
cp -r /app/config $BACKUP_DIR/

# Compress backup
tar -czf ${BACKUP_DIR}.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

#### Database Backups
```bash
# PostgreSQL backup
pg_dump -h db-host -U username -d fxorcist > backup.sql

# Restore
psql -h db-host -U username -d fxorcist < backup.sql
```

### Performance Optimization

#### Resource Tuning
```yaml
# Kubernetes resource limits
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Database Optimization
```sql
-- Create indexes
CREATE INDEX idx_timestamp ON forex_data (timestamp);
CREATE INDEX idx_symbol ON forex_data (symbol);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM forex_data
WHERE symbol = 'EURUSD'
AND timestamp BETWEEN '2023-01-01' AND '2023-12-31';
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
export STREAMLIT_SERVER_PORT=8502
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Monitor process memory
ps aux --sort=-%mem | head

# Reduce batch size in config
{
  "batch_size": 500,
  "memory_efficient": true
}
```

#### Permission Issues
```bash
# Fix directory permissions
chmod -R 755 /app/data
chmod -R 755 /app/models

# Change ownership
chown -R fxorcist:fxorcist /app
```

#### Network Issues
```bash
# Test connectivity
curl -I http://localhost:8501

# Check firewall
sudo ufw status
sudo ufw allow 8501
```

### Debug Mode

#### Enable Debug Logging
```bash
export FXORCIST_LOG_LEVEL=DEBUG
export STREAMLIT_LOG_LEVEL=DEBUG
```

#### Debug Commands
```bash
# Run with debug
python -m pdb fxorcist_cli.py --dashboard

# Check system info
python fxorcist_cli.py --command "system info"

# Validate configuration
python fxorcist_cli.py --command "config validate"
```

### Support Resources

#### Getting Help
- **Documentation**: https://docs.fxorcist.com
- **GitHub Issues**: https://github.com/your-org/FXorcist-FOMOFX-Edition/issues
- **Community Forum**: https://community.fxorcist.com
- **Professional Support**: support@fxorcist.com

#### Diagnostic Information
```bash
# Generate diagnostic report
python fxorcist_cli.py --command "diagnostics"

# System information
python fxorcist_cli.py --command "system info"

# Log analysis
python fxorcist_cli.py --command "logs analyze"
```

---

*Deployment Guide Version: 2.0*
*Last Updated: September 2, 2025*
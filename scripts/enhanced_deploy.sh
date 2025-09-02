#!/bin/bash

# Enhanced Deployment Script for FXorcist AI Dashboard
# Version: 2.0
# Date: September 2, 2025
# Description: Advanced multi-environment deployment with CI/CD integration

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
LOGS_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/deploy_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

# Create necessary directories
setup_directories() {
    mkdir -p "$LOGS_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$PROJECT_ROOT/backups"
    mkdir -p "$PROJECT_ROOT/temp"
}

# Load configuration
load_config() {
    local env=${1:-production}

    if [[ -f "$CONFIG_DIR/deploy_$env.json" ]]; then
        CONFIG_FILE="$CONFIG_DIR/deploy_$env.json"
    elif [[ -f "$CONFIG_DIR/deploy.json" ]]; then
        CONFIG_FILE="$CONFIG_DIR/deploy.json"
    else
        log_error "Configuration file not found for environment: $env"
        return 1
    fi

    # Load JSON configuration
    if command -v jq &> /dev/null; then
        DEPLOY_ENV=$(jq -r '.environment // "production"' "$CONFIG_FILE")
        DOCKER_IMAGE=$(jq -r '.docker.image // "fxorcist:latest"' "$CONFIG_FILE")
        DOCKER_REGISTRY=$(jq -r '.docker.registry // ""' "$CONFIG_FILE")
        K8S_NAMESPACE=$(jq -r '.kubernetes.namespace // "fxorcist"' "$CONFIG_FILE")
        AWS_REGION=$(jq -r '.aws.region // "us-east-1"' "$CONFIG_FILE")
        GCP_PROJECT=$(jq -r '.gcp.project // ""' "$CONFIG_FILE")
        AZURE_SUBSCRIPTION=$(jq -r '.azure.subscription // ""' "$CONFIG_FILE")
    else
        log_warn "jq not found, using default configuration"
        DEPLOY_ENV=${FXORCIST_ENV:-production}
        DOCKER_IMAGE=${DOCKER_IMAGE:-fxorcist:latest}
        DOCKER_REGISTRY=${DOCKER_REGISTRY:-}
        K8S_NAMESPACE=${K8S_NAMESPACE:-fxorcist}
        AWS_REGION=${AWS_REGION:-us-east-1}
        GCP_PROJECT=${GCP_PROJECT:-}
        AZURE_SUBSCRIPTION=${AZURE_SUBSCRIPTION:-}
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check required tools
    local required_tools=("docker" "git" "python3")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool not found: $tool"
            return 1
        fi
    done

    # Check Python dependencies
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log_info "Checking Python dependencies..."
        if ! python3 -c "import sys; sys.path.append('$PROJECT_ROOT'); import forex_ai_dashboard" 2>/dev/null; then
            log_warn "Python dependencies may not be installed"
        fi
    fi

    # Check disk space
    local available_space
    available_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    if (( available_space < 1048576 )); then  # Less than 1GB
        log_error "Insufficient disk space: ${available_space}KB available"
        return 1
    fi

    # Check network connectivity
    if ! ping -c 1 google.com &> /dev/null; then
        log_warn "No internet connectivity detected"
    fi

    log_success "Pre-deployment checks completed"
}

# Build Docker image
build_docker_image() {
    local tag=${1:-latest}
    local no_cache=${2:-false}

    log_info "Building Docker image: $DOCKER_IMAGE:$tag"

    local build_args=""
    if [[ "$no_cache" == "true" ]]; then
        build_args="--no-cache"
    fi

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local full_image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local full_image="$DOCKER_IMAGE:$tag"
    fi

    # Build image
    if ! docker build $build_args -t "$full_image" "$PROJECT_ROOT"; then
        log_error "Failed to build Docker image"
        return 1
    fi

    # Push to registry if specified
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        log_info "Pushing image to registry: $full_image"
        if ! docker push "$full_image"; then
            log_error "Failed to push Docker image to registry"
            return 1
        fi
    fi

    log_success "Docker image built and pushed: $full_image"
}

# Deploy to local environment
deploy_local() {
    log_info "Deploying to local environment..."

    # Create deployment directory
    local deploy_dir="$PROJECT_ROOT/deployments/local_$TIMESTAMP"
    mkdir -p "$deploy_dir"

    # Copy application files
    cp -r "$PROJECT_ROOT"/* "$deploy_dir/" 2>/dev/null || true

    # Create virtual environment
    cd "$deploy_dir"
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi

    # Create systemd service file
    cat > fxorcist.service << EOF
[Unit]
Description=FXorcist AI Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$deploy_dir
Environment=PATH=$deploy_dir/venv/bin
ExecStart=$deploy_dir/venv/bin/python -m streamlit run dashboard/app.py --server.port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Install and start service
    sudo cp fxorcist.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable fxorcist
    sudo systemctl start fxorcist

    log_success "Local deployment completed"
    log_info "Dashboard available at: http://localhost:8501"
}

# Deploy to Docker
deploy_docker() {
    local tag=${1:-latest}
    local replicas=${2:-1}

    log_info "Deploying to Docker environment..."

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local image="$DOCKER_IMAGE:$tag"
    fi

    # Create docker-compose file
    cat > docker-compose.yml << EOF
version: '3.8'
services:
  fxorcist:
    image: $image
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - FXORCIST_ENV=$DEPLOY_ENV
      - FXORCIST_LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8501/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: $replicas
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 512M

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
EOF

    # Deploy with docker-compose
    docker-compose up -d

    log_success "Docker deployment completed"
    log_info "Dashboard available at: http://localhost:8501"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    local tag=${1:-latest}

    log_info "Deploying to Kubernetes cluster..."

    # Check kubectl access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "No access to Kubernetes cluster"
        return 1
    fi

    # Create namespace
    kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local image="$DOCKER_IMAGE:$tag"
    fi

    # Create deployment
    cat > k8s-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fxorcist
  namespace: $K8S_NAMESPACE
  labels:
    app: fxorcist
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fxorcist
  template:
    metadata:
      labels:
        app: fxorcist
    spec:
      containers:
      - name: fxorcist
        image: $image
        ports:
        - containerPort: 8501
        env:
        - name: FXORCIST_ENV
          value: "$DEPLOY_ENV"
        - name: FXORCIST_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: fxorcist-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: fxorcist-models-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: fxorcist-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: fxorcist-service
  namespace: $K8S_NAMESPACE
spec:
  selector:
    app: fxorcist
  ports:
  - port: 8501
    targetPort: 8501
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fxorcist-data-pvc
  namespace: $K8S_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fxorcist-models-pvc
  namespace: $K8S_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fxorcist-logs-pvc
  namespace: $K8S_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

    # Apply Kubernetes manifests
    kubectl apply -f k8s-deployment.yaml

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/fxorcist -n "$K8S_NAMESPACE"

    # Get service URL
    local service_url
    service_url=$(kubectl get svc fxorcist-service -n "$K8S_NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

    log_success "Kubernetes deployment completed"
    log_info "Dashboard available at: http://$service_url:8501"
}

# Deploy to AWS
deploy_aws() {
    local tag=${1:-latest}

    log_info "Deploying to AWS environment..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found"
        return 1
    fi

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local image="$DOCKER_IMAGE:$tag"
    fi

    # Create ECS cluster (if it doesn't exist)
    aws ecs create-cluster --cluster-name fxorcist-cluster --region "$AWS_REGION" || true

    # Create task definition
    cat > task-definition.json << EOF
{
  "family": "fxorcist-task",
  "taskRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
  "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "fxorcist",
      "image": "$image",
      "portMappings": [
        {
          "containerPort": 8501,
          "hostPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "FXORCIST_ENV", "value": "$DEPLOY_ENV"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fxorcist",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "essential": true
    }
  ]
}
EOF

    # Register task definition
    aws ecs register-task-definition --cli-input-json file://task-definition.json --region "$AWS_REGION"

    # Create service
    aws ecs create-service \
      --cluster fxorcist-cluster \
      --service-name fxorcist-service \
      --task-definition fxorcist-task \
      --desired-count 1 \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-67890]}" \
      --region "$AWS_REGION" || true

    log_success "AWS deployment initiated"
    log_info "Monitor deployment status with: aws ecs describe-services --cluster fxorcist-cluster --services fxorcist-service --region $AWS_REGION"
}

# Deploy to Google Cloud
deploy_gcp() {
    local tag=${1:-latest}

    log_info "Deploying to Google Cloud Platform..."

    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found"
        return 1
    fi

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local image="gcr.io/$GCP_PROJECT/fxorcist:$tag"
        # Build and push to GCR
        gcloud builds submit --tag "$image" "$PROJECT_ROOT"
    fi

    # Deploy to Cloud Run
    gcloud run deploy fxorcist \
      --image "$image" \
      --platform managed \
      --port 8501 \
      --memory 4Gi \
      --cpu 2 \
      --max-instances 10 \
      --allow-unauthenticated \
      --set-env-vars "FXORCIST_ENV=$DEPLOY_ENV"

    log_success "GCP deployment completed"
    log_info "Dashboard URL: $(gcloud run services describe fxorcist --format 'value(status.url)')"
}

# Deploy to Azure
deploy_azure() {
    local tag=${1:-latest}

    log_info "Deploying to Microsoft Azure..."

    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not found"
        return 1
    fi

    if [[ -n "$DOCKER_REGISTRY" ]]; then
        local image="$DOCKER_REGISTRY/$DOCKER_IMAGE:$tag"
    else
        local image="$DOCKER_IMAGE:$tag"
    fi

    # Create resource group
    az group create --name fxorcist-rg --location eastus

    # Create container instance
    az container create \
      --resource-group fxorcist-rg \
      --name fxorcist-container \
      --image "$image" \
      --ports 8501 \
      --cpu 2 \
      --memory 4 \
      --environment-variables FXORCIST_ENV="$DEPLOY_ENV" \
      --ip-address public

    log_success "Azure deployment completed"
    log_info "Dashboard URL: $(az container show --resource-group fxorcist-rg --name fxorcist-container --query ipAddress.fqdn -o tsv):8501"
}

# CI/CD deployment
deploy_ci() {
    local branch=${1:-main}
    local tag=${2:-latest}

    log_info "Running CI/CD deployment for branch: $branch"

    # Run tests
    if [[ -f "$PROJECT_ROOT/test.sh" ]]; then
        log_info "Running tests..."
        if ! bash "$PROJECT_ROOT/test.sh"; then
            log_error "Tests failed, aborting deployment"
            return 1
        fi
    fi

    # Build and deploy
    build_docker_image "$tag" true

    case $DEPLOY_ENV in
        development)
            deploy_docker "$tag" 1
            ;;
        staging)
            deploy_kubernetes "$tag"
            ;;
        production)
            deploy_kubernetes "$tag"
            ;;
        *)
            log_error "Unknown deployment environment: $DEPLOY_ENV"
            return 1
            ;;
    esac

    log_success "CI/CD deployment completed"
}

# Rollback deployment
rollback_deployment() {
    local target=${1:-previous}

    log_info "Rolling back deployment to: $target"

    case $target in
        previous)
            # Rollback to previous version
            if [[ -f "docker-compose.yml" ]]; then
                docker-compose down
                # Restore previous image tag
                sed -i 's/fxorcist:latest/fxorcist:previous/g' docker-compose.yml
                docker-compose up -d
            elif command -v kubectl &> /dev/null; then
                kubectl rollout undo deployment/fxorcist -n "$K8S_NAMESPACE"
            fi
            ;;
        stable)
            # Rollback to stable version
            log_info "Rolling back to stable version..."
            ;;
        *)
            log_error "Unknown rollback target: $target"
            return 1
            ;;
    esac

    log_success "Rollback completed"
}

# Health check after deployment
post_deployment_health_check() {
    local url=${1:-http://localhost:8501/health}
    local timeout=${2:-60}

    log_info "Running post-deployment health checks..."

    local start_time=$SECONDS
    while (( SECONDS - start_time < timeout )); do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        sleep 5
    done

    log_error "Health check failed after $timeout seconds"
    return 1
}

# Main deployment function
main() {
    local action=${1:-help}
    local environment=${2:-production}
    local tag=${3:-latest}

    # Setup
    setup_directories
    load_config "$environment"

    log_info "FXorcist Enhanced Deployment Script v2.0"
    log_info "Environment: $environment"
    log_info "Action: $action"

    case $action in
        local)
            pre_deployment_checks
            deploy_local
            post_deployment_health_check
            ;;
        docker)
            pre_deployment_checks
            build_docker_image "$tag"
            deploy_docker "$tag"
            post_deployment_health_check
            ;;
        kubernetes|k8s)
            pre_deployment_checks
            build_docker_image "$tag"
            deploy_kubernetes "$tag"
            post_deployment_health_check
            ;;
        aws)
            pre_deployment_checks
            build_docker_image "$tag"
            deploy_aws "$tag"
            ;;
        gcp)
            pre_deployment_checks
            deploy_gcp "$tag"
            ;;
        azure)
            pre_deployment_checks
            deploy_azure "$tag"
            ;;
        ci)
            deploy_ci "${2:-main}" "$tag"
            ;;
        rollback)
            rollback_deployment "${2:-previous}"
            ;;
        build)
            build_docker_image "$tag"
            ;;
        check)
            pre_deployment_checks
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
}

# Show help
show_help() {
    cat << EOF
FXorcist Enhanced Deployment Script v2.0

USAGE:
    $0 <action> [environment] [tag]

ACTIONS:
    local       Deploy to local system
    docker      Deploy using Docker Compose
    kubernetes  Deploy to Kubernetes cluster
    aws         Deploy to Amazon Web Services
    gcp         Deploy to Google Cloud Platform
    azure       Deploy to Microsoft Azure
    ci          CI/CD deployment pipeline
    build       Build Docker image only
    check       Run pre-deployment checks
    rollback    Rollback deployment
    help        Show this help message

ENVIRONMENTS:
    development Local development environment
    staging     Staging environment
    production  Production environment

EXAMPLES:
    $0 docker development latest
    $0 kubernetes production v1.2.3
    $0 ci main latest
    $0 rollback previous

CONFIGURATION:
    Create config/deploy_<environment>.json for environment-specific settings

LOGGING:
    Deployment logs are saved to: $LOGS_DIR/deploy_*.log
EOF
}

# Run main function with all arguments
main "$@"
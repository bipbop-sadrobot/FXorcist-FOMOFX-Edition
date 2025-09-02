#!/bin/bash
# FXorcist Automated Deployment Script
# Supports multiple deployment environments and strategies

set -e  # Exit on any error

# Configuration
PROJECT_NAME="fxorcist-fomofx-edition"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_ENV="${1:-development}"
DEPLOY_TYPE="${2:-local}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/fxorcist_cli.py" ]]; then
        log_error "Not in FXorcist project directory"
        exit 1
    fi

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.8" | sort -V | head -n1)" != "3.8" ]]; then
        log_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi

    # Check for required files
    local required_files=("requirements.txt" "fxorcist_cli.py" "setup.py")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done

    log_success "Pre-deployment checks passed"
}

# Setup environment
setup_environment() {
    local env="$1"
    log_info "Setting up environment: $env"

    # Create environment-specific directories
    mkdir -p "$PROJECT_ROOT/config/environments"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/data"

    # Copy environment configuration
    if [[ -f "$PROJECT_ROOT/config/environments/${env}.json" ]]; then
        cp "$PROJECT_ROOT/config/environments/${env}.json" "$PROJECT_ROOT/config/current_env.json"
        log_info "Loaded environment configuration: $env"
    else
        log_warning "Environment configuration not found: $env.json"
        # Create default configuration
        cat > "$PROJECT_ROOT/config/environments/${env}.json" << EOF
{
  "environment": "$env",
  "debug": $([[ "$env" == "development" ]] && echo "true" || echo "false"),
  "log_level": $([[ "$env" == "production" ]] && echo "\"INFO\"" || echo "\"DEBUG\""),
  "dashboard_port": 8501,
  "training_port": 8502,
  "memory_port": 8503,
  "auto_start_dashboards": $([[ "$env" == "production" ]] && echo "true" || echo "false")
}
EOF
        log_info "Created default environment configuration: $env"
    fi
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."

    cd "$PROJECT_ROOT"

    # Upgrade pip
    python3 -m pip install --upgrade pip

    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        python3 -m pip install -r requirements.txt
        log_success "Dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi

    # Install project in development mode
    python3 -m pip install -e .
    log_success "Project installed in development mode"
}

# Setup data directories
setup_data_directories() {
    log_info "Setting up data directories..."

    cd "$PROJECT_ROOT"

    # Create data subdirectories
    local data_dirs=("raw" "processed" "temp_extracted")
    for dir in "${data_dirs[@]}"; do
        mkdir -p "data/$dir"
    done

    # Create forex pair directories
    local pairs=("eurusd" "gbpusd" "usdchf" "usdjpy" "audusd" "usdcad" "nzdusd")
    for pair in "${pairs[@]}"; do
        mkdir -p "data/raw/$pair"
        mkdir -p "data/processed/$pair"
    done

    log_success "Data directories created"
}

# Deploy locally
deploy_local() {
    log_info "Deploying locally..."

    cd "$PROJECT_ROOT"

    # Run setup script
    if [[ -f "scripts/setup_fxorcist.py" ]]; then
        python3 scripts/setup_fxorcist.py
    fi

    # Initialize project
    python3 fxorcist_cli.py --command setup

    # Start dashboards based on environment
    if [[ "$DEPLOY_ENV" == "production" ]]; then
        log_info "Starting production dashboards..."
        python3 scripts/dashboard_launcher.py start-all
    else
        log_info "Starting main dashboard for development..."
        python3 scripts/dashboard_launcher.py start main
    fi

    log_success "Local deployment completed"
    echo ""
    echo "🚀 FXorcist is now running!"
    echo ""
    if [[ "$DEPLOY_ENV" == "production" ]]; then
        echo "📊 Main Dashboard:    http://localhost:8501"
        echo "🎯 Training Dashboard: http://localhost:8502"
        echo "🧠 Memory Dashboard:   http://localhost:8503"
    else
        echo "📊 Main Dashboard:    http://localhost:8501"
    fi
    echo ""
    echo "To stop: python3 scripts/dashboard_launcher.py stop-all"
}

# Deploy with Docker
deploy_docker() {
    log_info "Deploying with Docker..."

    cd "$PROJECT_ROOT"

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi

    # Build Docker image
    log_info "Building Docker image..."
    docker build -t fxorcist:$DEPLOY_ENV .

    # Create docker-compose file if it doesn't exist
    if [[ ! -f "docker-compose.yml" ]]; then
        cat > docker-compose.yml << EOF
version: '3.8'
services:
  fxorcist:
    build: .
    ports:
      - "8501:8501"
      - "8502:8502"
      - "8503:8503"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DEPLOY_ENV=$DEPLOY_ENV
    restart: unless-stopped
EOF
        log_info "Created docker-compose.yml"
    fi

    # Start services
    log_info "Starting Docker services..."
    docker-compose up -d

    log_success "Docker deployment completed"
    echo ""
    echo "🐳 FXorcist is running in Docker!"
    echo "📊 Main Dashboard:    http://localhost:8501"
    echo "🎯 Training Dashboard: http://localhost:8502"
    echo "🧠 Memory Dashboard:   http://localhost:8503"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
}

# Deploy to cloud (AWS/GCP/Azure)
deploy_cloud() {
    local provider="${3:-aws}"
    log_info "Deploying to cloud: $provider"

    case $provider in
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
        *)
            log_error "Unsupported cloud provider: $provider"
            log_info "Supported providers: aws, gcp, azure"
            exit 1
            ;;
    esac
}

# AWS deployment
deploy_aws() {
    log_info "Deploying to AWS..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is required but not installed"
        exit 1
    fi

    # Create deployment package
    log_info "Creating deployment package..."
    mkdir -p deploy/aws
    cp -r . deploy/aws/ 2>/dev/null || true
    cd deploy/aws

    # Create AWS deployment files
    cat > appspec.yml << EOF
version: 0.0
os: linux
files:
  - source: /
    destination: /home/ec2-user/fxorcist
hooks:
  ApplicationStart:
    - location: scripts/deploy_aws.sh
      timeout: 300
      runas: ec2-user
EOF

    cat > scripts/deploy_aws.sh << 'EOF'
#!/bin/bash
cd /home/ec2-user/fxorcist
python3 scripts/setup_fxorcist.py
python3 scripts/dashboard_launcher.py start-all
EOF

    chmod +x scripts/deploy_aws.sh

    log_info "AWS deployment package created in deploy/aws/"
    log_info "Upload this package to AWS CodeDeploy or Elastic Beanstalk"
}

# GCP deployment
deploy_gcp() {
    log_info "Deploying to Google Cloud Platform..."

    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is required but not installed"
        exit 1
    fi

    # Create app.yaml for App Engine
    cat > app.yaml << EOF
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
  DEPLOY_ENV: production
  GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT
EOF

    log_info "GCP deployment configuration created"
    log_info "Deploy with: gcloud app deploy"
}

# Azure deployment
deploy_azure() {
    log_info "Deploying to Microsoft Azure..."

    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is required but not installed"
        exit 1
    fi

    # Create Azure deployment files
    mkdir -p deploy/azure

    cat > deploy/azure/Dockerfile << EOF
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501 8502 8503
CMD ["python", "fxorcist_cli.py", "--dashboard"]
EOF

    log_info "Azure deployment configuration created in deploy/azure/"
    log_info "Deploy with: az webapp up --name fxorcist --resource-group fxorcist-rg"
}

# Health check after deployment
post_deployment_check() {
    log_info "Running post-deployment health checks..."

    # Wait for services to start
    sleep 10

    # Check main dashboard
    if curl -f http://localhost:8501 > /dev/null 2>&1; then
        log_success "Main dashboard is healthy"
    else
        log_warning "Main dashboard health check failed"
    fi

    # Check system status
    if [[ -f "$PROJECT_ROOT/fxorcist_cli.py" ]]; then
        python3 "$PROJECT_ROOT/fxorcist_cli.py" --command "health system" > /dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "System health check passed"
        else
            log_warning "System health check failed"
        fi
    fi
}

# Main deployment logic
main() {
    echo "🚀 FXorcist Deployment Script"
    echo "Environment: $DEPLOY_ENV"
    echo "Type: $DEPLOY_TYPE"
    echo ""

    # Run pre-deployment checks
    pre_deployment_checks

    # Setup environment
    setup_environment "$DEPLOY_ENV"

    # Install dependencies
    install_dependencies

    # Setup data directories
    setup_data_directories

    # Deploy based on type
    case $DEPLOY_TYPE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        cloud)
            deploy_cloud "$@"
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOY_TYPE"
            log_info "Supported types: local, docker, cloud"
            exit 1
            ;;
    esac

    # Post-deployment checks
    post_deployment_check

    log_success "Deployment completed successfully!"
}

# Show usage if no arguments provided
if [[ $# -eq 0 ]]; then
    echo "FXorcist Deployment Script"
    echo ""
    echo "Usage: $0 [environment] [type] [options]"
    echo ""
    echo "Environments:"
    echo "  development    Development environment (default)"
    echo "  staging        Staging environment"
    echo "  production     Production environment"
    echo ""
    echo "Types:"
    echo "  local          Local deployment (default)"
    echo "  docker         Docker deployment"
    echo "  cloud          Cloud deployment"
    echo ""
    echo "Cloud Options:"
    echo "  aws            Deploy to AWS (default)"
    echo "  gcp            Deploy to Google Cloud"
    echo "  azure          Deploy to Azure"
    echo ""
    echo "Examples:"
    echo "  $0                    # Local development deployment"
    echo "  $0 production local   # Local production deployment"
    echo "  $0 production docker  # Docker production deployment"
    echo "  $0 production cloud aws  # AWS production deployment"
    echo ""
    exit 0
fi

# Run main deployment
main "$@"
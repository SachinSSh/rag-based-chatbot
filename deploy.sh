#!/bin/bash

# Advanced RAG Chatbot Deployment Script
# This script handles deployment to various environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="advanced-rag-chatbot"
DEFAULT_PORT=8501
DEFAULT_ENV="production"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Advanced RAG Chatbot Deploy  ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command -v pip &> /dev/null; then
        print_error "pip is required but not installed"
        exit 1
    fi
    
    # Check if required environment variables are set
    if [[ -z "$GEMINI_API_KEY" ]]; then
        print_warning "GEMINI_API_KEY environment variable is not set"
        print_info "Please set it in your .env file or environment"
    fi
    
    print_success "Dependencies check completed"
}

setup_environment() {
    print_info "Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Download required models
    print_info "Downloading required ML models..."
    python -m spacy download en_core_web_sm || print_warning "Failed to download spaCy model"
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || print_warning "Failed to download NLTK data"
    
    # Create necessary directories
    mkdir -p data/{chromadb,faiss_index}
    mkdir -p {models,temp_uploads,logs}
    
    print_success "Environment setup completed"
}

run_tests() {
    print_info "Running tests..."
    
    if [ -f "test_rag.py" ]; then
        python -m pytest test_rag.py -v --tb=short -m "not integration"
        if [ $? -eq 0 ]; then
            print_success "All tests passed"
        else
            print_error "Some tests failed"
            return 1
        fi
    else
        print_warning "Test file not found, skipping tests"
    fi
}

deploy_local() {
    local port=${1:-$DEFAULT_PORT}
    
    print_info "Deploying locally on port $port..."
    
    # Check if port is available
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port $port is already in use"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Start the application
    export STREAMLIT_SERVER_PORT=$port
    export STREAMLIT_SERVER_ADDRESS=0.0.0.0
    
    print_success "Starting application on http://localhost:$port"
    python main.py --mode web
}

deploy_docker() {
    print_info "Deploying with Docker..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Build Docker image
    print_info "Building Docker image..."
    docker build -t $PROJECT_NAME:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    # Run container
    print_info "Starting Docker container..."
    docker run -d \
        --name $PROJECT_NAME \
        -p 8501:8501 \
        -e GEMINI_API_KEY="$GEMINI_API_KEY" \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/logs:/app/logs \
        --restart unless-stopped \
        $PROJECT_NAME:latest
    
    if [ $? -eq 0 ]; then
        print_success "Docker container started successfully"
        print_info "Application available at http://localhost:8501"
    else
        print_error "Failed to start Docker container"
        exit 1
    fi
}

deploy_docker_compose() {
    print_info "Deploying with Docker Compose..."
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    # Start services
    print_info "Starting services with Docker Compose..."
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "All services started successfully"
        print_info "Application available at http://localhost:8501"
        print_info "Grafana dashboard available at http://localhost:3000"
        print_info "Prometheus metrics available at http://localhost:9090"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

deploy_kubernetes() {
    print_info "Deploying to Kubernetes..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check if kubernetes manifests exist
    if [ ! -d "k8s" ]; then
        print_error "Kubernetes manifests directory (k8s) not found"
        exit 1
    fi
    
    # Apply manifests
    print_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/
    
    if [ $? -eq 0 ]; then
        print_success "Kubernetes deployment successful"
        print_info "Check service status with: kubectl get services"
    else
        print_error "Kubernetes deployment failed"
        exit 1
    fi
}

deploy_cloud() {
    local platform=$1
    
    case $platform in
        "aws")
            print_info "Deploying to AWS..."
            # AWS deployment logic would go here
            print_warning "AWS deployment not implemented yet"
            ;;
        "gcp")
            print_info "Deploying to Google Cloud Platform..."
            # GCP deployment logic would go here
            print_warning "GCP deployment not implemented yet"
            ;;
        "azure")
            print_info "Deploying to Azure..."
            # Azure deployment logic would go here
            print_warning "Azure deployment not implemented yet"
            ;;
        *)
            print_error "Unknown cloud platform: $platform"
            print_info "Supported platforms: aws, gcp, azure"
            exit 1
            ;;
    esac
}

health_check() {
    local url=${1:-"http://localhost:8501"}
    
    print_info "Running health check on $url..."
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -f "$url/_stcore/health" >/dev/null 2>&1; then
            print_success "Health check passed"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    print_error "Health check failed"
    return 1
}

backup_data() {
    print_info "Creating data backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup data directory
    if [ -d "data" ]; then
        cp -r data "$backup_dir/"
        print_success "Data backed up to $backup_dir"
    else
        print_warning "No data directory found"
    fi
    
    # Backup logs
    if [ -d "logs" ]; then
        cp -r logs "$backup_dir/"
        print_success "Logs backed up to $backup_dir"
    fi
}

restore_data() {
    local backup_path=$1
    
    if [ -z "$backup_path" ]; then
        print_error "Backup path is required"
        exit 1
    fi
    
    if [ ! -d "$backup_path" ]; then
        print_error "Backup directory not found: $backup_path"
        exit 1
    fi
    
    print_warning "This will overwrite existing data. Continue? (y/N)"
    read -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Restoring data from $backup_path..."
        
        # Restore data
        if [ -d "$backup_path/data" ]; then
            rm -rf data
            cp -r "$backup_path/data" .
            print_success "Data restored"
        fi
        
        # Restore logs
        if [ -d "$backup_path/logs" ]; then
            rm -rf logs
            cp -r "$backup_path/logs" .
            print_success "Logs restored"
        fi
    else
        print_info "Restore cancelled"
    fi
}

stop_services() {
    print_info "Stopping services..."
    
    # Stop Docker containers
    if docker ps | grep -q $PROJECT_NAME; then
        docker stop $PROJECT_NAME
        docker rm $PROJECT_NAME
        print_success "Docker container stopped"
    fi
    
    # Stop Docker Compose services
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
        print_success "Docker Compose services stopped"
    fi
    
    # Kill any running Python processes
    pkill -f "streamlit" || true
    pkill -f "main.py" || true
    
    print_success "All services stopped"
}

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  local [PORT]              Deploy locally (default port: 8501)"
    echo "  docker                    Deploy with Docker"
    echo "  docker-compose           Deploy with Docker Compose"
    echo "  kubernetes               Deploy to Kubernetes"
    echo "  cloud [aws|gcp|azure]    Deploy to cloud platform"
    echo "  test                     Run tests"
    echo "  health [URL]             Run health check"
    echo "  backup                   Backup data"
    echo "  restore [PATH]           Restore from backup"
    echo "  stop                     Stop all services"
    echo "  setup                    Setup environment only"
    echo
    echo "Options:"
    echo "  -h, --help               Show this help message"
    echo "  --skip-tests            Skip running tests"
    echo "  --no-backup             Skip creating backup"
    echo
    echo "Environment Variables:"
    echo "  GEMINI_API_KEY          Required: Google Gemini API key"
    echo "  ENVIRONMENT             Optional: deployment environment (default: production)"
    echo "  DEBUG                   Optional: enable debug mode (default: false)"
}

# Parse command line arguments
SKIP_TESTS=false
NO_BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Main execution
print_header

case ${1:-"local"} in
    "local")
        check_dependencies
        setup_environment
        if [ "$SKIP_TESTS" = false ]; then
            run_tests || exit 1
        fi
        deploy_local ${2:-$DEFAULT_PORT}
        ;;
    "docker")
        check_dependencies
        deploy_docker
        sleep 5
        health_check
        ;;
    "docker-compose")
        check_dependencies
        deploy_docker_compose
        sleep 10
        health_check
        ;;
    "kubernetes")
        deploy_kubernetes
        ;;
    "cloud")
        deploy_cloud $2
        ;;
    "test")
        check_dependencies
        setup_environment
        run_tests
        ;;
    "health")
        health_check $2
        ;;
    "backup")
        backup_data
        ;;
    "restore")
        restore_data $2
        ;;
    "stop")
        stop_services
        ;;
    "setup")
        check_dependencies
        setup_environment
        print_success "Setup completed"
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

print_success "Deployment script completed successfully!"
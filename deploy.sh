# =============================================================================
# FILE: deploy.sh
# =============================================================================
#!/bin/bash

set -e

echo "🚀 Deploying Dynamic Data Mapping Validator..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p models reports logs schemas

# Set permissions
print_status "Setting directory permissions..."
chmod 755 models reports logs schemas mappings sql src 2>/dev/null || true

# Check if .env file exists, if not create from template
if [ ! -f .env ]; then
    if [ -f .env.template ]; then
        print_status "Creating .env file..."
        cp .env.template .env
        print_warning "Please review and update .env file with your configuration"
    else
        print_error ".env.template not found. Cannot create .env file."
        exit 1
    fi
fi

# Build and start services
print_status "Building Docker images..."
docker-compose build --no-cache

print_status "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_status "Waiting for services to be ready..."
sleep 15

# Check service health
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "$service.*healthy\|$service.*Up"; then
            print_success "$service is ready"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "$service failed to become ready"
            return 1
        fi
        
        print_status "Waiting for $service to be ready... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
}

# Check application readiness
print_status "Waiting for application to be ready..."
for i in {1..20}; do
    if curl -f http://localhost:8000/status >/dev/null 2>&1; then
        print_success "Application is ready!"
        break
    fi
    if [ $i -eq 20 ]; then
        print_error "Application failed to start"
        docker-compose logs validator
        exit 1
    fi
    sleep 3
done

# Display status
print_status "Checking deployment status..."
docker-compose ps

echo ""
print_success "🎉 Deployment completed successfully!"
echo ""
echo "Access the application at:"
echo "  Web Interface: http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo ""
echo "Database connections:"
echo "  Source DB: localhost:5432 (source_system/source_user/source_pass)"
echo "  Target DB: localhost:5433 (target_system/target_user/target_pass)"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose logs -f validator"
echo "  Stop services: docker-compose down"
echo "  Restart: docker-compose restart"
echo "  View status: docker-compose ps"

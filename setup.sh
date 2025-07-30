# =============================================================================
# FILE: setup.sh
# =============================================================================
#!/bin/bash

# Dynamic Data Mapping Validator - Project Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}  Dynamic Data Mapping Validator Setup${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

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
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create project structure
create_structure() {
    print_status "Creating project directory structure..."
    
    # Create directories
    mkdir -p {scripts,config,src/{database,models,api},sql/{migrations,sample_data},mappings/{examples,templates},models,reports,logs,schemas,backups,tests/{fixtures/{sample_mappings,test_data},integration},docs/examples,monitoring/{grafana/dashboards,alerting}}
    
    # Create .gitkeep files for empty directories
    touch {models,reports,logs,schemas,backups}/.gitkeep
    
    # Create __init__.py files for Python packages
    touch src/__init__.py
    touch src/database/__init__.py
    touch src/models/__init__.py
    touch src/api/__init__.py
    touch tests/__init__.py
    
    print_success "Project structure created"
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."
    
    # Copy environment template if .env doesn't exist
    if [ ! -f .env ]; then
        cp .env.template .env
        print_warning "Created .env file from template. Please review and update as needed."
    fi
    
    # Make scripts executable
    chmod +x deploy.sh 2>/dev/null || true
    chmod +x manage.sh 2>/dev/null || true
    find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
    
    print_success "Configuration setup complete"
}

# Validate setup
validate_setup() {
    print_status "Validating setup..."
    
    ERRORS=0
    
    # Check essential files
    for file in "docker-compose.yml" "Dockerfile" "requirements.txt" ".env"; do
        if [ ! -f "$file" ]; then
            print_error "Missing file: $file"
            ((ERRORS++))
        fi
    done
    
    # Check directories
    for dir in "src" "sql" "mappings" "scripts"; do
        if [ ! -d "$dir" ]; then
            print_error "Missing directory: $dir"
            ((ERRORS++))
        fi
    done
    
    if [ $ERRORS -eq 0 ]; then
        print_success "Setup validation complete"
    else
        print_error "Found $ERRORS error(s). Please fix them before proceeding."
        exit 1
    fi
}

# Main setup process
main() {
    print_header
    
    print_status "Starting setup process..."
    
    check_prerequisites
    create_structure
    setup_config
    validate_setup
    
    print_success "Setup completed successfully!"
    
    echo -e "\n${GREEN}Next steps:${NC}"
    echo "1. Review and update .env file"
    echo "2. Run: ./deploy.sh"
    echo "3. Access: http://localhost:8000"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@"

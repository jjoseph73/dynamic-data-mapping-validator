#!/bin/bash
# deploy.sh - Production Deployment Script

set -euo pipefail

# Configuration
APP_NAME="ddmv-app"
DEPLOY_DIR="/opt/${APP_NAME}"
BACKUP_DIR="/opt/backups"
LOG_FILE="/var/log/deploy.log"
HEALTH_CHECK_URL="https://your-domain.com/health"
MAX_HEALTH_CHECKS=30
HEALTH_CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
}

# Backup current deployment
backup_current() {
    log "Creating backup of current deployment..."
    
    local backup_name="${APP_NAME}-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup application files
    if [[ -d "$DEPLOY_DIR" ]]; then
        cp -r "$DEPLOY_DIR" "$backup_path"
        log "Application files backed up to $backup_path"
    fi
    
    # Backup database
    docker exec ddmv-postgres pg_dump -U ddmv_user ddmv_db > "${backup_path}/database.sql"
    log "Database backed up to ${backup_path}/database.sql"
    
    # Cleanup old backups (keep last 7 days)
    find "$BACKUP_DIR" -type d -name "${APP_NAME}-*" -mtime +7 -exec rm -rf {} \;
    log "Old backups cleaned up"
}

# Update application code
update_code() {
    log "Updating application code..."
    
    cd "$DEPLOY_DIR"
    
    # Stash any local changes
    git stash
    
    # Pull latest changes
    git fetch origin
    git checkout main
    git pull origin main
    
    log "Code updated successfully"
}

# Update Docker images
update_images() {
    log "Updating Docker images..."
    
    cd "$DEPLOY_DIR"
    
    # Pull latest images
    docker-compose -f docker-compose.prod.yml pull
    
    log "Docker images updated"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Check if migrations are needed
    if docker exec ddmv-app alembic current | grep -q "head"; then
        log "Database is up to date"
        return 0
    fi
    
    # Run migrations
    docker exec ddmv-app alembic upgrade head
    
    if [[ $? -eq 0 ]]; then
        log "Database migrations completed successfully"
    else
        error "Database migrations failed"
    fi
}

# Deploy application
deploy_app() {
    log "Deploying application..."
    
    cd "$DEPLOY_DIR"
    
    # Start services with zero-downtime deployment
    docker-compose -f docker-compose.prod.yml up -d --no-deps --remove-orphans app
    
    log "Application deployed"
}

# Health check
health_check() {
    log "Running health checks..."
    
    local attempts=0
    while [[ $attempts -lt $MAX_HEALTH_CHECKS ]]; do
        if curl -f -s "$HEALTH_CHECK_URL" > /dev/null; then
            log "Health check passed"
            return 0
        fi
        
        attempts=$((attempts + 1))
        warning "Health check failed (attempt $attempts/$MAX_HEALTH_CHECKS), retrying in ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    error "Health checks failed after $MAX_HEALTH_CHECKS attempts"
}

# Cleanup old Docker resources
cleanup() {
    log "Cleaning up old Docker resources..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log "Cleanup completed"
}

# Rollback function
rollback() {
    log "Rolling back to previous version..."
    
    # Get latest backup
    local latest_backup
    latest_backup=$(find "$BACKUP_DIR" -type d -name "${APP_NAME}-*" | sort -r | head -n1)
    
    if [[ -z "$latest_backup" ]]; then
        error "No backup found for rollback"
    fi
    
    # Stop current services
    cd "$DEPLOY_DIR"
    docker-compose -f docker-compose.prod.yml down
    
    # Restore backup
    rm -rf "${DEPLOY_DIR}.old"
    mv "$DEPLOY_DIR" "${DEPLOY_DIR}.old"
    cp -r "$latest_backup" "$DEPLOY_DIR"
    
    # Restore database
    docker exec ddmv-postgres psql -U ddmv_user -d ddmv_db < "${latest_backup}/database.sql"
    
    # Start services
    cd "$DEPLOY_DIR"
    docker-compose -f docker-compose.prod.yml up -d
    
    log "Rollback completed"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 Deployment ${status}: ${message}\"}" \
            "$SLACK_WEBHOOK"
    fi
    
    if [[ -n "${EMAIL_RECIPIENTS:-}" ]]; then
        echo "$message" | mail -s "Deployment ${status}" "$EMAIL_RECIPIENTS"
    fi
}

# Main deployment function
main() {
    local start_time
    start_time=$(date +%s)
    
    log "Starting deployment of $APP_NAME"
    
    # Check permissions
    check_permissions
    
    # Handle rollback option
    if [[ "${1:-}" == "--rollback" ]]; then
        rollback
        send_notification "ROLLBACK" "Application rolled back successfully"
        exit 0
    fi
    
    # Create backup
    backup_current
    
    # Update code and images
    update_code
    update_images
    
    # Run migrations
    run_migrations
    
    # Deploy application
    deploy_app
    
    # Health check
    if ! health_check; then
        warning "Health check failed, initiating rollback..."
        rollback
        send_notification "FAILED" "Deployment failed, rolled back to previous version"
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Deployment completed successfully in ${duration}s"
    send_notification "SUCCESS" "Deployment completed successfully in ${duration}s"
}

# Trap errors and rollback
trap 'error "Deployment failed, initiating rollback..."; rollback; send_notification "FAILED" "Deployment failed and was rolled back"' ERR

# Run main function
main "$@"
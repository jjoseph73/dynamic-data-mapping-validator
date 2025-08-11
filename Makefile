# Makefile for DDMV Production Deployment

.PHONY: help build test deploy stop clean backup restore logs monitor security
.DEFAULT_GOAL := help

# Configuration
PROJECT_NAME := ddmv-app
COMPOSE_FILE := docker-compose.prod.yml
ENV_FILE := /etc/ddmv/secrets/.env.production
BACKUP_DIR := /opt/backups
LOG_DIR := ./logs

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "DDMV Production Deployment Commands"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Commands
dev-setup: ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	python -m venv venv
	. venv/bin/activate && pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-start: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment started at http://localhost:8000$(NC)"

dev-stop: ## Stop development environment
	@echo "$(YELLOW)Stopping development environment...$(NC)"
	docker-compose -f docker-compose.dev.yml down

# Testing Commands
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	pytest tests/performance/ -v --benchmark-only

lint: ## Run code linting
	@echo "$(GREEN)Running linters...$(NC)"
	flake8 app/
	mypy app/
	bandit -r app/

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	black app/ tests/
	isort app/ tests/

# Build Commands
build: ## Build production Docker image
	@echo "$(GREEN)Building production image...$(NC)"
	docker-compose -f $(COMPOSE_FILE) build --no-cache

build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development image...$(NC)"
	docker-compose -f docker-compose.dev.yml build --no-cache

pull: ## Pull latest images
	@echo "$(GREEN)Pulling latest images...$(NC)"
	docker-compose -f $(COMPOSE_FILE) pull

# Deployment Commands
deploy: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)Environment file not found: $(ENV_FILE)$(NC)"; \
		exit 1; \
	fi
	./scripts/deploy.sh

deploy-staging: ## Deploy to staging
	@echo "$(GREEN)Deploying to staging...$(NC)"
	docker-compose -f docker-compose.staging.yml up -d

rollback: ## Rollback to previous version
	@echo "$(YELLOW)Rolling back to previous version...$(NC)"
	./scripts/deploy.sh --rollback

# Service Management
start: ## Start all services
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d

stop: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down

restart: ## Restart all services
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) restart

restart-app: ## Restart only the application
	@echo "$(YELLOW)Restarting application...$(NC)"
	docker-compose -f $(COMPOSE_FILE) restart app

status: ## Show service status
	@echo "$(GREEN)Service Status:$(NC)"
	docker-compose -f $(COMPOSE_FILE) ps

# Database Commands
db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec app alembic upgrade head

db-rollback: ## Rollback database migration
	@echo "$(YELLOW)Rolling back database migration...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec app alembic downgrade -1

db-shell: ## Open database shell
	@echo "$(GREEN)Opening database shell...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U ddmv_user -d ddmv_db

db-backup: ## Backup database
	@echo "$(GREEN)Backing up database...$(NC)"
	mkdir -p $(BACKUP_DIR)
	docker-compose -f $(COMPOSE_FILE) exec postgres pg_dump -U ddmv_user ddmv_db > $(BACKUP_DIR)/db-backup-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "$(GREEN)Database backed up to $(BACKUP_DIR)$(NC)"

db-restore: ## Restore database from backup
	@echo "$(YELLOW)Restoring database...$(NC)"
	@read -p "Enter backup file path: " backup_file; \
	docker-compose -f $(COMPOSE_FILE) exec -T postgres psql -U ddmv_user -d ddmv_db < $$backup_file

# Monitoring Commands
logs: ## Show application logs
	@echo "$(GREEN)Showing application logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f app

logs-all: ## Show all service logs
	@echo "$(GREEN)Showing all service logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-nginx: ## Show Nginx logs
	@echo "$(GREEN)Showing Nginx logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f nginx

logs-db: ## Show database logs
	@echo "$(GREEN)Showing database logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f postgres

health: ## Check service health
	@echo "$(GREEN)Checking service health...$(NC)"
	@curl -f http://localhost/health && echo "$(GREEN)✓ Application is healthy$(NC)" || echo "$(RED)✗ Application is unhealthy$(NC)"
	@docker-compose -f $(COMPOSE_FILE) exec postgres pg_isready -U ddmv_user && echo "$(GREEN)✓ Main Database is healthy$(NC)" || echo "$(RED)✗ Main Database is unhealthy$(NC)"
	@docker-compose -f $(COMPOSE_FILE) exec source-postgres pg_isready -U source_user && echo "$(GREEN)✓ Source Database is healthy$(NC)" || echo "$(RED)✗ Source Database is unhealthy$(NC)"
	@docker-compose -f $(COMPOSE_FILE) exec redis redis-cli ping && echo "$(GREEN)✓ Redis is healthy$(NC)" || echo "$(RED)✗ Redis is unhealthy$(NC)"

monitor: ## Open monitoring dashboard
	@echo "$(GREEN)Opening monitoring dashboards...$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Kibana: http://localhost:5601"

# Backup and Restore Commands
backup: ## Create full system backup
	@echo "$(GREEN)Creating full system backup...$(NC)"
	./scripts/backup.sh

backup-code: ## Backup application code
	@echo "$(GREEN)Backing up application code...$(NC)"
	mkdir -p $(BACKUP_DIR)
	tar -czf $(BACKUP_DIR)/code-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		--exclude='.git' \
		--exclude='venv' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='logs' \
		.

restore: ## Restore from backup
	@echo "$(YELLOW)Restoring from backup...$(NC)"
	./scripts/restore.sh

# Security Commands
security: ## Run security setup
	@echo "$(GREEN)Running security setup...$(NC)"
	sudo ./security/setup-security.sh

security-scan: ## Run security scan
	@echo "$(GREEN)Running security scan...$(NC)"
	docker run --rm -v $(PWD):/app aquasec/trivy fs /app
	bandit -r app/ -f json -o security-report.json

ssl-renew: ## Renew SSL certificates
	@echo "$(GREEN)Renewing SSL certificates...$(NC)"
	certbot renew --nginx
	docker-compose -f $(COMPOSE_FILE) restart nginx

# Maintenance Commands
clean: ## Clean up Docker resources
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker system prune -f
	docker volume prune -f
	docker image prune -f

clean-all: ## Clean up everything (DANGEROUS)
	@echo "$(RED)WARNING: This will remove all Docker resources!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker system prune -af --volumes; \
	fi

update: ## Update system and Docker images
	@echo "$(GREEN)Updating system and Docker images...$(NC)"
	sudo apt update && sudo apt upgrade -y
	docker-compose -f $(COMPOSE_FILE) pull
	$(MAKE) restart

# Environment Commands
env-check: ## Check environment configuration
	@echo "$(GREEN)Checking environment configuration...$(NC)"
	@if [ -f $(ENV_FILE) ]; then \
		echo "$(GREEN)✓ Environment file exists$(NC)"; \
	else \
		echo "$(RED)✗ Environment file missing: $(ENV_FILE)$(NC)"; \
	fi
	@docker --version && echo "$(GREEN)✓ Docker is installed$(NC)" || echo "$(RED)✗ Docker is not installed$(NC)"
	@docker-compose --version && echo "$(GREEN)✓ Docker Compose is installed$(NC)" || echo "$(RED)✗ Docker Compose is not installed$(NC)"

env-template: ## Generate environment template
	@echo "$(GREEN)Generating environment template...$(NC)"
	cp .env.production.template .env.production
	@echo "$(YELLOW)Please edit .env.production with your values$(NC)"

# Performance Commands
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	docker run --rm --network host jordi/ab -n 1000 -c 10 http://localhost/api/health

load-test: ## Run load tests
	@echo "$(GREEN)Running load tests...$(NC)"
	locust -f tests/load/locustfile.py --host=http://localhost

# CI/CD Commands
ci: ## Run CI pipeline locally
	@echo "$(GREEN)Running CI pipeline...$(NC)"
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security-scan
	$(MAKE) build

cd: ## Run CD pipeline
	@echo "$(GREEN)Running CD pipeline...$(NC)"
	$(MAKE) pull
	$(MAKE) deploy
	$(MAKE) health

# Quick Commands
quick-deploy: ## Quick deployment (pull, restart, migrate)
	@echo "$(GREEN)Quick deployment...$(NC)"
	$(MAKE) pull
	$(MAKE) restart
	$(MAKE) db-migrate
	$(MAKE) health

quick-fix: ## Quick fix (restart services and check health)
	@echo "$(GREEN)Quick fix...$(NC)"
	$(MAKE) restart
	sleep 30
	$(MAKE) health

emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP - Stopping all services...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --remove-orphans
	docker stop $$(docker ps -aq) 2>/dev/null || true

# Documentation
docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html

# Development Helpers
shell: ## Open application shell
	@echo "$(GREEN)Opening application shell...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec app /bin/bash

shell-db: ## Open database shell
	@echo "$(GREEN)Opening database shell...$(NC)"
	$(MAKE) db-shell

tail-logs: ## Tail all logs
	@echo "$(GREEN)Tailing logs...$(NC)"
	tail -f $(LOG_DIR)/*.log

watch-health: ## Watch health status
	@echo "$(GREEN)Watching health status...$(NC)"
	watch -n 5 'curl -s http://localhost/health || echo "Health check failed"'
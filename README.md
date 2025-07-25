# =============================================================================
# FILE: README.md
# =============================================================================
# Dynamic Data Mapping Validator

AI-powered validation system for database migration mappings with dynamic configuration management.

## 🚀 Quick Start

```bash
# Setup project
git clone <repository>
cd dynamic-data-mapping-validator
./setup.sh

# Deploy
./deploy.sh

# Access
open http://localhost:8000
```

## 📁 Project Structure

```
dynamic-data-mapping-validator/
├── setup.sh                    # Project setup script
├── deploy.sh                   # Deployment script
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # Application container
├── requirements.txt            # Python dependencies
├── .env.template               # Environment template
├── src/                        # Application source code
├── sql/                        # Database initialization
├── mappings/                   # Dynamic mapping configurations
├── models/                     # Saved ML models
├── reports/                    # Validation reports
└── logs/                       # Application logs
```

## 🛠️ Management

```bash
# Start services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f validator

# Stop services
docker-compose down

# Restart services
docker-compose restart
```

## 🌐 Access Points

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Source Database**: localhost:5432 (source_system/source_user/source_pass)
- **Target Database**: localhost:5433 (target_system/target_user/target_pass)

## 📝 Adding New Mappings

### Method 1: Web Interface
1. Go to http://localhost:8000
2. Use "Add New Mapping" section
3. Upload JSON file or paste configuration

### Method 2: File System
```bash
# Add new mapping file
cp my_new_mapping.json mappings/
# Reload mappings
curl -X POST http://localhost:8000/reload-mappings
```

### Method 3: API
```bash
curl -X POST http://localhost:8000/mappings \
  -H "Content-Type: application/json" \
  -d @new_mapping.json
```

## 📖 Documentation

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add your mappings to `mappings/`
4. Test with validation endpoints
5. Submit pull request

## 📄 License

[Your License Here]

# =============================================================================
# END OF FILE BUNDLE
# =============================================================================

# EXTRACTION INSTRUCTIONS:
# 1. Copy each section above into separate files as indicated by "FILE: filename"
# 2. Maintain the exact file structure shown
# 3. Make scripts executable: chmod +x setup.sh deploy.sh
# 4. Run: ./setup.sh
# 5. Then: ./deploy.sh

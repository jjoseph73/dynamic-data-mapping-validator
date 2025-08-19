"""
Dynamic Data Mapping Validator - Main Application Entry Point
Updated to work with FastAPI application structure
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to Python path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import the FastAPI app so it's available at module level for ASGI servers
from api.app import app

def setup_logging(log_level: str = "INFO") -> None:
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log') if Path('logs').exists() else logging.StreamHandler(sys.stdout)
        ]
    )

def validate_environment() -> bool:
    """Validate that all required environment variables and directories exist"""
    logger = logging.getLogger(__name__)
    
    # Required directories
    required_dirs = ['mappings', 'models', 'sql']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.warning(f"Creating missing directory: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Optional logs directory
    logs_dir = Path('logs')
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check environment variables (with defaults)
    env_vars = {
        'DATABASE_URL': 'postgresql://user:password@localhost:5432/validation_db',
        'SOURCE_DB_URL': 'postgresql://user:password@localhost:5432/source_db',
        'TARGET_DB_URL': 'postgresql://user:password@localhost:5432/target_db',
        'LOG_LEVEL': 'INFO',
        'ENVIRONMENT': 'development'
    }
    
    for var, default in env_vars.items():
        if not os.getenv(var):
            logger.info(f"Environment variable {var} not set, using default")
            os.environ[var] = default
    
    return True

def create_sample_mappings() -> None:
    """Create sample mapping files if they don't exist"""
    logger = logging.getLogger(__name__)
    mappings_dir = Path('mappings')
    
    # Check if we have any mapping files
    existing_mappings = list(mappings_dir.glob('*.json'))
    if existing_mappings:
        logger.info(f"Found {len(existing_mappings)} existing mapping files")
        return
    
    logger.info("No mapping files found, sample mappings should be created")
    logger.info("Sample mappings can be found in the artifacts section")

def main() -> None:
    """Main application entry point"""
    # Setup logging first
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Dynamic Data Mapping Validator")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Create sample mappings if needed
        create_sample_mappings()
        
        # Import and run the FastAPI application
        logger.info("Initializing FastAPI application")
        
        # The actual app is now in src/api/app.py
        from api.app import app
        
        # If running directly (not via uvicorn), start the server
        if __name__ == "__main__":
            import uvicorn
            
            host = os.getenv('HOST', '0.0.0.0')
            port = int(os.getenv('PORT', 8000))
            reload = os.getenv('ENVIRONMENT', 'development') == 'development'
            
            logger.info(f"Starting server on {host}:{port}")
            logger.info(f"Environment: {os.getenv('ENVIRONMENT')}")
            logger.info(f"API Documentation: http://{host}:{port}/api/docs")
            
            uvicorn.run(
                "api.app:app",
                host=host,
                port=port,
                reload=reload,
                log_level=log_level.lower()
            )
        
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
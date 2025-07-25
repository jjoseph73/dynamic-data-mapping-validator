# =============================================================================
# src/main.py - Main Application Entry Point
# Dynamic Data Mapping Validator
# =============================================================================

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core application imports
from dynamic_validator import app, mapping_manager, validator
from mapping_manager import MappingManager
from database_connector import DatabaseConnector

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_dir = Path('/app/logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / 'validator.log')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

def wait_for_database(db_connector: DatabaseConnector, max_attempts: int = 30) -> bool:
    """
    Wait for database to become available
    
    Args:
        db_connector: Database connector instance
        max_attempts: Maximum number of connection attempts
        
    Returns:
        True if database is ready, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    for attempt in range(1, max_attempts + 1):
        try:
            conn = db_connector.get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
            conn.close()
            logger.info(f"Database connection successful on attempt {attempt}")
            return True
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(2)
    
    logger.error(f"Database failed to become ready after {max_attempts} attempts")
    return False

def validate_environment() -> bool:
    """
    Validate required environment variables and setup
    
    Returns:
        True if environment is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    required_vars = [
        'SOURCE_DB_HOST', 'SOURCE_DB_USER', 'SOURCE_DB_PASSWORD', 'SOURCE_DB_NAME',
        'TARGET_DB_HOST', 'TARGET_DB_USER', 'TARGET_DB_PASSWORD', 'TARGET_DB_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check required directories
    required_dirs = ['/app/mappings', '/app/models', '/app/reports', '/app/logs']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
    
    logger.info("Environment validation passed")
    return True

def initialize_databases() -> tuple[Optional[DatabaseConnector], Optional[DatabaseConnector]]:
    """
    Initialize database connections
    
    Returns:
        Tuple of (source_db, target_db) connectors or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize source database connection
        source_db = DatabaseConnector(
            host=os.getenv('SOURCE_DB_HOST', 'source_db'),
            port=int(os.getenv('SOURCE_DB_PORT', 5432)),
            database=os.getenv('SOURCE_DB_NAME', 'source_system'),
            username=os.getenv('SOURCE_DB_USER', 'source_user'),
            password=os.getenv('SOURCE_DB_PASSWORD', 'source_pass')
        )
        
        # Initialize target database connection
        target_db = DatabaseConnector(
            host=os.getenv('TARGET_DB_HOST', 'target_db'),
            port=int(os.getenv('TARGET_DB_PORT', 5432)),
            database=os.getenv('TARGET_DB_NAME', 'target_system'),
            username=os.getenv('TARGET_DB_USER', 'target_user'),
            password=os.getenv('TARGET_DB_PASSWORD', 'target_pass')
        )
        
        # Wait for databases to be ready
        logger.info("Waiting for source database to be ready...")
        if not wait_for_database(source_db):
            logger.error("Source database failed to become ready")
            return None, None
        
        logger.info("Waiting for target database to be ready...")
        if not wait_for_database(target_db):
            logger.error("Target database failed to become ready")
            return None, None
        
        logger.info("Database connections initialized successfully")
        return source_db, target_db
        
    except Exception as e:
        logger.error(f"Failed to initialize database connections: {e}")
        return None, None

def initialize_application() -> bool:
    """
    Initialize the application components
    
    Returns:
        True if initialization successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize mapping manager
        logger.info("Initializing mapping manager...")
        mapping_manager.load_all_mappings()
        mappings_count = len(mapping_manager.get_all_mappings())
        logger.info(f"Loaded {mappings_count} mapping configurations")
        
        if mappings_count == 0:
            logger.warning("No mapping configurations found. The system will start but cannot validate until mappings are added.")
        
        # Try to load existing AI model
        logger.info("Loading AI validation model...")
        if validator.load_model():
            logger.info("Existing AI model loaded successfully")
        else:
            logger.info("No existing model found. Will train new model...")
            if mappings_count > 0:
                try:
                    logger.info("Training new AI model...")
                    accuracy = validator.train_model()
                    logger.info(f"AI model trained successfully with accuracy: {accuracy:.3f}")
                except Exception as e:
                    logger.error(f"Failed to train AI model: {e}")
                    logger.warning("System will continue without trained model. Use /retrain endpoint to train later.")
            else:
                logger.warning("Cannot train model without mapping configurations. Add mappings and use /retrain endpoint.")
        
        logger.info("Application initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        return False

def create_sample_mapping_if_none():
    """Create a sample mapping if no mappings exist"""
    logger = logging.getLogger(__name__)
    
    if len(mapping_manager.get_all_mappings()) == 0:
        logger.info("No mappings found. Creating sample mapping...")
        
        sample_mapping = {
            "mapping_id": "sample_customers_mapping",
            "name": "Sample Customer Table Migration",
            "description": "Sample mapping created automatically for testing",
            "version": "1.0",
            "source": {
                "database": "source_system",
                "schema": "legacy",
                "table": "customers"
            },
            "target": {
                "database": "target_system",
                "schema": "modern",
                "table": "customers"
            },
            "column_mappings": {
                "cust_id": {
                    "target_column": "customer_id",
                    "transformation": "direct",
                    "source_type": "integer",
                    "target_type": "integer",
                    "nullable": False
                },
                "cust_name": {
                    "target_column": "full_name",
                    "transformation": "direct",
                    "source_type": "varchar",
                    "target_type": "varchar",
                    "nullable": False
                },
                "email_addr": {
                    "target_column": "email",
                    "transformation": "direct",
                    "source_type": "varchar",
                    "target_type": "varchar",
                    "nullable": True
                }
            },
            "validation_rules": {
                "row_count_tolerance": 0.01,
                "null_value_tolerance": 0.05
            }
        }
        
        if mapping_manager.add_mapping(sample_mapping):
            logger.info("Sample mapping created successfully")
        else:
            logger.error("Failed to create sample mapping")

def print_startup_banner():
    """Print application startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              Dynamic Data Mapping Validator                  ║
    ║                     AI-Powered Migration                     ║
    ║                    Validation System                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_access_info():
    """Print access information"""
    logger = logging.getLogger(__name__)
    
    app_port = os.getenv('APP_PORT', '8000')
    
    logger.info("=" * 60)
    logger.info("🚀 Dynamic Data Mapping Validator Started Successfully!")
    logger.info("=" * 60)
    logger.info(f"📱 Web Interface:      http://localhost:{app_port}")
    logger.info(f"📚 API Documentation: http://localhost:{app_port}/docs")
    logger.info(f"💗 Health Check:      http://localhost:{app_port}/status")
    logger.info("=" * 60)
    logger.info("📊 Database Connections:")
    logger.info(f"   Source DB: {os.getenv('SOURCE_DB_HOST')}:{os.getenv('SOURCE_DB_PORT')}")
    logger.info(f"   Target DB: {os.getenv('TARGET_DB_HOST')}:{os.getenv('TARGET_DB_PORT')}")
    logger.info("=" * 60)
    logger.info("🛠️  Management Commands:")
    logger.info("   View logs:     docker-compose logs -f validator")
    logger.info("   Restart:       docker-compose restart validator")
    logger.info("   Stop:          docker-compose down")
    logger.info("=" * 60)

def main():
    """Main application entry point"""
    # Print startup banner
    print_startup_banner()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Dynamic Data Mapping Validator...")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)
        
        # Initialize databases
        source_db, target_db = initialize_databases()
        if not source_db or not target_db:
            logger.error("Database initialization failed. Exiting.")
            sys.exit(1)
        
        # Create sample mapping if none exist
        create_sample_mapping_if_none()
        
        # Initialize application components
        if not initialize_application():
            logger.error("Application initialization failed. Exiting.")
            sys.exit(1)
        
        # Print access information
        print_access_info()
        
        # Start the web server
        import uvicorn
        
        app_host = os.getenv('APP_HOST', '0.0.0.0')
        app_port = int(os.getenv('APP_PORT', 8000))
        reload = os.getenv('RELOAD_ON_CHANGE', 'false').lower() == 'true'
        
        logger.info(f"Starting web server on {app_host}:{app_port}")
        
        uvicorn.run(
            app, 
            host=app_host, 
            port=app_port, 
            log_level=os.getenv('LOG_LEVEL', 'info').lower(),
            reload=reload,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping application...")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# =============================================================================
# Additional utility functions for debugging and maintenance
# =============================================================================

def check_system_health():
    """Check overall system health"""
    logger = logging.getLogger(__name__)
    
    health_status = {
        "timestamp": time.time(),
        "database_connections": {},
        "mapping_manager": {},
        "ai_model": {},
        "file_system": {}
    }
    
    try:
        # Check database connections
        source_db, target_db = initialize_databases()
        health_status["database_connections"]["source"] = source_db is not None
        health_status["database_connections"]["target"] = target_db is not None
        
        # Check mapping manager
        mappings_count = len(mapping_manager.get_all_mappings())
        health_status["mapping_manager"]["mappings_loaded"] = mappings_count
        health_status["mapping_manager"]["status"] = "healthy" if mappings_count > 0 else "warning"
        
        # Check AI model
        health_status["ai_model"]["trained"] = validator.is_trained
        health_status["ai_model"]["status"] = "healthy" if validator.is_trained else "needs_training"
        
        # Check file system
        required_dirs = ['/app/mappings', '/app/models', '/app/reports', '/app/logs']
        for dir_path in required_dirs:
            health_status["file_system"][dir_path] = os.path.exists(dir_path)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["error"] = str(e)
    
    return health_status

def run_diagnostic():
    """Run comprehensive system diagnostic"""
    logger = logging.getLogger(__name__)
    
    logger.info("Running system diagnostic...")
    
    # Environment check
    logger.info("Checking environment variables...")
    env_vars = [
        'SOURCE_DB_HOST', 'SOURCE_DB_USER', 'SOURCE_DB_NAME',
        'TARGET_DB_HOST', 'TARGET_DB_USER', 'TARGET_DB_NAME',
        'APP_HOST', 'APP_PORT', 'LOG_LEVEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask passwords
            if 'PASSWORD' in var:
                value = '*' * len(value)
            logger.info(f"  {var}: {value}")
        else:
            logger.warning(f"  {var}: NOT SET")
    
    # File system check
    logger.info("Checking file system...")
    paths_to_check = [
        '/app/mappings',
        '/app/models', 
        '/app/reports',
        '/app/logs',
        '/app/src'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len(os.listdir(path))
                logger.info(f"  {path}: Directory exists ({file_count} items)")
            else:
                logger.info(f"  {path}: File exists")
        else:
            logger.warning(f"  {path}: NOT FOUND")
    
    # Health check
    health = check_system_health()
    logger.info("System health summary:")
    logger.info(f"  Database connections: {health.get('database_connections', {})}")
    logger.info(f"  Mappings loaded: {health.get('mapping_manager', {}).get('mappings_loaded', 0)}")
    logger.info(f"  AI model trained: {health.get('ai_model', {}).get('trained', False)}")
    
    logger.info("Diagnostic complete.")

# Export main components for potential external use
__all__ = [
    'main',
    'setup_logging', 
    'validate_environment',
    'initialize_databases',
    'initialize_application',
    'check_system_health',
    'run_diagnostic'
]

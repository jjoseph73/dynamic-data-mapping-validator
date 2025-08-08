"""
FastAPI Main Application
Dynamic Data Mapping Validator - REST API Layer
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

# Import endpoint routers
from .endpoints import validation, mappings, model, health
from ..validation_engine import ValidationEngine
from ..mapping_manager import MappingManager
from ..database_connector import DatabaseConnector

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances (will be initialized in lifespan)
validation_engine = None
mapping_manager = None
db_connector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown"""
    global validation_engine, mapping_manager, db_connector
    
    # Startup
    logger.info("Starting Dynamic Data Mapping Validator API...")
    
    try:
        # Initialize database connector
        db_connector = DatabaseConnector()
        await db_connector.initialize()
        logger.info("Database connector initialized")
        
        # Initialize mapping manager
        mapping_manager = MappingManager()
        logger.info("Mapping manager initialized")
        
        # Initialize validation engine
        validation_engine = ValidationEngine(db_connector)
        logger.info("Validation engine initialized")
        
        # Load any existing model
        try:
            validation_engine.load_model()
            logger.info("Pre-trained model loaded successfully")
        except FileNotFoundError:
            logger.info("No pre-trained model found, will train on first request")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Dynamic Data Mapping Validator API...")
    if db_connector:
        await db_connector.close()
    logger.info("API shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="Dynamic Data Mapping Validator",
    description="AI-powered data mapping validation system for Oracle-to-PostgreSQL migrations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add after creating your FastAPI app
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get validation engine
async def get_validation_engine() -> ValidationEngine:
    if validation_engine is None:
        raise HTTPException(status_code=503, detail="Validation engine not initialized")
    return validation_engine

# Dependency to get mapping manager
async def get_mapping_manager() -> MappingManager:
    if mapping_manager is None:
        raise HTTPException(status_code=503, detail="Mapping manager not initialized")
    return mapping_manager

# Dependency to get database connector
async def get_db_connector() -> DatabaseConnector:
    if db_connector is None:
        raise HTTPException(status_code=503, detail="Database connector not initialized")
    return db_connector

# Include API routers
app.include_router(
    health.router,
    prefix="/api/health",
    tags=["health"]
)

app.include_router(
    validation.router,
    prefix="/api/validation",
    tags=["validation"],
    dependencies=[Depends(get_validation_engine)]
)

app.include_router(
    mappings.router,
    prefix="/api/mappings",
    tags=["mappings"],
    dependencies=[Depends(get_mapping_manager)]
)

app.include_router(
    model.router,
    prefix="/api/model",
    tags=["model"],
    dependencies=[Depends(get_validation_engine)]
)

# Mount static files
static_path = Path(__file__).parent.parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Root endpoint - serve web interface
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    html_path = Path(__file__).parent.parent / "web" / "templates" / "index.html"
    
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Dynamic Data Mapping Validator</title></head>
            <body>
                <h1>Dynamic Data Mapping Validator API</h1>
                <p>Welcome to the AI-powered data mapping validation system!</p>
                <p><a href="/api/docs">View API Documentation</a></p>
                <p><a href="/api/health/status">Check System Health</a></p>
            </body>
        </html>
        """)

# Add this route to serve the dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
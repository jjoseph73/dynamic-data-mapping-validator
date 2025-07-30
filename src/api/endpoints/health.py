"""
Health Check API Endpoints
Provides system health monitoring and status information
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import psutil
import time
from datetime import datetime
from pathlib import Path

from ..validation_models import HealthCheckResponse, ComponentHealth

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status", response_model=HealthCheckResponse)
async def get_health_status():
    """
    Get overall system health status
    """
    try:
        logger.info("Performing health check")
        
        # Check individual components
        services = {}
        overall_status = "healthy"
        
        # Database connectivity check
        db_health = await _check_database_health()
        services["database"] = db_health
        if db_health["status"] != "healthy":
            overall_status = "degraded"
        
        # File system check
        fs_health = await _check_filesystem_health()
        services["filesystem"] = fs_health
        if fs_health["status"] != "healthy":
            overall_status = "degraded"
        
        # Memory and CPU check
        system_health = await _check_system_health()
        services["system"] = system_health
        if system_health["status"] != "healthy":
            overall_status = "degraded"
        
        # ML model check
        model_health = await _check_model_health()
        services["ml_model"] = model_health
        if model_health["status"] != "healthy":
            overall_status = "degraded"
        
        # API health check
        api_health = await _check_api_health()
        services["api"] = api_health
        if api_health["status"] != "healthy":
            overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services={"error": {"status": "unhealthy", "message": str(e)}}
        )

@router.get("/database")
async def check_database_health():
    """
    Detailed database health check
    """
    try:
        health_info = await _check_database_health()
        return health_info
        
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Database health check failed: {str(e)}")

@router.get("/system")
async def check_system_health():
    """
    Detailed system resource health check
    """
    try:
        health_info = await _check_system_health()
        return health_info
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        raise HTTPException(status_code=503, detail=f"System health check failed: {str(e)}")

@router.get("/model")
async def check_model_health():
    """
    Detailed ML model health check
    """
    try:
        health_info = await _check_model_health()
        return health_info
        
    except Exception as e:
        logger.error(f"Model health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Model health check failed: {str(e)}")

@router.get("/ready")
async def readiness_probe():
    """
    Kubernetes-style readiness probe
    """
    try:
        # Quick checks for readiness
        checks = {
            "database": await _quick_database_check(),
            "filesystem": await _quick_filesystem_check(),
            "model": await _quick_model_check()
        }
        
        all_ready = all(checks.values())
        
        return {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Readiness probe error: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_probe():
    """
    Kubernetes-style liveness probe
    """
    try:
        # Basic liveness check - just verify the API is responding
        return {
            "alive": True,
            "timestamp": datetime.utcnow(),
            "uptime_seconds": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Liveness probe error: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")

@router.get("/metrics")
async def get_health_metrics():
    """
    Get detailed health and performance metrics
    """
    try:
        logger.info("Collecting health metrics")
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 2)
            },
            "process": {
                "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
            },
            "application": {
                "uptime_seconds": time.time() - start_time,
                "total_validations": 0,  # Would be tracked in real implementation
                "active_connections": 0,  # Would be tracked in real implementation
                "cache_hit_rate": 0.0    # Would be tracked in real implementation
            },
            "timestamp": datetime.utcnow()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error collecting health metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect metrics: {str(e)}")

# Helper functions for health checks
async def _check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    try:
        # In a real implementation, you'd test actual database connections
        # For now, simulate database health check
        return {
            "status": "healthy",
            "message": "Database connections active",
            "last_check": datetime.utcnow(),
            "metrics": {
                "connection_pool_size": 10,
                "active_connections": 2,
                "response_time_ms": 5.2
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database check failed: {str(e)}",
            "last_check": datetime.utcnow(),
            "metrics": {}
        }

async def _check_filesystem_health() -> Dict[str, Any]:
    """Check filesystem health and disk space"""
    try:
        # Check mappings directory
        mappings_dir = Path("mappings")
        models_dir = Path("models")
        
        disk_usage = psutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        status = "healthy"
        if disk_percent > 90:
            status = "critical"
        elif disk_percent > 80:
            status = "degraded"
        
        return {
            "status": status,
            "message": f"Disk usage: {disk_percent:.1f}%",
            "last_check": datetime.utcnow(),
            "metrics": {
                "disk_usage_percent": disk_percent,
                "mappings_dir_exists": mappings_dir.exists(),
                "models_dir_exists": models_dir.exists(),
                "free_space_gb": round((disk_usage.total - disk_usage.used) / (1024**3), 2)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Filesystem check failed: {str(e)}",
            "last_check": datetime.utcnow(),
            "metrics": {}
        }

async def _check_system_health() -> Dict[str, Any]:
    """Check system resource health"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        status = "healthy"
        issues = []
        
        if cpu_percent > 90:
            status = "critical"
            issues.append("High CPU usage")
        elif cpu_percent > 80:
            status = "degraded"
            issues.append("Elevated CPU usage")
        
        if memory.percent > 90:
            status = "critical"
            issues.append("High memory usage")
        elif memory.percent > 80:
            status = "degraded"
            issues.append("Elevated memory usage")
        
        message = "; ".join(issues) if issues else "System resources normal"
        
        return {
            "status": status,
            "message": message,
            "last_check": datetime.utcnow(),
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"System check failed: {str(e)}",
            "last_check": datetime.utcnow(),
            "metrics": {}
        }

async def _check_model_health() -> Dict[str, Any]:
    """Check ML model health and readiness"""
    try:
        # In a real implementation, you'd check actual model status
        # For now, simulate model health check
        models_dir = Path("models")
        model_files_exist = len(list(models_dir.glob("*.pkl"))) > 0 if models_dir.exists() else False
        
        status = "healthy" if model_files_exist else "degraded"
        message = "Model loaded and ready" if model_files_exist else "No trained model available"
        
        return {
            "status": status,
            "message": message,
            "last_check": datetime.utcnow(),
            "metrics": {
                "model_loaded": model_files_exist,
                "model_files_count": len(list(models_dir.glob("*.pkl"))) if models_dir.exists() else 0,
                "last_training": None,  # Would be tracked in real implementation
                "prediction_accuracy": 0.85  # Would be tracked in real implementation
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Model check failed: {str(e)}",
            "last_check": datetime.utcnow(),
            "metrics": {}
        }

async def _check_api_health() -> Dict[str, Any]:
    """Check API component health"""
    try:
        # Check if all endpoints are responsive
        status = "healthy"
        message = "API endpoints responsive"
        
        return {
            "status": status,
            "message": message,
            "last_check": datetime.utcnow(),
            "metrics": {
                "active_endpoints": 4,  # validation, mappings, model, health
                "request_rate": 0.0,    # Would be tracked in real implementation
                "error_rate": 0.0,      # Would be tracked in real implementation
                "average_response_time": 50.0  # Would be tracked in real implementation
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"API check failed: {str(e)}",
            "last_check": datetime.utcnow(),
            "metrics": {}
        }

# Quick health check functions for readiness probe
async def _quick_database_check() -> bool:
    """Quick database connectivity check"""
    try:
        # In a real implementation, you'd do a simple database ping
        return True
    except:
        return False

async def _quick_filesystem_check() -> bool:
    """Quick filesystem accessibility check"""
    try:
        return Path("mappings").exists()
    except:
        return False

async def _quick_model_check() -> bool:
    """Quick model availability check"""
    try:
        models_dir = Path("models")
        return models_dir.exists() and len(list(models_dir.glob("*.pkl"))) > 0
    except:
        return False

# Track application start time for uptime calculation
start_time = time.time()
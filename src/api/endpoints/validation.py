"""
Validation API Endpoints
Handles all validation-related REST API operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from ..validation_models import (
    ValidationRequest,
    BatchValidationRequest,
    ValidationResponse,
    BatchValidationResponse,
    ColumnValidationRequest,
    ColumnValidationResponse,
    ValidationHistoryResponse
)
from validation_engine import ValidationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection will be handled by main app
def get_validation_engine() -> ValidationEngine:
    # This will be overridden by the main app dependency
    pass

@router.post("/validate-column", response_model=ColumnValidationResponse)
async def validate_column_mapping(
    request: ColumnValidationRequest,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Validate a single column mapping configuration
    """
    try:
        logger.info(f"Validating column mapping: {request.source_column} -> {request.target_column}")
        
        result = await engine.validate_column_mapping(
            source_column=request.source_column,
            target_column=request.target_column,
            source_type=request.source_type,
            target_type=request.target_type,
            source_constraints=request.source_constraints,
            target_constraints=request.target_constraints,
            business_rules=request.business_rules or {},
            table_context=request.table_context or {}
        )
        
        return ColumnValidationResponse(
            success=True,
            validation_result=result,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Column validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/validate-mapping", response_model=ValidationResponse)
async def validate_table_mapping(
    request: ValidationRequest,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Validate a complete table mapping configuration
    """
    try:
        logger.info(f"Validating table mapping: {request.mapping_name}")
        
        result = await engine.validate_mapping(request.mapping_config)
        
        return ValidationResponse(
            success=True,
            mapping_name=request.mapping_name,
            validation_result=result,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Table mapping validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/validate-batch", response_model=BatchValidationResponse)
async def validate_mappings_batch(
    request: BatchValidationRequest,
    background_tasks: BackgroundTasks,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Validate multiple mapping configurations in batch
    """
    try:
        logger.info(f"Starting batch validation for {len(request.mappings)} mappings")
        
        # For large batches, run in background
        if len(request.mappings) > 5:
            # Start background task
            task_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            background_tasks.add_task(
                _run_batch_validation,
                engine,
                request.mappings,
                task_id
            )
            
            return BatchValidationResponse(
                success=True,
                task_id=task_id,
                status="processing",
                message=f"Batch validation started for {len(request.mappings)} mappings",
                timestamp=datetime.utcnow()
            )
        else:
            # Run synchronously for small batches
            results = await engine.validate_mapping_batch(request.mappings)
            
            return BatchValidationResponse(
                success=True,
                status="completed",
                results=results,
                timestamp=datetime.utcnow()
            )
        
    except Exception as e:
        logger.error(f"Batch validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")

@router.get("/batch-status/{task_id}")
async def get_batch_status(
    task_id: str,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get status of a batch validation task
    """
    try:
        # In a real implementation, you'd track batch tasks in a database or cache
        # For now, return a simple response
        return {
            "task_id": task_id,
            "status": "completed",  # This would be dynamic based on actual task status
            "message": "Batch validation completed",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")

@router.get("/history", response_model=List[ValidationHistoryResponse])
async def get_validation_history(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    mapping_name: Optional[str] = None,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get validation history with optional filtering
    """
    try:
        # In a real implementation, this would query a database
        # For now, return empty list as history tracking would need to be implemented
        logger.info(f"Retrieving validation history (limit: {limit}, offset: {offset})")
        
        return []  # Placeholder - implement actual history retrieval
        
    except Exception as e:
        logger.error(f"Error retrieving validation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@router.delete("/history/{validation_id}")
async def delete_validation_result(
    validation_id: str,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Delete a specific validation result from history
    """
    try:
        logger.info(f"Deleting validation result: {validation_id}")
        
        # Placeholder - implement actual deletion
        return {"message": f"Validation result {validation_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting validation result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete validation result: {str(e)}")

@router.get("/statistics")
async def get_validation_statistics(
    days: int = Query(30, ge=1, le=365),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get validation statistics and metrics
    """
    try:
        logger.info(f"Retrieving validation statistics for last {days} days")
        
        # Placeholder statistics - implement actual metrics collection
        stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_confidence_score": 0.0,
            "common_issues": [],
            "validation_trends": [],
            "period_days": days,
            "timestamp": datetime.utcnow()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving validation statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

async def _run_batch_validation(
    engine: ValidationEngine,
    mappings: List[Dict[str, Any]],
    task_id: str
):
    """
    Background task for batch validation
    """
    try:
        logger.info(f"Running batch validation task {task_id}")
        results = await engine.validate_mapping_batch(mappings)
        
        # In a real implementation, you'd store results in a database or cache
        logger.info(f"Batch validation task {task_id} completed with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Batch validation task {task_id} failed: {e}")
        # In a real implementation, you'd update task status to failed
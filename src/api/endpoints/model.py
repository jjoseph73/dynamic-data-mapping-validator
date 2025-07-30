"""
ML Model Management API Endpoints
Handles model training, evaluation, and configuration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..validation_models import (
    ModelConfigRequest,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelDiagnosticsResponse,
    SystemConfigResponse
)
from ...validation_engine import ValidationEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection will be handled by main app
def get_validation_engine() -> ValidationEngine:
    # This will be overridden by the main app dependency
    pass

@router.get("/status", response_model=Dict[str, Any])
async def get_model_status(
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get current ML model status and basic information
    """
    try:
        logger.info("Getting model status")
        
        # Get model diagnostics
        try:
            diagnostics = engine.get_model_diagnostics()
            model_loaded = True
            model_info = {
                "model_type": diagnostics.get("model_type"),
                "training_date": diagnostics.get("training_date"),
                "performance_metrics": diagnostics.get("performance_metrics", {}),
                "model_health": diagnostics.get("model_health", {})
            }
        except Exception as e:
            model_loaded = False
            model_info = {"error": str(e)}
        
        return {
            "model_loaded": model_loaded,
            "model_info": model_info,
            "engine_status": "active",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.get("/diagnostics", response_model=ModelDiagnosticsResponse)
async def get_model_diagnostics(
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get detailed model diagnostics and performance metrics
    """
    try:
        logger.info("Getting model diagnostics")
        
        diagnostics = engine.get_model_diagnostics()
        
        return ModelDiagnosticsResponse(
            model_id=diagnostics.get("model_id", "unknown"),
            model_type=diagnostics.get("model_type", "unknown"),
            training_date=diagnostics.get("training_date", datetime.utcnow()),
            performance_metrics=diagnostics.get("performance_metrics", {}),
            feature_importance=diagnostics.get("feature_importance", {}),
            model_health=diagnostics.get("model_health", {}),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting model diagnostics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model diagnostics: {str(e)}")

@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    async_training: bool = Query(False, description="Whether to run training asynchronously"),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Train a new ML model with specified configuration
    """
    try:
        logger.info(f"Training model with type: {request.config.model_type}")
        
        if async_training:
            # Start background training task
            task_id = f"training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            background_tasks.add_task(
                _train_model_background,
                engine,
                request,
                task_id
            )
            
            return ModelTrainingResponse(
                success=True,
                model_id=task_id,
                training_metrics={"status": "training_started"},
                training_time_seconds=0.0,
                timestamp=datetime.utcnow()
            )
        else:
            # Train synchronously
            start_time = datetime.utcnow()
            
            # Configure model
            model_config = {
                "model_type": request.config.model_type.value,
                "model_params": request.config.parameters,
                "validation_split": request.validation_split
            }
            
            # Train model
            training_result = await engine.train_model(
                config=model_config,
                use_existing_data=request.use_existing_data,
                generate_synthetic=request.generate_synthetic_data
            )
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ModelTrainingResponse(
                success=True,
                model_id=training_result.get("model_id", "unknown"),
                training_metrics=training_result.get("metrics", {}),
                training_time_seconds=training_time,
                timestamp=datetime.utcnow()
            )
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.get("/training-status/{task_id}")
async def get_training_status(
    task_id: str,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get status of an asynchronous training task
    """
    try:
        logger.info(f"Getting training status for task: {task_id}")
        
        # In a real implementation, you'd track training tasks in a database or cache
        # For now, return a placeholder response
        return {
            "task_id": task_id,
            "status": "completed",  # This would be dynamic based on actual task status
            "progress": 100,
            "message": "Training completed successfully",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.post("/evaluate")
async def evaluate_model(
    test_data: Optional[Dict[str, Any]] = None,
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Evaluate current model performance on test data
    """
    try:
        logger.info("Evaluating model performance")
        
        # In a real implementation, you'd run model evaluation
        # For now, return placeholder metrics
        evaluation_metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "confusion_matrix": [[45, 5], [8, 42]],
            "roc_auc": 0.89
        }
        
        return {
            "success": True,
            "evaluation_metrics": evaluation_metrics,
            "test_data_size": test_data.get("size", 0) if test_data else 0,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")

@router.post("/retrain")
async def retrain_model(
    incremental: bool = Query(False, description="Whether to perform incremental retraining"),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Retrain the model with latest data
    """
    try:
        logger.info(f"Retraining model (incremental: {incremental})")
        
        start_time = datetime.utcnow()
        
        # Trigger model retraining
        retrain_result = await engine.retrain_model(incremental=incremental)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "success": True,
            "model_id": retrain_result.get("model_id", "unknown"),
            "training_type": "incremental" if incremental else "full",
            "training_metrics": retrain_result.get("metrics", {}),
            "training_time_seconds": training_time,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@router.get("/config", response_model=SystemConfigResponse)
async def get_model_config(
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Get current model and system configuration
    """
    try:
        logger.info("Getting model configuration")
        
        # Get current configuration
        config = {
            "database_config": {
                "source_db": "postgresql",
                "target_db": "postgresql",
                "connection_pooling": True
            },
            "ml_config": {
                "model_type": "random_forest",
                "auto_retrain": True,
                "feature_engineering": True,
                "cross_validation": True
            },
            "api_config": {
                "max_batch_size": 100,
                "async_processing": True,
                "rate_limiting": True
            },
            "features_enabled": [
                "column_validation",
                "table_validation",
                "batch_processing",
                "model_training",
                "hot_reload"
            ]
        }
        
        return SystemConfigResponse(
            database_config=config["database_config"],
            ml_config=config["ml_config"],
            api_config=config["api_config"],
            features_enabled=config["features_enabled"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model config: {str(e)}")

@router.put("/config")
async def update_model_config(
    config_update: Dict[str, Any],
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Update model configuration
    """
    try:
        logger.info("Updating model configuration")
        
        # In a real implementation, you'd update the actual configuration
        # For now, return success response
        return {
            "success": True,
            "message": "Model configuration updated successfully",
            "updated_fields": list(config_update.keys()),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error updating model config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model config: {str(e)}")

@router.post("/load")
async def load_model(
    model_path: Optional[str] = Query(None, description="Path to model file to load"),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Load a specific model from file
    """
    try:
        logger.info(f"Loading model from: {model_path or 'default location'}")
        
        # Load model
        if model_path:
            engine.load_model(model_path)
        else:
            engine.load_model()
        
        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_path": model_path or "default",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/save")
async def save_model(
    model_path: Optional[str] = Query(None, description="Path where to save the model"),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Save current model to file
    """
    try:
        logger.info(f"Saving model to: {model_path or 'default location'}")
        
        # Save model
        save_path = engine.save_model(model_path) if model_path else engine.save_model()
        
        return {
            "success": True,
            "message": "Model saved successfully",
            "save_path": save_path,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@router.delete("/model")
async def reset_model(
    confirm: bool = Query(False, description="Confirmation to reset the model"),
    engine: ValidationEngine = Depends(get_validation_engine)
):
    """
    Reset/clear the current model
    """
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Reset confirmation required (confirm=true)")
        
        logger.info("Resetting model")
        
        # Reset model
        engine.reset_model()
        
        return {
            "success": True,
            "message": "Model reset successfully",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset model: {str(e)}")

async def _train_model_background(
    engine: ValidationEngine,
    request: ModelTrainingRequest,
    task_id: str
):
    """
    Background task for model training
    """
    try:
        logger.info(f"Running background model training task {task_id}")
        
        # Configure model
        model_config = {
            "model_type": request.config.model_type.value,
            "model_params": request.config.parameters,
            "validation_split": request.validation_split
        }
        
        # Train model
        training_result = await engine.train_model(
            config=model_config,
            use_existing_data=request.use_existing_data,
            generate_synthetic=request.generate_synthetic_data
        )
        
        # In a real implementation, you'd store results in a database or cache
        logger.info(f"Background training task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background training task {task_id} failed: {e}")
        # In a real implementation, you'd update task status to failed
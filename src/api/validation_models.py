"""
Pydantic Models for API Request/Response Validation
Dynamic Data Mapping Validator
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
class ValidationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"

# Base Models
class ValidationIssueModel(BaseModel):
    """Model for validation issues"""
    issue_type: str
    severity: IssueSeverity
    description: str
    column: Optional[str] = None
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidationMetricsModel(BaseModel):
    """Model for validation metrics"""
    type_compatibility_score: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    business_rule_score: float = Field(ge=0.0, le=1.0)
    performance_impact_score: float = Field(ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)

class ValidationResultModel(BaseModel):
    """Model for validation results"""
    is_valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    predicted_success_probability: float = Field(ge=0.0, le=1.0)
    execution_time_seconds: float = Field(ge=0.0)
    metrics: ValidationMetricsModel
    issues: List[ValidationIssueModel] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Column Validation Models
class ColumnValidationRequest(BaseModel):
    """Request model for column validation"""
    source_column: str = Field(min_length=1, max_length=128)
    target_column: str = Field(min_length=1, max_length=128)
    source_type: str = Field(min_length=1, max_length=64)
    target_type: str = Field(min_length=1, max_length=64)
    source_constraints: Dict[str, Any] = Field(default_factory=dict)
    target_constraints: Dict[str, Any] = Field(default_factory=dict)
    business_rules: Optional[Dict[str, Any]] = None
    table_context: Optional[Dict[str, Any]] = None

    @validator('source_column', 'target_column')
    def validate_column_names(cls, v):
        if not v.strip():
            raise ValueError('Column name cannot be empty')
        return v.strip()

class ColumnValidationResponse(BaseModel):
    """Response model for column validation"""
    success: bool
    validation_result: ValidationResultModel
    timestamp: datetime
    error_message: Optional[str] = None

# Table Validation Models
class MappingConfigModel(BaseModel):
    """Model for mapping configuration"""
    source_table: str = Field(min_length=1, max_length=128)
    target_table: str = Field(min_length=1, max_length=128)
    column_mappings: List[Dict[str, Any]] = Field(min_items=1)
    business_rules: Dict[str, Any] = Field(default_factory=dict)
    transformation_rules: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('column_mappings')
    def validate_column_mappings(cls, v):
        for mapping in v:
            if 'source_column' not in mapping or 'target_column' not in mapping:
                raise ValueError('Each column mapping must have source_column and target_column')
        return v

class ValidationRequest(BaseModel):
    """Request model for table mapping validation"""
    mapping_name: str = Field(min_length=1, max_length=128)
    mapping_config: MappingConfigModel

class ValidationResponse(BaseModel):
    """Response model for table mapping validation"""
    success: bool
    mapping_name: str
    validation_result: ValidationResultModel
    timestamp: datetime
    error_message: Optional[str] = None

# Batch Validation Models
class BatchValidationRequest(BaseModel):
    """Request model for batch validation"""
    mappings: List[Dict[str, Any]] = Field(min_items=1, max_items=100)
    options: Dict[str, Any] = Field(default_factory=dict)

    @validator('mappings')
    def validate_mappings_format(cls, v):
        for mapping in v:
            if 'name' not in mapping or 'config' not in mapping:
                raise ValueError('Each mapping must have name and config fields')
        return v

class BatchValidationResponse(BaseModel):
    """Response model for batch validation"""
    success: bool
    task_id: Optional[str] = None
    status: ValidationStatus
    message: str
    results: Optional[List[ValidationResultModel]] = None
    timestamp: datetime
    error_message: Optional[str] = None

# History Models
class ValidationHistoryResponse(BaseModel):
    """Response model for validation history"""
    validation_id: str
    mapping_name: str
    validation_type: str  # 'column' or 'table' or 'batch'
    result: ValidationResultModel
    timestamp: datetime
    user_id: Optional[str] = None

# Mapping Management Models
class MappingListResponse(BaseModel):
    """Response model for mapping list"""
    mappings: List[Dict[str, Any]]
    total_count: int
    timestamp: datetime

class MappingCreateRequest(BaseModel):
    """Request model for creating mappings"""
    name: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    config: MappingConfigModel
    description: Optional[str] = Field(max_length=500)
    tags: List[str] = Field(default_factory=list, max_items=10)

    @validator('tags')
    def validate_tags(cls, v):
        return [tag.strip().lower() for tag in v if tag.strip()]

class MappingUpdateRequest(BaseModel):
    """Request model for updating mappings"""
    config: Optional[MappingConfigModel] = None
    description: Optional[str] = Field(max_length=500)
    tags: Optional[List[str]] = Field(max_items=10)

class MappingResponse(BaseModel):
    """Response model for mapping operations"""
    success: bool
    mapping_name: str
    message: str
    mapping_config: Optional[Dict[str, Any]] = None
    timestamp: datetime
    error_message: Optional[str] = None

# Model Management Models
class ModelConfigRequest(BaseModel):
    """Request model for ML model configuration"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    parameters: Dict[str, Any] = Field(default_factory=dict)
    training_options: Dict[str, Any] = Field(default_factory=dict)

class ModelTrainingRequest(BaseModel):
    """Request model for model training"""
    config: ModelConfigRequest
    use_existing_data: bool = True
    generate_synthetic_data: bool = False
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)

class ModelTrainingResponse(BaseModel):
    """Response model for model training"""
    success: bool
    model_id: str
    training_metrics: Dict[str, float]
    training_time_seconds: float
    timestamp: datetime
    error_message: Optional[str] = None

class ModelDiagnosticsResponse(BaseModel):
    """Response model for model diagnostics"""
    model_id: str
    model_type: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model_health: Dict[str, Any]
    timestamp: datetime

# Health Check Models
class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, Dict[str, Any]]
    
class ComponentHealth(BaseModel):
    """Model for individual component health"""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    last_check: datetime
    metrics: Dict[str, Any] = Field(default_factory=dict)

# Error Models
class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None

# Configuration Models
class SystemConfigResponse(BaseModel):
    """Response model for system configuration"""
    database_config: Dict[str, Any]
    ml_config: Dict[str, Any]
    api_config: Dict[str, Any]
    features_enabled: List[str]
    timestamp: datetime
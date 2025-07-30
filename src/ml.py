except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_status = ModelStatus.ERROR
            return False
    
    def validate_column_mapping(self, source_col_stats: Dict[str, Any], 
                               target_col_stats: Dict[str, Any], 
                               mapping_info: Dict[str, Any],
                               validation_type: str = 'data_type') -> ValidationResult:
        """
        Validate a specific column mapping using the trained AI model
        
        Args:
            source_col_stats: Source column statistics
            target_col_stats: Target column statistics
            mapping_info: Mapping configuration for this column
            validation_type: Type of validation to perform
            
        Returns:
            ValidationResult with detailed analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        try:
            # Calculate metrics
            row_count_diff = ((target_col_stats['row_count'] - source_col_stats['row_count']) 
                            / source_col_stats['row_count'])
            null_diff = target_col_stats['null_percentage'] - source_col_stats['null_percentage']
            
            # Get data types
            source_type = mapping_info.get('source_type', source_col_stats['data_type'])
            target_type = mapping_info.get('target_type', target_col_stats['data_type'])
            transformation = mapping_info.get('transformation', 'direct')
            
            # Calculate compatibility and quality scores
            type_compat = self._calculate_type_compatibility(source_type, target_type)
            data_quality_score = self._calculate_data_quality_score(transformation, mapping_info)
            business_rule_score = self._calculate_business_rule_score(mapping_info)
            
            # Prepare feature data for prediction
            feature_data = pd.DataFrame([{
                'source_data_type': source_type,
                'target_data_type': target_type,
                'transformation': transformation,
                'validation_type': validation_type,
                'row_count_diff_pct': row_count_diff,
                'null_percentage_diff': null_diff,
                'type_compatibility_score': type_compat,
                'data_quality_score': data_quality_score,
                'business_rule_score': business_rule_score,
                'nullable': mapping_info.get('nullable', True)
            }])
            
            # Prepare features
            X = self.prepare_features(feature_data)
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get prediction result
            outcome = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            # Create metrics object
            metrics = ValidationMetrics(
                row_count_difference_pct=row_count_diff * 100,
                null_percentage_difference=null_diff * 100,
                type_compatibility_score=type_compat,
                data_quality_score=data_quality_score,
                business_rule_score=business_rule_score
            )
            
            # Additional validation details
            details = {
                'prediction_probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                'feature_values': X.iloc[0].to_dict(),
                'source_stats': source_col_stats,
                'target_stats': target_col_stats,
                'validation_type': validation_type
            }
            
            return ValidationResult(
                validation_result=ValidationStatus(outcome),
                confidence=confidence,
                metrics=metrics,
                mapping_info=mapping_info,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Column validation failed: {e}")
            return ValidationResult(
                validation_result=ValidationStatus.ERROR,
                confidence=0.0,
                metrics=ValidationMetrics(0, 0, 0),
                mapping_info=mapping_info,
                details={'error': str(e)}
            )
    
    def validate_mapping(self, mapping_id: str, source_db: DatabaseConnector, 
                        target_db: DatabaseConnector) -> Dict[str, Any]:
        """
        Validate a complete mapping between source and target tables
        
        Args:
            mapping_id: Mapping identifier
            source_db: Source database connector
            target_db: Target database connector
            
        Returns:
            Complete validation results
        """
        try:
            # Get mapping configuration
            mapping_config = self.mapping_manager.get_mapping(mapping_id)
            if not mapping_config:
                raise ValueError(f"Mapping {mapping_id} not found")
            
            # Get table information
            source_info = mapping_config['source']
            target_info = mapping_config['target']
            
            # Get table statistics
            logger.info(f"Collecting statistics for {mapping_id}")
            source_stats = source_db.get_table_stats(source_info['schema'], source_info['table'])
            target_stats = target_db.get_table_stats(target_info['schema'], target_info['table'])
            
            # Initialize results
            results = {
                'mapping_id': mapping_id,
                'mapping_name': mapping_config.get('name', 'Unknown'),
                'validation_timestamp': datetime.now().isoformat(),
                'overall_result': ValidationStatus.PASS.value,
                'overall_confidence': 0.0,
                'table_level_metrics': {},
                'column_validations': {},
                'summary': {
                    'total_columns': 0,
                    'passed': 0,
                    'failed': 0,
                    'warnings': 0,
                    'errors': 0
                },
                'recommendations': []
            }
            
            # Table-level validation
            table_metrics = self._validate_table_level(source_stats, target_stats, mapping_config)
            results['table_level_metrics'] = table_metrics
            
            # Column-level validation
            confidences = []
            column_mappings = mapping_config.get('column_mappings', {})
            
            for source_col, mapping_info in column_mappings.items():
                target_col = mapping_info.get('target_column')
                
                if (source_col in source_stats['columns'] and 
                    target_col in target_stats['columns']):
                    
                    try:
                        validation_result = self.validate_column_mapping(
                            source_stats['columns'][source_col],
                            target_stats['columns'][target_col],
                            mapping_info
                        )
                        
                        results['column_validations'][f"{source_col}->{target_col}"] = {
                            'validation_result': validation_result.validation_result.value,
                            'confidence': validation_result.confidence,
                            'metrics': {
                                'row_count_difference_pct': validation_result.metrics.row_count_difference_pct,
                                'null_percentage_difference': validation_result.metrics.null_percentage_difference,
                                'type_compatibility_score': validation_result.metrics.type_compatibility_score,
                                'data_quality_score': validation_result.metrics.data_quality_score,
                                'business_rule_score': validation_result.metrics.business_rule_score
                            },
                            'mapping_info': validation_result.mapping_info,
                            'details': validation_result.details
                        }
                        
                        confidences.append(validation_result.confidence)
                        
                        # Update summary
                        if validation_result.validation_result == ValidationStatus.PASS:
                            results['summary']['passed'] += 1
                        elif validation_result.validation_result == ValidationStatus.FAIL:
                            results['summary']['failed'] += 1
                            results['overall_result'] = ValidationStatus.FAIL.value
                        elif validation_result.validation_result == ValidationStatus.WARNING:
                            results['summary']['warnings'] += 1
                        else:
                            results['summary']['errors'] += 1
                            results['overall_result'] = ValidationStatus.ERROR.value
                        
                        results['summary']['total_columns'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to validate column {source_col}: {e}")
                        results['summary']['errors'] += 1
                        results['column_validations'][f"{source_col}->{target_col}"] = {
                            'validation_result': ValidationStatus.ERROR.value,
                            'error': str(e)
                        }
                else:
                    missing_cols = []
                    if source_col not in source_stats['columns']:
                        missing_cols.append(f"source column '{source_col}'")
                    if target_col not in target_stats['columns']:
                        missing_cols.append(f"target column '{target_col}'")
                    
                    error_msg = f"Missing: {', '.join(missing_cols)}"
                    results['column_validations'][f"{source_col}->{target_col}"] = {
                        'validation_result': ValidationStatus.ERROR.value,
                        'error': error_msg
                    }
                    results['summary']['errors'] += 1
            
            # Calculate overall confidence
            if confidences:
                results['overall_confidence'] = float(np.mean(confidences))
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Mapping validation failed for {mapping_id}: {e}")
            return {
                'mapping_id': mapping_id,
                'validation_timestamp': datetime.now().isoformat(),
                'overall_result': ValidationStatus.ERROR.value,
                'error': str(e)
            }
    
    def _validate_table_level(self, source_stats: Dict[str, Any], 
                             target_stats: Dict[str, Any], 
                             mapping_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform table-level validation checks
        
        Args:
            source_stats: Source table statistics
            target_stats: Target table statistics
            mapping_config: Mapping configuration
            
        Returns:
            Table-level validation metrics
        """
        validation_rules = mapping_config.get('validation_rules', {})
        row_count_tolerance = validation_rules.get('row_count_tolerance', 0.01)
        
        source_row_count = source_stats['row_count']
        target_row_count = target_stats['row_count']
        
        # Row count validation
        row_count_diff_pct = ((target_row_count - source_row_count) / source_row_count * 100 
                             if source_row_count > 0 else 0)
        row_count_within_tolerance = abs(row_count_diff_pct) <= (row_count_tolerance * 100)
        
        # Column count validation
        source_column_count = len(source_stats['columns'])
        mapped_column_count = len(mapping_config.get('column_mappings', {}))
        coverage_pct = (mapped_column_count / source_column_count * 100 
                       if source_column_count > 0 else 0)
        
        return {
            'source_row_count': source_row_count,
            'target_row_count': target_row_count,
            'row_count_difference_pct': row_count_diff_pct,
            'row_count_within_tolerance': row_count_within_tolerance,
            'source_column_count': source_column_count,
            'mapped_column_count': mapped_column_count,
            'mapping_coverage_pct': coverage_pct,
            'row_count_tolerance': row_count_tolerance * 100
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results
        
        Args:
            validation_results: Complete validation results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Table-level recommendations
        table_metrics = validation_results.get('table_level_metrics', {})
        
        if not table_metrics.get('row_count_within_tolerance', True):
            row_diff = table_metrics.get('row_count_difference_pct', 0)
            if row_diff > 0:
                recommendations.append(f"Target table has {row_diff:.1f}% more rows than source. Investigate data duplication or transformation logic.")
            else:
                recommendations.append(f"Target table has {abs(row_diff):.1f}% fewer rows than source. Check for data filtering or failed migrations.")
        
        coverage = table_metrics.get('mapping_coverage_pct', 100)
        if coverage < 90:
            recommendations.append(f"Only {coverage:.1f}% of source columns are mapped. Consider mapping additional columns or documenting intentional exclusions.")
        
        # Column-level recommendations
        summary = validation_results.get('summary', {})
        failed_columns = summary.get('failed', 0)
        total_columns = summary.get('total_columns', 1)
        
        if failed_columns > 0:
            failure_rate = (failed_columns / total_columns) * 100
            if failure_rate > 20:
                recommendations.append(f"High failure rate ({failure_rate:.1f}%) suggests systematic issues. Review mapping configuration and data transformation logic.")
            
            # Analyze specific failure patterns
            column_validations = validation_results.get('column_validations', {})
            type_compatibility_issues = []
            null_value_issues = []
            
            for col_mapping, result in column_validations.items():
                if result.get('validation_result') == 'FAIL':
                    metrics = result.get('metrics', {})
                    if metrics.get('type_compatibility_score', 1.0) < 0.7:
                        type_compatibility_issues.append(col_mapping)
                    if abs(metrics.get('null_percentage_difference', 0)) > 5:
                        null_value_issues.append(col_mapping)
            
            if type_compatibility_issues:
                recommendations.append(f"Type compatibility issues in: {', '.join(type_compatibility_issues[:3])}{'...' if len(type_compatibility_issues) > 3 else ''}")
            
            if null_value_issues:
                recommendations.append(f"Null value handling issues in: {', '.join(null_value_issues[:3])}{'...' if len(null_value_issues) > 3 else ''}")
        
        # Performance recommendations
        overall_confidence = validation_results.get('overall_confidence', 1.0)
        if overall_confidence < 0.8:
            recommendations.append("Low confidence scores suggest the model needs more training data or the mapping configuration needs refinement.")
        
        if not recommendations:
            recommendations.append("Validation completed successfully with no major issues detected.")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'status': self.model_status.value,
            'metrics': self.model_metrics,
            'training_history': self.training_history,
            'feature_encoders': {k: list(v.classes_) for k, v in self.feature_encoders.items()},
            'configuration': {
                'max_training_samples': self.max_training_samples,
                'accuracy_threshold': self.model_accuracy_threshold,
                'auto_retrain': self.auto_retrain
            }
        }
    
    def retrain_if_needed(self) -> bool:
        """
        Retrain the model if conditions are met
        
        Returns:
            True if retraining was performed, False otherwise
        """
        current_mappings = len(self.mapping_manager.get_all_mappings())
        
        # Check if retraining is needed
        if not self.is_trained:
            logger.info("Model not trained, initiating training...")
            self.train_model()
            return True
        
        if self.training_history:
            last_training = self.training_history[-1]
            last_mapping_count = last_training.get('mapping_count', 0)
            
            # Retrain if mapping count increased significantly
            if current_mappings > last_mapping_count * 1.2:
                logger.info(f"Mapping count increased significantly ({last_mapping_count} -> {current_mappings}), retraining...")
                self.train_model()
                return True
        
        return False

# Pydantic models for API
class MappingRequest(BaseModel):
    mapping_data: Dict[str, Any] = Field(..., description="Mapping configuration data")

class ValidationRequest(BaseModel):
    mapping_ids: List[str] = Field(..., description="List of mapping IDs to validate")

class BulkValidationRequest(BaseModel):
    mapping_ids: Optional[List[str]] = Field(None, description="Specific mapping IDs (all if not provided)")
    include_details: bool = Field(True, description="Include detailed validation information")
    parallel: bool = Field(False, description="Run validations in parallel")

# FastAPI Application Setup
app = FastAPI(
    title="Dynamic Data Mapping Validator",
    description="AI-powered validation system for database migration mappings",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized in main.py)
mapping_manager = MappingManager()
validator = DynamicDataMappingValidator(mapping_manager)

# Database connector instances (will be initialized as needed)
_source_db = None
_target_db = None

def get_database_connectors():
    """Get database connector instances"""
    global _source_db, _target_db
    
    if _source_db is None:
        _source_db = DatabaseConnector(
            host=os.getenv('SOURCE_DB_HOST', 'source_db'),
            port=int(os.getenv('SOURCE_DB_PORT', 5432)),
            database=os.getenv('SOURCE_DB_NAME', 'source_system'),
            username=os.getenv('SOURCE_DB_USER', 'source_user'),
            password=os.getenv('SOURCE_DB_PASSWORD', 'source_pass')
        )
    
    if _target_db is None:
        _target_db = DatabaseConnector(
            host=os.getenv('TARGET_DB_HOST', 'target_db'),
            port=int(os.getenv('TARGET_DB_PORT', 5432)),
            database=os.getenv('TARGET_DB_NAME', 'target_system'),
            username=os.getenv('TARGET_DB_USER', 'target_user'),
            password=os.getenv('TARGET_DB_PASSWORD', 'target_pass')
        )
    
    return _source_db, _target_db

# API Routes would continue here...
# (Due to length constraints, I'll provide the main FastAPI routes in the next section)

# Export main components
__all__ = [
    'DynamicDataMappingValidator',
    'ValidationStatus',
    'ModelStatus', 
    'ValidationMetrics',
    'ValidationResult',
    'app',
    'mapping_manager',
    'validator'
]# =============================================================================
# src/dynamic_validator.py - AI-Powered Dynamic Validation System
# Dynamic Data Mapping Validator
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Union

# Internal imports
from mapping_manager import MappingManager, MappingStatus
from database_connector import DatabaseConnector

# Configure logging
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    ERROR = "ERROR"

class ModelStatus(Enum):
    """AI model status"""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"

@dataclass
class ValidationMetrics:
    """Metrics for a validation result"""
    row_count_difference_pct: float
    null_percentage_difference: float
    type_compatibility_score: float
    data_quality_score: float = 0.0
    business_rule_score: float = 0.0

@dataclass
class ValidationResult:
    """Complete validation result"""
    validation_result: ValidationStatus
    confidence: float
    metrics: ValidationMetrics
    mapping_info: Dict[str, Any]
    details: Dict[str, Any] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class DynamicDataMappingValidator:
    """
    AI-powered validation system that learns from mapping configurations
    and validates data migrations with high accuracy and adaptability.
    """
    
    def __init__(self, mapping_manager: MappingManager):
        """
        Initialize the dynamic validator
        
        Args:
            mapping_manager: MappingManager instance for accessing configurations
        """
        self.mapping_manager = mapping_manager
        
        # AI Model components
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        
        # Model state
        self.model_status = ModelStatus.NOT_TRAINED
        self.model_metrics = {}
        self.training_history = []
        
        # Configuration
        self.max_training_samples = int(os.getenv('MAX_TRAINING_SAMPLES', 5000))
        self.model_accuracy_threshold = float(os.getenv('MODEL_ACCURACY_THRESHOLD', 0.85))
        self.auto_retrain = os.getenv('MODEL_AUTO_RETRAIN', 'true').lower() == 'true'
        
        # Setup change watcher for automatic retraining
        if self.auto_retrain:
            self.mapping_manager.add_change_watcher(self._on_mapping_change)
        
        logger.info("DynamicDataMappingValidator initialized")
    
    @property
    def is_trained(self) -> bool:
        """Check if the model is trained and ready"""
        return self.model_status == ModelStatus.TRAINED
    
    def _on_mapping_change(self, event: str, mapping_id: str):
        """Handle mapping changes for automatic retraining"""
        if event in ['mapping_added', 'mapping_updated', 'bulk_reload'] and self.auto_retrain:
            logger.info(f"Mapping change detected ({event}). Scheduling model retraining...")
            # Note: In a production system, you might want to debounce this
            # or schedule retraining during off-peak hours
            try:
                self.train_model()
            except Exception as e:
                logger.error(f"Automatic retraining failed: {e}")
    
    def generate_training_data_from_mappings(self, samples_per_mapping: int = None) -> pd.DataFrame:
        """
        Generate comprehensive training data based on all available mappings
        
        Args:
            samples_per_mapping: Number of samples per mapping (auto-calculated if None)
            
        Returns:
            DataFrame with training data
        """
        mappings = self.mapping_manager.get_all_mappings()
        if not mappings:
            raise ValueError("No mappings available for training data generation")
        
        # Auto-calculate samples per mapping
        if samples_per_mapping is None:
            samples_per_mapping = max(50, min(500, self.max_training_samples // len(mappings)))
        
        logger.info(f"Generating training data: {len(mappings)} mappings × {samples_per_mapping} samples")
        
        all_training_data = []
        
        for mapping_id, mapping_config in mappings.items():
            try:
                mapping_data = self._generate_mapping_samples(
                    mapping_id, mapping_config, samples_per_mapping
                )
                all_training_data.extend(mapping_data)
            except Exception as e:
                logger.warning(f"Failed to generate data for mapping {mapping_id}: {e}")
        
        if not all_training_data:
            raise ValueError("No training data generated from mappings")
        
        df = pd.DataFrame(all_training_data)
        logger.info(f"Generated {len(df)} training samples from {len(mappings)} mappings")
        
        return df
    
    def _generate_mapping_samples(self, mapping_id: str, mapping_config: Dict[str, Any], 
                                 num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate training samples for a specific mapping
        
        Args:
            mapping_id: Mapping identifier
            mapping_config: Mapping configuration
            num_samples: Number of samples to generate
            
        Returns:
            List of training samples
        """
        samples = []
        
        # Extract configuration parameters
        column_mappings = mapping_config.get('column_mappings', {})
        validation_rules = mapping_config.get('validation_rules', {})
        
        # Get tolerance settings
        row_count_tolerance = validation_rules.get('row_count_tolerance', 0.01)
        null_tolerance = validation_rules.get('null_value_tolerance', 0.05)
        
        # Generate samples for each column mapping
        for source_col, mapping_info in column_mappings.items():
            source_type = mapping_info.get('source_type', 'varchar')
            target_type = mapping_info.get('target_type', 'varchar')
            transformation = mapping_info.get('transformation', 'direct')
            nullable = mapping_info.get('nullable', True)
            
            # Generate scenarios for this column
            for _ in range(num_samples // len(column_mappings)):
                sample = self._generate_validation_scenario(
                    mapping_id=mapping_id,
                    source_column=source_col,
                    target_column=mapping_info.get('target_column', source_col),
                    source_type=source_type,
                    target_type=target_type,
                    transformation=transformation,
                    nullable=nullable,
                    row_count_tolerance=row_count_tolerance,
                    null_tolerance=null_tolerance,
                    mapping_info=mapping_info
                )
                samples.append(sample)
        
        return samples
    
    def _generate_validation_scenario(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a single validation scenario with realistic parameters
        
        Returns:
            Dictionary representing a validation scenario
        """
        # Extract parameters
        mapping_id = kwargs['mapping_id']
        source_column = kwargs['source_column']
        target_column = kwargs['target_column']
        source_type = kwargs['source_type']
        target_type = kwargs['target_type']
        transformation = kwargs['transformation']
        nullable = kwargs['nullable']
        row_count_tolerance = kwargs['row_count_tolerance']
        null_tolerance = kwargs['null_tolerance']
        mapping_info = kwargs['mapping_info']
        
        # Generate realistic variations
        row_count_diff = np.random.normal(0, row_count_tolerance / 2)
        row_count_diff = np.clip(row_count_diff, -row_count_tolerance * 3, row_count_tolerance * 3)
        
        null_percentage_diff = np.random.normal(0, null_tolerance / 2)
        null_percentage_diff = np.clip(null_percentage_diff, -null_tolerance * 3, null_tolerance * 3)
        
        # Calculate type compatibility
        type_compatibility = self._calculate_type_compatibility(source_type, target_type)
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(transformation, mapping_info)
        
        # Calculate business rule score
        business_rule_score = self._calculate_business_rule_score(mapping_info)
        
        # Determine validation type
        validation_type = np.random.choice([
            'row_count', 'data_type', 'null_check', 'value_range', 
            'format_check', 'business_rule', 'referential_integrity'
        ], p=[0.2, 0.25, 0.2, 0.1, 0.1, 0.1, 0.05])
        
        # Determine outcome based on comprehensive rules
        outcome = self._determine_scenario_outcome(
            row_count_diff=row_count_diff,
            null_diff=null_percentage_diff,
            type_compatibility=type_compatibility,
            data_quality_score=data_quality_score,
            business_rule_score=business_rule_score,
            validation_type=validation_type,
            row_tolerance=row_count_tolerance,
            null_tolerance=null_tolerance,
            transformation=transformation
        )
        
        return {
            'mapping_id': mapping_id,
            'source_column': source_column,
            'target_column': target_column,
            'source_data_type': source_type,
            'target_data_type': target_type,
            'transformation': transformation,
            'validation_type': validation_type,
            'row_count_diff_pct': row_count_diff,
            'null_percentage_diff': null_percentage_diff,
            'type_compatibility_score': type_compatibility,
            'data_quality_score': data_quality_score,
            'business_rule_score': business_rule_score,
            'nullable': nullable,
            'outcome': outcome
        }
    
    def _calculate_type_compatibility(self, source_type: str, target_type: str) -> float:
        """
        Calculate compatibility score between source and target data types
        
        Args:
            source_type: Source data type
            target_type: Target data type
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Comprehensive type compatibility matrix
        compatibility_matrix = {
            # Perfect matches
            ('integer', 'integer'): 1.0,
            ('varchar', 'varchar'): 1.0,
            ('text', 'text'): 1.0,
            ('timestamp', 'timestamp'): 1.0,
            ('date', 'date'): 1.0,
            ('numeric', 'numeric'): 1.0,
            ('boolean', 'boolean'): 1.0,
            ('char', 'char'): 1.0,
            
            # High compatibility (safe conversions)
            ('integer', 'numeric'): 0.95,
            ('integer', 'bigint'): 0.98,
            ('varchar', 'text'): 0.95,
            ('date', 'timestamp'): 0.9,
            ('char', 'varchar'): 0.9,
            ('numeric', 'decimal'): 0.98,
            
            # Medium compatibility (possible with validation)
            ('varchar', 'char'): 0.8,
            ('text', 'varchar'): 0.85,
            ('timestamp', 'date'): 0.7,
            ('numeric', 'integer'): 0.75,
            ('bigint', 'integer'): 0.7,
            
            # Low compatibility (requires transformation)
            ('integer', 'varchar'): 0.4,
            ('date', 'varchar'): 0.3,
            ('boolean', 'varchar'): 0.3,
            ('timestamp', 'varchar'): 0.25,
            
            # Very low compatibility
            ('text', 'integer'): 0.1,
            ('varchar', 'boolean'): 0.15,
        }
        
        # Check direct match
        key = (source_type.lower(), target_type.lower())
        if key in compatibility_matrix:
            return compatibility_matrix[key]
        
        # Check reverse match
        reverse_key = (target_type.lower(), source_type.lower())
        if reverse_key in compatibility_matrix:
            return compatibility_matrix[reverse_key] * 0.9  # Slight penalty for reverse
        
        # Default compatibility for unknown types
        if source_type.lower() == target_type.lower():
            return 1.0
        
        # Check type families
        numeric_types = {'integer', 'numeric', 'decimal', 'bigint', 'smallint', 'float', 'double'}
        string_types = {'varchar', 'text', 'char', 'string'}
        datetime_types = {'date', 'timestamp', 'datetime', 'time'}
        
        source_lower = source_type.lower()
        target_lower = target_type.lower()
        
        if source_lower in numeric_types and target_lower in numeric_types:
            return 0.8
        elif source_lower in string_types and target_lower in string_types:
            return 0.85
        elif source_lower in datetime_types and target_lower in datetime_types:
            return 0.75
        
        # Default for completely different type families
        return 0.3
    
    def _calculate_data_quality_score(self, transformation: str, mapping_info: Dict[str, Any]) -> float:
        """
        Calculate data quality score based on transformation complexity
        
        Args:
            transformation: Type of transformation
            mapping_info: Additional mapping information
            
        Returns:
            Data quality score (0.0 to 1.0)
        """
        base_scores = {
            'direct': 0.95,
            'lookup': 0.85,
            'custom': 0.7,
            'calculated': 0.75,
            'split': 0.8,
            'concatenate': 0.9,
            'format': 0.8,
            'aggregate': 0.65
        }
        
        base_score = base_scores.get(transformation, 0.6)
        
        # Adjust based on additional factors
        if 'validation_rules' in mapping_info:
            base_score += 0.05  # Bonus for having validation rules
        
        if 'lookup_table' in mapping_info and transformation == 'lookup':
            lookup_size = len(mapping_info['lookup_table'])
            if lookup_size > 10:
                base_score -= 0.1  # Penalty for large lookup tables
        
        return np.clip(base_score + np.random.normal(0, 0.05), 0.0, 1.0)
    
    def _calculate_business_rule_score(self, mapping_info: Dict[str, Any]) -> float:
        """
        Calculate business rule compliance score
        
        Args:
            mapping_info: Mapping information
            
        Returns:
            Business rule score (0.0 to 1.0)
        """
        base_score = 0.8
        
        # Check for nullable constraints
        if 'nullable' in mapping_info:
            if not mapping_info['nullable']:
                base_score += 0.1  # Bonus for non-nullable fields
        
        # Check for primary key
        if mapping_info.get('primary_key', False):
            base_score += 0.1
        
        # Check for foreign key references
        if 'references' in mapping_info:
            base_score += 0.05
        
        # Add some realistic variation
        return np.clip(base_score + np.random.normal(0, 0.1), 0.0, 1.0)
    
    def _determine_scenario_outcome(self, **kwargs) -> str:
        """
        Determine the outcome of a validation scenario based on multiple factors
        
        Returns:
            'PASS' or 'FAIL'
        """
        row_count_diff = kwargs['row_count_diff']
        null_diff = kwargs['null_diff']
        type_compatibility = kwargs['type_compatibility']
        data_quality_score = kwargs['data_quality_score']
        business_rule_score = kwargs['business_rule_score']
        validation_type = kwargs['validation_type']
        row_tolerance = kwargs['row_tolerance']
        null_tolerance = kwargs['null_tolerance']
        transformation = kwargs['transformation']
        
        # Hard failure conditions
        if abs(row_count_diff) > row_tolerance:
            return 'FAIL'
        
        if abs(null_diff) > null_tolerance:
            return 'FAIL'
        
        if type_compatibility < 0.5:
            return 'FAIL'
        
        # Composite score calculation
        composite_score = (
            type_compatibility * 0.3 +
            data_quality_score * 0.25 +
            business_rule_score * 0.2 +
            (1 - abs(row_count_diff) / row_tolerance) * 0.15 +
            (1 - abs(null_diff) / null_tolerance) * 0.1
        )
        
        # Adjust threshold based on validation type
        thresholds = {
            'row_count': 0.7,
            'data_type': 0.75,
            'null_check': 0.65,
            'value_range': 0.8,
            'format_check': 0.85,
            'business_rule': 0.9,
            'referential_integrity': 0.95
        }
        
        threshold = thresholds.get(validation_type, 0.75)
        
        # Add some randomness for edge cases
        if composite_score > threshold:
            return 'PASS' if np.random.random() > 0.05 else 'FAIL'
        else:
            return 'FAIL' if np.random.random() > 0.1 else 'PASS'
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training or prediction
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with prepared features
        """
        features = df.copy()
        
        # Categorical columns to encode
        categorical_columns = ['source_data_type', 'target_data_type', 'validation_type', 'transformation']
        
        for col in categorical_columns:
            if col in features.columns:
                if col not in self.feature_encoders:
                    self.feature_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.feature_encoders[col].fit_transform(features[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        features[f'{col}_encoded'] = self.feature_encoders[col].transform(features[col].astype(str))
                    except ValueError:
                        # Handle new categories by adding them
                        unique_values = features[col].astype(str).unique()
                        known_values = self.feature_encoders[col].classes_
                        new_values = set(unique_values) - set(known_values)
                        
                        if new_values:
                            # Extend the encoder with new values
                            all_values = list(known_values) + list(new_values)
                            self.feature_encoders[col].classes_ = np.array(all_values)
                        
                        features[f'{col}_encoded'] = self.feature_encoders[col].transform(features[col].astype(str))
        
        # Select feature columns for model
        feature_columns = [
            'source_data_type_encoded', 'target_data_type_encoded', 
            'validation_type_encoded', 'transformation_encoded',
            'row_count_diff_pct', 'null_percentage_diff', 
            'type_compatibility_score', 'data_quality_score', 
            'business_rule_score'
        ]
        
        # Filter to only include columns that exist
        available_columns = [col for col in feature_columns if col in features.columns]
        
        return features[available_columns]
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the AI validation model using all available mappings
        
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.model_status = ModelStatus.TRAINING
            logger.info("Starting AI model training...")
            
            # Generate training data
            training_data = self.generate_training_data_from_mappings()
            
            if training_data.empty:
                raise ValueError("No training data generated. Check your mappings.")
            
            logger.info(f"Preparing features from {len(training_data)} samples")
            
            # Prepare features and labels
            X = self.prepare_features(training_data)
            y = self.label_encoder.fit_transform(training_data['outcome'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            logger.info("Training Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate additional metrics
            conf_matrix = confusion_matrix(y_test, y_pred)
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            # Store model metrics
            self.model_metrics = {
                'accuracy': accuracy,
                'training_samples': len(training_data),
                'feature_count': len(X.columns),
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': feature_importance,
                'class_distribution': dict(zip(
                    self.label_encoder.classes_,
                    np.bincount(y_train)
                )),
                'trained_timestamp': datetime.now().isoformat()
            }
            
            # Update training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'training_samples': len(training_data),
                'mapping_count': len(self.mapping_manager.get_all_mappings())
            })
            
            # Check if model meets threshold
            if accuracy >= self.model_accuracy_threshold:
                self.model_status = ModelStatus.TRAINED
                logger.info(f"Model training successful! Accuracy: {accuracy:.3f}")
            else:
                self.model_status = ModelStatus.ERROR
                logger.warning(f"Model accuracy {accuracy:.3f} below threshold {self.model_accuracy_threshold}")
            
            # Print detailed results
            print("\n" + "="*60)
            print("MODEL TRAINING RESULTS")
            print("="*60)
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Training samples: {len(training_data)}")
            print(f"Features: {len(X.columns)}")
            print(f"Mappings used: {len(self.mapping_manager.get_all_mappings())}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
            print("="*60)
            
            # Save model
            self.save_model()
            
            return self.model_metrics
            
        except Exception as e:
            self.model_status = ModelStatus.ERROR
            logger.error(f"Model training failed: {e}")
            raise
    
    def save_model(self, filepath: str = None) -> bool:
        """
        Save the trained model and related components
        
        Args:
            filepath: Custom file path (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath is None:
                models_dir = Path('/app/models')
                models_dir.mkdir(exist_ok=True)
                filepath = models_dir / 'dynamic_validator_model.pkl'
            
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_encoders': self.feature_encoders,
                'scaler': self.scaler,
                'model_status': self.model_status.value,
                'model_metrics': self.model_metrics,
                'training_history': self.training_history,
                'mapping_count': len(self.mapping_manager.get_all_mappings()),
                'saved_timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """
        Load a previously trained model
        
        Args:
            filepath: Custom file path (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath is None:
                filepath = Path('/app/models/dynamic_validator_model.pkl')
            
            if not os.path.exists(filepath):
                logger.info("No existing model file found")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load model components
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_encoders = model_data['feature_encoders']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.model_metrics = model_data.get('model_metrics', {})
            self.training_history = model_data.get('training_history', [])
            
            # Update status
            self.model_status = ModelStatus.TRAINED
            
            mapping_count = model_data.get('mapping_count', 0)
            saved_time = model_data.get('saved_timestamp', 'unknown')
            
            logger.info(f"Model loaded successfully (trained on {mapping_count} mappings, saved: {saved_time})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_status = ModelStatus.ERROR
            return False

# Continue in next part due to length...
# =============================================================================
# FastAPI API Routes - Complete the dynamic_validator.py file
# =============================================================================

# Continue from the previous file...

# API Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Enhanced web interface with real-time status"""
    mappings = mapping_manager.get_all_mappings()
    mapping_options = ""
    mapping_cards = ""

    for mapping_id, mapping_data in mappings.items():
        name = mapping_data.get('name', mapping_id)
        mapping_options += f'<option value="{mapping_id}">{name}</option>'

        # Generate mapping cards with enhanced information
        description = mapping_data.get('description', 'No description')
        source = mapping_data.get('source', {})
        target = mapping_data.get('target', {})
        column_count = len(mapping_data.get('column_mappings', {}))
        version = mapping_data.get('version', '1.0')

        # Get metadata if available
        metadata = mapping_manager.get_mapping_metadata(mapping_id)
        status_info = ""
        if metadata:
            status = metadata.get('status', MappingStatus.ACTIVE)
            modified = metadata.get('modified_time', 'Unknown')
            status_info = f"<p><strong>Status:</strong> {status.value if hasattr(status, 'value') else status}</p>"
            status_info += f"<p><strong>Modified:</strong> {modified}</p>"

        mapping_cards += f"""
        <div class="card">
            <h3>{name} <span class="version">v{version}</span></h3>
            <p><strong>ID:</strong> {mapping_id}</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Source:</strong> {source.get('database', '')}.{source.get('schema', '')}.{source.get('table', '')}</p>
            <p><strong>Target:</strong> {target.get('database', '')}.{target.get('schema', '')}.{target.get('table', '')}</p>
            <p><strong>Columns:</strong> {column_count}</p>
            {status_info}
            <div class="button-group">
                <a href="/validate/{mapping_id}" class="button validate-btn">Validate</a>
                <a href="/mappings/{mapping_id}" class="button view-btn">View Details</a>
                <button onclick="deleteMappingConfirm('{mapping_id}')" class="button delete-btn">Delete</button>
            </div>
        </div>
        """

    # Get model information
    model_info = validator.get_model_info()
    model_status = model_info.get('status', 'unknown')
    model_accuracy = model_info.get('metrics', {}).get('accuracy', 0)
    training_samples = model_info.get('metrics', {}).get('training_samples', 0)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Data Mapping Validator</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px; margin: 0 auto; background: white;
                padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .header h1 {{
                color: #2c3e50; margin: 0; font-size: 2.5em;
                background: linear-gradient(45deg, #3498db, #9b59b6);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }}
            .header p {{ color: #7f8c8d; font-size: 1.2em; margin: 10px 0; }}

            .button {{
                background: linear-gradient(45deg, #3498db, #2980b9); color: white;
                padding: 12px 24px; text-decoration: none; border-radius: 8px;
                margin: 5px; display: inline-block; border: none; cursor: pointer;
                transition: all 0.3s ease; font-weight: 500;
            }}
            .button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
            .validate-btn {{ background: linear-gradient(45deg, #27ae60, #2ecc71); }}
            .view-btn {{ background: linear-gradient(45deg, #f39c12, #e67e22); }}
            .delete-btn {{ background: linear-gradient(45deg, #e74c3c, #c0392b); }}

            .section {{ margin: 40px 0; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; }}
            .card {{
                background: #fff; padding: 25px; border: 2px solid #ecf0f1;
                border-radius: 12px; transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                border-color: #3498db;
            }}

            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
            .status-card {{
                background: linear-gradient(135deg, #74b9ff, #0984e3);
                color: white; padding: 20px; border-radius: 10px; text-align: center;
            }}
            .status-card.model {{ background: linear-gradient(135deg, #fd79a8, #e84393); }}
            .status-card.mappings {{ background: linear-gradient(135deg, #55a3ff, #003d82); }}

            textarea, select, input {{
                width: 100%; padding: 12px; margin: 8px 0; border: 2px solid #ddd;
                border-radius: 8px; font-size: 14px; transition: border-color 0.3s ease;
            }}
            textarea:focus, select:focus, input:focus {{
                outline: none; border-color: #3498db;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }}

            .version {{
                background: #3498db; color: white; padding: 2px 8px;
                border-radius: 12px; font-size: 0.8em; margin-left: 10px;
            }}
            .button-group {{ margin-top: 15px; }}
            .mapping-list {{ max-height: 600px; overflow-y: auto; }}

            .success {{ color: #27ae60; font-weight: bold; }}
            .error {{ color: #e74c3c; font-weight: bold; }}
            .warning {{ color: #f39c12; font-weight: bold; }}

            .loading {{
                display: none; background: rgba(0,0,0,0.7); position: fixed;
                top: 0; left: 0; width: 100%; height: 100%; z-index: 1000;
                justify-content: center; align-items: center; color: white; font-size: 1.5em;
            }}

            @media (max-width: 768px) {{
                .grid {{ grid-template-columns: 1fr; }}
                .container {{ padding: 15px; }}
            }}
        </style>
    </head>
    <body>
        <div class="loading" id="loading">
            <div>🔄 Processing...</div>
        </div>

        <div class="container">
            <div class="header">
                <h1>🔄 Dynamic Data Mapping Validator</h1>
                <p>AI-powered validation system with dynamic mapping management</p>
            </div>

            <div class="section">
                <div class="status-grid">
                    <div class="status-card">
                        <h3>📊 System Status</h3>
                        <div id="status-info">Loading...</div>
                    </div>
                    <div class="status-card model">
                        <h3>🧠 AI Model</h3>
                        <div>Status: <span class="{('success' if model_status == 'trained' else 'error')}">{model_status.title()}</span></div>
                        <div>Accuracy: {model_accuracy:.1%}</div>
                        <div>Samples: {training_samples:,}</div>
                    </div>
                    <div class="status-card mappings">
                        <h3>📋 Mappings</h3>
                        <div>Total: {len(mappings)}</div>
                        <div>Last Updated: <span id="last-updated">Just Now</span></div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🚀 Quick Actions</h2>
                <div style="text-align: center;">
                    <a href="/validate-all" class="button">Validate All Mappings</a>
                    <a href="/mappings" class="button">View All Mappings</a>
                    <a href="/retrain" class="button">Retrain Model</a>
                    <a href="/docs" class="button">API Documentation</a>
                </div>
            </div>

            <div class="section">
                <h2>📋 Available Mappings ({len(mappings)})</h2>
                <div class="mapping-list">
                    {mapping_cards if mapping_cards else '<p>No mappings available. Add your first mapping below!</p>'}
                </div>
            </div>

            <div class="section">
                <h2>➕ Add New Mapping</h2>
                <div class="grid">
                    <div class="card">
                        <h3>📁 Upload Mapping File</h3>
                        <input type="file" id="mapping-file" accept=".json">
                        <button onclick="uploadMapping()" class="button">Upload Mapping</button>
                        <div id="upload-result"></div>
                    </div>

                    <div class="card">
                        <h3>✏️ Create New Mapping</h3>
                        <textarea id="new-mapping" rows="15" placeholder="Paste JSON mapping configuration here...">{{
  "mapping_id": "new_table_mapping",
  "name": "New Table Migration",
  "description": "Description of the mapping",
  "version": "1.0",
  "source": {{
    "database": "source_system",
    "schema": "legacy",
    "table": "table_name"
  }},
  "target": {{
    "database": "target_system",
    "schema": "modern",
    "table": "table_name"
  }},
  "column_mappings": {{
    "source_col": {{
      "target_column": "target_col",
      "transformation": "direct",
      "source_type": "varchar",
      "target_type": "varchar",
      "nullable": true
    }}
  }},
  "validation_rules": {{
    "row_count_tolerance": 0.01,
    "null_value_tolerance": 0.05
  }}
}}</textarea>
                        <button onclick="createMapping()" class="button">Create Mapping</button>
                        <div id="create-result"></div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>🔍 Validate Specific Mappings</h2>
                <div class="card">
                    <select id="mapping-selector" multiple size="6">
                        {mapping_options}
                    </select>
                    <div style="margin: 15px 0;">
                        <button onclick="validateSelected()" class="button">Validate Selected</button>
                        <button onclick="selectAllMappings()" class="button view-btn">Select All</button>
                        <button onclick="clearSelection()" class="button delete-btn">Clear Selection</button>
                    </div>
                    <div id="validation-results"></div>
                </div>
            </div>
        </div>

        <script>
            // Enhanced JavaScript functionality
            let lastStatusUpdate = new Date();

            // Load status on page load and refresh periodically
            function updateStatus() {{
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {{
                        const statusElement = document.getElementById('status-info');
                        const trainedClass = data.model_trained ? 'success' : 'error';
                        statusElement.innerHTML = `
                            <div>Model: <span class="${{trainedClass}}">${{data.model_trained ? 'Ready' : 'Not Trained'}}</span></div>
                            <div>Mappings: ${{data.mappings_loaded}}</div>
                        `;
                        document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                        lastStatusUpdate = new Date();
                    }})
                    .catch(error => {{
                        console.error('Status update failed:', error);
                        document.getElementById('status-info').innerHTML = '<div class="error">Connection Error</div>';
                    }});
            }}

            updateStatus();
            setInterval(updateStatus, 30000); // Update every 30 seconds

            function showLoading() {{
                document.getElementById('loading').style.display = 'flex';
            }}

            function hideLoading() {{
                document.getElementById('loading').style.display = 'none';
            }}

            function uploadMapping() {{
                const fileInput = document.getElementById('mapping-file');
                const file = fileInput.files[0];
                const resultDiv = document.getElementById('upload-result');

                if (!file) {{
                    resultDiv.innerHTML = '<div class="error">Please select a file</div>';
                    return;
                }}

                showLoading();
                const formData = new FormData();
                formData.append('file', file);

                fetch('/upload-mapping', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    hideLoading();
                    if (data.status === 'success') {{
                        resultDiv.innerHTML = '<div class="success">✅ Mapping uploaded successfully!</div>';
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        resultDiv.innerHTML = `<div class="error">❌ Error: ${{data.message}}</div>`;
                    }}
                }})
                .catch(error => {{
                    hideLoading();
                    resultDiv.innerHTML = `<div class="error">❌ Upload failed: ${{error.message}}</div>`;
                }});
            }}

            function createMapping() {{
                const mappingText = document.getElementById('new-mapping').value;
                const resultDiv = document.getElementById('create-result');

                try {{
                    const mappingData = JSON.parse(mappingText);
                    showLoading();

                    fetch('/mappings', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ mapping_data: mappingData }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        hideLoading();
                        if (data.status === 'success') {{
                            resultDiv.innerHTML = '<div class="success">✅ Mapping created successfully!</div>';
                            setTimeout(() => location.reload(), 2000);
                        }} else {{
                            resultDiv.innerHTML = `<div class="error">❌ Error: ${{data.message}}</div>`;
                        }}
                    }});
                }} catch (e) {{
                    resultDiv.innerHTML = `<div class="error">❌ Invalid JSON: ${{e.message}}</div>`;
                }}
            }}

            function validateSelected() {{
                const selector = document.getElementById('mapping-selector');
                const selectedMappings = Array.from(selector.selectedOptions).map(option => option.value);
                const resultDiv = document.getElementById('validation-results');

                if (selectedMappings.length === 0) {{
                    resultDiv.innerHTML = '<div class="warning">⚠️ Please select at least one mapping</div>';
                    return;
                }}

                showLoading();

                fetch('/validate', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ mapping_ids: selectedMappings }})
                }})
                .then(response => response.json())
                .then(data => {{
                    hideLoading();
                    if (data.status === 'success') {{
                        const results = data.validation_results;
                        let html = '<h3>🔍 Validation Results:</h3>';

                        for (const [mappingId, result] of Object.entries(results)) {{
                            const status = result.overall_result || 'ERROR';
                            const statusClass = status === 'PASS' ? 'success' : 'error';
                            const confidence = result.overall_confidence ? (result.overall_confidence * 100).toFixed(1) : 'N/A';

                            html += `
                                <div class="card">
                                    <h4>${{mappingId}}</h4>
                                    <p><strong>Result:</strong> <span class="${{statusClass}}">${{status}}</span></p>
                                    <p><strong>Confidence:</strong> ${{confidence}}%</p>
                                    <p><strong>Columns:</strong> ${{result.summary?.passed || 0}} passed, ${{result.summary?.failed || 0}} failed</p>
                                </div>
                            `;
                        }}

                        resultDiv.innerHTML = html;
                    }} else {{
                        resultDiv.innerHTML = `<div class="error">❌ Validation failed: ${{data.message}}</div>`;
                    }}
                }})
                .catch(error => {{
                    hideLoading();
                    resultDiv.innerHTML = `<div class="error">❌ Validation error: ${{error.message}}</div>`;
                }});
            }}

            function selectAllMappings() {{
                const selector = document.getElementById('mapping-selector');
                for (let option of selector.options) {{
                    option.selected = true;
                }}
            }}

            function clearSelection() {{
                const selector = document.getElementById('mapping-selector');
                for (let option of selector.options) {{
                    option.selected = false;
                }}
            }}

            function deleteMappingConfirm(mappingId) {{
                if (confirm(`Are you sure you want to delete mapping "${{mappingId}}"?`)) {{
                    showLoading();
                    fetch(`/mappings/${{mappingId}}`, {{
                        method: 'DELETE'
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        hideLoading();
                        if (data.status === 'success') {{
                            alert('Mapping deleted successfully!');
                            location.reload();
                        }} else {{
                            alert(`Error: ${{data.message}}`);
                        }}
                    }});
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    mappings = mapping_manager.get_all_mappings()
    model_info = validator.get_model_info()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": validator.is_trained,
        "model_status": model_info.get('status'),
        "model_accuracy": model_info.get('metrics', {}).get('accuracy'),
        "mappings_loaded": len(mappings),
        "mapping_list": list(mappings.keys()),
        "system_info": {
            "max_training_samples": validator.max_training_samples,
            "accuracy_threshold": validator.model_accuracy_threshold,
            "auto_retrain": validator.auto_retrain
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/mappings")
async def get_all_mappings():
    """Get all mappings with metadata"""
    mappings = mapping_manager.get_all_mappings()
    metadata = mapping_manager.get_all_metadata()

    enhanced_mappings = {}
    for mapping_id, mapping_data in mappings.items():
        enhanced_mappings[mapping_id] = {
            **mapping_data,
            "metadata": metadata.get(mapping_id, {})
        }

    return {
        "status": "success",
        "mappings": enhanced_mappings,
        "count": len(mappings),
        "statistics": mapping_manager.get_statistics()
    }

@app.get("/mappings/{mapping_id}")
async def get_mapping(mapping_id: str):
    """Get a specific mapping with detailed information"""
    mapping = mapping_manager.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail=f"Mapping {mapping_id} not found")

    metadata = mapping_manager.get_mapping_metadata(mapping_id)

    return {
        "status": "success",
        "mapping": mapping,
        "metadata": metadata
    }

@app.post("/mappings")
async def create_mapping(request: MappingRequest):
    """Create a new mapping"""
    try:
        success = mapping_manager.add_mapping(request.mapping_data)
        if success:
            # Trigger model retraining if auto-retrain is enabled
            if validator.auto_retrain:
                try:
                    validator.retrain_if_needed()
                except Exception as e:
                    logger.warning(f"Auto-retrain failed: {e}")

            return {"status": "success", "message": "Mapping created successfully"}
        else:
            return {"status": "error", "message": "Failed to create mapping"}
    except Exception as e:
        logger.error(f"Error creating mapping: {e}")
        return {"status": "error", "message": str(e)}

@app.put("/mappings/{mapping_id}")
async def update_mapping(mapping_id: str, request: MappingRequest):
    """Update an existing mapping"""
    try:
        success = mapping_manager.update_mapping(mapping_id, request.mapping_data)
        if success:
            return {"status": "success", "message": "Mapping updated successfully"}
        else:
            return {"status": "error", "message": "Failed to update mapping"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/mappings/{mapping_id}")
async def delete_mapping(mapping_id: str):
    """Delete a mapping"""
    try:
        success = mapping_manager.delete_mapping(mapping_id)
        if success:
            return {"status": "success", "message": "Mapping deleted successfully"}
        else:
            return {"status": "error", "message": "Mapping not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload-mapping")
async def upload_mapping_file(file: UploadFile = File(...)):
    """Upload a mapping file"""
    try:
        content = await file.read()
        mapping_data = json.loads(content.decode('utf-8'))

        success = mapping_manager.add_mapping(mapping_data)
        if success:
            return {"status": "success", "message": f"Mapping uploaded: {mapping_data.get('mapping_id')}"}
        else:
            return {"status": "error", "message": "Failed to save mapping"}
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON file"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the AI model with current mappings"""
    try:
        mappings = mapping_manager.get_all_mappings()
        if not mappings:
            return {"status": "error", "message": "No mappings available for training"}

        # Run training in background for better responsiveness
        def train_model_background():
            try:
                validator.train_model()
                logger.info("Background model training completed successfully")
            except Exception as e:
                logger.error(f"Background model training failed: {e}")

        background_tasks.add_task(train_model_background)

        return {
            "status": "success",
            "message": f"Model retraining started with {len(mappings)} mappings",
            "background": True
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/model/info")
async def get_model_info():
    """Get detailed AI model information"""
    return validator.get_model_info()

@app.get("/validate/{mapping_id}")
async def validate_single_mapping(mapping_id: str):
    """Validate a single mapping"""
    try:
        if not validator.is_trained:
            return {"status": "error", "message": "Model not trained. Please retrain first."}

        source_db, target_db = get_database_connectors()
        validation_results = validator.validate_mapping(mapping_id, source_db, target_db)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/app/reports/validation_{mapping_id}_{timestamp}.json"
        Path(report_path).parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

        return {
            "status": "success",
            "validation_results": validation_results,
            "report_saved": report_path
        }

    except Exception as e:
        logger.error(f"Validation failed for {mapping_id}: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/validate")
async def validate_multiple_mappings(request: ValidationRequest):
    """Validate multiple mappings"""
    try:
        if not validator.is_trained:
            return {"status": "error", "message": "Model not trained. Please retrain first."}

        source_db, target_db = get_database_connectors()
        results = {}

        for mapping_id in request.mapping_ids:
            try:
                validation_result = validator.validate_mapping(mapping_id, source_db, target_db)
                results[mapping_id] = validation_result
            except Exception as e:
                logger.error(f"Validation failed for {mapping_id}: {e}")
                results[mapping_id] = {"status": "error", "message": str(e)}

        # Save consolidated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/app/reports/validation_batch_{timestamp}.json"
        Path(report_path).parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate summary statistics
        summary = {
            "total_mappings": len(request.mapping_ids),
            "successful_validations": len([r for r in results.values() if r.get("overall_result") == "PASS"]),
            "failed_validations": len([r for r in results.values() if r.get("overall_result") == "FAIL"]),
            "errors": len([r for r in results.values() if "error" in r.get("status", "")])
        }

        return {
            "status": "success",
            "validation_results": results,
            "report_saved": report_path,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Batch validation failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/validate-all")
async def validate_all_mappings():
    """Validate all available mappings"""
    try:
        all_mappings = list(mapping_manager.get_all_mappings().keys())
        if not all_mappings:
            return {"status": "error", "message": "No mappings available"}

        request = ValidationRequest(mapping_ids=all_mappings)
        return await validate_multiple_mappings(request)

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reload-mappings")
async def reload_mappings():
    """Reload all mappings from disk"""
    try:
        summary = mapping_manager.reload_mappings()
        return {
            "status": "success",
            "message": f"Reloaded {summary['new_count']} mappings",
            "summary": summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive system statistics"""
    mapping_stats = mapping_manager.get_statistics()
    model_info = validator.get_model_info()

    return {
        "timestamp": datetime.now().isoformat(),
        "mappings": mapping_stats,
        "model": model_info,
        "system": {
            "total_validations_today": len(glob.glob("/app/reports/validation_*_*.json")),
            "up

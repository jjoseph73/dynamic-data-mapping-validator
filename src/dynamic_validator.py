# =============================================================================
# src/dynamic_validator.py - Enhanced AI-Powered Dynamic Validation System
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from contextlib import contextmanager

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal

# Internal imports
from mapping_manager import MappingManager, MappingStatus
from database_connector import DatabaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Enhanced validation result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

class ModelStatus(Enum):
    """Enhanced AI model status"""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    RETRAINING = "retraining"
    ERROR = "error"
    OUTDATED = "outdated"

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ValidationMetrics:
    """Enhanced metrics for validation results"""
    row_count_difference_pct: float
    null_percentage_difference: float
    type_compatibility_score: float
    data_quality_score: float = 0.0
    business_rule_score: float = 0.0
    data_integrity_score: float = 0.0
    performance_impact_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        weights = {
            'type_compatibility_score': 0.25,
            'data_quality_score': 0.20,
            'business_rule_score': 0.20,
            'data_integrity_score': 0.15,
            'performance_impact_score': 0.10,
            'null_handling': 0.10  # Based on null_percentage_difference
        }
        
        null_handling_score = max(0, 1 - abs(self.null_percentage_difference) / 100)
        
        return (
            self.type_compatibility_score * weights['type_compatibility_score'] +
            self.data_quality_score * weights['data_quality_score'] +
            self.business_rule_score * weights['business_rule_score'] +
            self.data_integrity_score * weights['data_integrity_score'] +
            self.performance_impact_score * weights['performance_impact_score'] +
            null_handling_score * weights['null_handling']
        )

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    column: Optional[str] = None
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ValidationResult:
    """Enhanced validation result with issues tracking"""
    validation_result: ValidationStatus
    confidence: float
    metrics: ValidationMetrics
    mapping_info: Dict[str, Any]
    issues: List[ValidationIssue] = None
    details: Dict[str, Any] = None
    timestamp: str = None
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.issues is None:
            self.issues = []
    
    def add_issue(self, severity: ValidationSeverity, category: str, 
                  message: str, column: str = None, recommendation: str = None):
        """Add a validation issue"""
        self.issues.append(ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            column=column,
            recommendation=recommendation
        ))
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'validation_result': self.validation_result.value,
            'confidence': self.confidence,
            'metrics': self.metrics.to_dict(),
            'mapping_info': self.mapping_info,
            'issues': [issue.to_dict() for issue in self.issues],
            'details': self.details,
            'timestamp': self.timestamp,
            'execution_time_ms': self.execution_time_ms,
            'overall_score': self.metrics.overall_score
        }

class ModelConfiguration:
    """Configuration for ML model training"""
    def __init__(self):
        self.model_type = os.getenv('MODEL_TYPE', 'random_forest')  # or 'gradient_boosting'
        self.max_training_samples = int(os.getenv('MAX_TRAINING_SAMPLES', 5000))
        self.accuracy_threshold = float(os.getenv('MODEL_ACCURACY_THRESHOLD', 0.85))
        self.auto_retrain = os.getenv('MODEL_AUTO_RETRAIN', 'true').lower() == 'true'
        self.retrain_threshold = float(os.getenv('RETRAIN_THRESHOLD', 0.8))
        self.use_class_weights = os.getenv('USE_CLASS_WEIGHTS', 'true').lower() == 'true'
        self.feature_selection = os.getenv('FEATURE_SELECTION', 'auto')  # auto, manual, all
        self.validation_split = float(os.getenv('VALIDATION_SPLIT', 0.2))
        self.random_state = int(os.getenv('RANDOM_STATE', 42))

class DynamicDataMappingValidator:
    """
    Enhanced AI-powered validation system with improved performance,
    error handling, and advanced analytics.
    """
    
    def __init__(self, mapping_manager: MappingManager, config: ModelConfiguration = None):
        """Initialize the enhanced dynamic validator"""
        self.mapping_manager = mapping_manager
        self.config = config or ModelConfiguration()
        
        # Initialize model based on configuration
        self._init_model()
        
        # Enhanced components
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # State management
        self.model_status = ModelStatus.NOT_TRAINED
        self.model_metrics = {}
        self.training_history = []
        self._training_lock = threading.Lock()
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_execution_time': 0.0,
            'last_reset': datetime.now().isoformat()
        }
        
        # Setup change watcher
        if self.config.auto_retrain:
            self.mapping_manager.add_change_watcher(self._on_mapping_change)
        
        logger.info(f"DynamicDataMappingValidator initialized with {self.config.model_type} model")
    
    def _init_model(self):
        """Initialize ML model based on configuration"""
        if self.config.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state
            )
        else:  # default to random_forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight='balanced' if self.config.use_class_weights else None
            )
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained and ready"""
        return self.model_status == ModelStatus.TRAINED
    
    @property
    def needs_retraining(self) -> bool:
        """Check if model needs retraining"""
        if not self.is_trained:
            return True
        
        if not self.training_history:
            return True
        
        # Check if accuracy dropped below threshold
        last_training = self.training_history[-1]
        if last_training.get('accuracy', 0) < self.config.retrain_threshold:
            return True
        
        # Check if mappings changed significantly
        current_mappings = len(self.mapping_manager.get_all_mappings())
        last_mapping_count = last_training.get('mapping_count', 0)
        
        return current_mappings > last_mapping_count * 1.3
    
    @contextmanager
    def _performance_timer(self):
        """Context manager for tracking execution time"""
        start_time = datetime.now()
        try:
            yield start_time
        finally:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(execution_time)
    
    def _update_performance_stats(self, execution_time_ms: float):
        """Update performance statistics"""
        self.validation_stats['total_validations'] += 1
        current_avg = self.validation_stats['average_execution_time']
        total = self.validation_stats['total_validations']
        
        # Calculate running average
        self.validation_stats['average_execution_time'] = (
            (current_avg * (total - 1) + execution_time_ms) / total
        )
    
    def _on_mapping_change(self, event: str, mapping_id: str):
        """Enhanced mapping change handler with debouncing"""
        if event in ['mapping_added', 'mapping_updated', 'bulk_reload']:
            logger.info(f"Mapping change detected: {event} for {mapping_id}")
            
            # Simple debouncing - don't retrain immediately
            if self.needs_retraining:
                logger.info("Scheduling model retraining due to significant changes")
                # In production, you might want to use a task queue here
                threading.Timer(30.0, self._auto_retrain).start()
    
    def _auto_retrain(self):
        """Perform automatic retraining in background"""
        try:
            with self._training_lock:
                if self.needs_retraining:
                    logger.info("Starting automatic model retraining")
                    self.train_model()
        except Exception as e:
            logger.error(f"Automatic retraining failed: {e}")
    
    def _calculate_type_compatibility(self, source_type: str, target_type: str) -> float:
        """Enhanced type compatibility calculation"""
        # Expanded compatibility matrix with more precise scores
        compatibility_matrix = {
            # Perfect matches
            ('integer', 'integer'): 1.0,
            ('varchar', 'varchar'): 1.0,
            ('text', 'text'): 1.0,
            ('timestamp', 'timestamp'): 1.0,
            ('date', 'date'): 1.0,
            ('numeric', 'numeric'): 1.0,
            ('decimal', 'decimal'): 1.0,
            ('boolean', 'boolean'): 1.0,
            ('char', 'char'): 1.0,
            ('json', 'json'): 1.0,
            ('uuid', 'uuid'): 1.0,
            
            # High compatibility (safe conversions)
            ('integer', 'numeric'): 0.95,
            ('integer', 'decimal'): 0.93,
            ('integer', 'bigint'): 0.98,
            ('smallint', 'integer'): 0.97,
            ('varchar', 'text'): 0.95,
            ('char', 'varchar'): 0.92,
            ('date', 'timestamp'): 0.90,
            ('numeric', 'decimal'): 0.98,
            ('float', 'double'): 0.95,
            
            # Medium compatibility (possible with validation)
            ('varchar', 'char'): 0.80,
            ('text', 'varchar'): 0.85,
            ('timestamp', 'date'): 0.70,  # Data loss possible
            ('numeric', 'integer'): 0.75,  # Precision loss
            ('bigint', 'integer'): 0.70,   # Overflow risk
            ('double', 'float'): 0.72,     # Precision loss
            
            # Low compatibility (requires transformation)
            ('integer', 'varchar'): 0.45,
            ('numeric', 'varchar'): 0.40,
            ('date', 'varchar'): 0.35,
            ('boolean', 'varchar'): 0.30,
            ('timestamp', 'varchar'): 0.25,
            ('json', 'text'): 0.60,
            ('uuid', 'varchar'): 0.50,
            
            # Very low compatibility
            ('text', 'integer'): 0.10,
            ('varchar', 'boolean'): 0.15,
            ('text', 'date'): 0.05,
            ('json', 'integer'): 0.05,
        }
        
        # Normalize type names
        source_lower = source_type.lower().strip()
        target_lower = target_type.lower().strip()
        
        # Direct lookup
        key = (source_lower, target_lower)
        if key in compatibility_matrix:
            return compatibility_matrix[key]
        
        # Reverse lookup with slight penalty
        reverse_key = (target_lower, source_lower)
        if reverse_key in compatibility_matrix:
            return compatibility_matrix[reverse_key] * 0.9
        
        # Exact match for unknown types
        if source_lower == target_lower:
            return 1.0
        
        # Type family compatibility
        numeric_types = {'integer', 'numeric', 'decimal', 'bigint', 'smallint', 'float', 'double', 'real'}
        string_types = {'varchar', 'text', 'char', 'string', 'clob'}
        datetime_types = {'date', 'timestamp', 'datetime', 'time'}
        binary_types = {'blob', 'bytea', 'binary', 'varbinary'}
        
        if source_lower in numeric_types and target_lower in numeric_types:
            return 0.80
        elif source_lower in string_types and target_lower in string_types:
            return 0.85
        elif source_lower in datetime_types and target_lower in datetime_types:
            return 0.75
        elif source_lower in binary_types and target_lower in binary_types:
            return 0.70
        
        # Cross-family conversions
        if (source_lower in numeric_types and target_lower in string_types) or \
           (source_lower in string_types and target_lower in numeric_types):
            return 0.40
        
        if (source_lower in datetime_types and target_lower in string_types) or \
           (source_lower in string_types and target_lower in datetime_types):
            return 0.30
        
        # Default for completely incompatible types
        return 0.20
    
    def _calculate_data_quality_score(self, transformation: str, mapping_info: Dict[str, Any]) -> float:
        """Enhanced data quality scoring"""
        base_scores = {
            'direct': 0.95,
            'lookup': 0.85,
            'custom': 0.70,
            'calculated': 0.75,
            'split': 0.80,
            'concatenate': 0.90,
            'format': 0.82,
            'aggregate': 0.65,
            'conditional': 0.78,
            'cast': 0.88,
            'substring': 0.83,
            'replace': 0.79
        }
        
        base_score = base_scores.get(transformation.lower(), 0.60)
        
        # Quality adjustments
        adjustments = 0.0
        
        # Validation rules bonus
        if 'validation_rules' in mapping_info:
            rules = mapping_info['validation_rules']
            if isinstance(rules, dict) and rules:
                adjustments += 0.05
        
        # Lookup table assessment
        if transformation == 'lookup' and 'lookup_table' in mapping_info:
            lookup_size = len(mapping_info['lookup_table'])
            if lookup_size > 100:
                adjustments -= 0.15  # Large lookup penalty
            elif lookup_size > 50:
                adjustments -= 0.08
            elif lookup_size < 10:
                adjustments += 0.03  # Small lookup bonus
        
        # Custom function complexity
        if transformation == 'custom':
            if 'function_complexity' in mapping_info:
                complexity = mapping_info['function_complexity'].lower()
                complexity_penalties = {
                    'low': 0.05,
                    'medium': 0.0,
                    'high': -0.10,
                    'very_high': -0.20
                }
                adjustments += complexity_penalties.get(complexity, -0.05)
        
        # Data type conversion assessment
        if 'source_type' in mapping_info and 'target_type' in mapping_info:
            type_compat = self._calculate_type_compatibility(
                mapping_info['source_type'], 
                mapping_info['target_type']
            )
            if type_compat < 0.7:
                adjustments -= 0.08  # Risky conversion penalty
        
        final_score = base_score + adjustments
        return np.clip(final_score + np.random.normal(0, 0.03), 0.0, 1.0)
    
    def _calculate_business_rule_score(self, mapping_info: Dict[str, Any]) -> float:
        """Enhanced business rule compliance scoring"""
        base_score = 0.75
        
        # Constraint analysis
        constraints_bonus = 0.0
        
        # Nullable constraints
        if 'nullable' in mapping_info:
            if not mapping_info['nullable']:
                constraints_bonus += 0.08  # NOT NULL bonus
        
        # Primary key
        if mapping_info.get('primary_key', False):
            constraints_bonus += 0.10
        
        # Foreign key references
        if 'references' in mapping_info:
            constraints_bonus += 0.06
        
        # Unique constraints
        if mapping_info.get('unique', False):
            constraints_bonus += 0.05
        
        # Check constraints
        if 'check_constraints' in mapping_info:
            check_count = len(mapping_info['check_constraints'])
            constraints_bonus += min(0.08, check_count * 0.02)
        
        # Default values
        if 'default_value' in mapping_info:
            constraints_bonus += 0.03
        
        # Business rule validation
        if 'business_rules' in mapping_info:
            rules = mapping_info['business_rules']
            if isinstance(rules, (list, dict)) and rules:
                constraints_bonus += 0.07
        
        final_score = base_score + constraints_bonus
        return np.clip(final_score + np.random.normal(0, 0.05), 0.0, 1.0)
    
    def _calculate_data_integrity_score(self, mapping_info: Dict[str, Any]) -> float:
        """Calculate data integrity score"""
        base_score = 0.80
        
        # Referential integrity
        if 'foreign_keys' in mapping_info:
            base_score += 0.10
        
        # Data consistency rules
        if 'consistency_rules' in mapping_info:
            base_score += 0.08
        
        # Duplicate handling
        if 'duplicate_strategy' in mapping_info:
            strategy = mapping_info['duplicate_strategy'].lower()
            if strategy in ['reject', 'merge']:
                base_score += 0.05
        
        return np.clip(base_score + np.random.normal(0, 0.05), 0.0, 1.0)
    
    def _calculate_performance_impact_score(self, mapping_info: Dict[str, Any]) -> float:
        """Calculate performance impact score"""
        base_score = 0.85
        
        # Transformation complexity impact
        transformation = mapping_info.get('transformation', 'direct').lower()
        complexity_impact = {
            'direct': 0.0,
            'cast': -0.02,
            'format': -0.05,
            'lookup': -0.08,
            'calculated': -0.10,
            'custom': -0.15,
            'aggregate': -0.20
        }
        
        base_score += complexity_impact.get(transformation, -0.10)
        
        # Index availability
        if mapping_info.get('indexed', False):
            base_score += 0.08
        
        # Data volume impact
        if 'estimated_rows' in mapping_info:
            rows = mapping_info['estimated_rows']
            if rows > 10000000:  # 10M+ rows
                base_score -= 0.10
            elif rows > 1000000:  # 1M+ rows
                base_score -= 0.05
        
        return np.clip(base_score + np.random.normal(0, 0.03), 0.0, 1.0)

    async def validate_column_mapping_async(self, source_col_stats: Dict[str, Any], 
                                          target_col_stats: Dict[str, Any], 
                                          mapping_info: Dict[str, Any],
                                          validation_type: str = 'data_type') -> ValidationResult:
        """Async version of column validation"""
        # This would be the async wrapper around the sync method
        return await asyncio.get_event_loop().run_in_executor(
            None, 
            self.validate_column_mapping,
            source_col_stats,
            target_col_stats, 
            mapping_info,
            validation_type
        )
    
    def validate_column_mapping(self, source_col_stats: Dict[str, Any], 
                               target_col_stats: Dict[str, Any], 
                               mapping_info: Dict[str, Any],
                               validation_type: str = 'data_type') -> ValidationResult:
        """Enhanced column mapping validation with detailed issue tracking"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        
        with self._performance_timer() as start_time:
            try:
                # Calculate comprehensive metrics
                row_count_diff = ((target_col_stats['row_count'] - source_col_stats['row_count']) 
                                / source_col_stats['row_count']) if source_col_stats['row_count'] > 0 else 0
                
                null_diff = target_col_stats['null_percentage'] - source_col_stats['null_percentage']
                
                # Enhanced type information
                source_type = mapping_info.get('source_type', source_col_stats['data_type'])
                target_type = mapping_info.get('target_type', target_col_stats['data_type'])
                transformation = mapping_info.get('transformation', 'direct')
                
                # Calculate all scores
                type_compat = self._calculate_type_compatibility(source_type, target_type)
                data_quality_score = self._calculate_data_quality_score(transformation, mapping_info)
                business_rule_score = self._calculate_business_rule_score(mapping_info)
                data_integrity_score = self._calculate_data_integrity_score(mapping_info)
                performance_impact_score = self._calculate_performance_impact_score(mapping_info)
                
                # Create enhanced metrics
                metrics = ValidationMetrics(
                    row_count_difference_pct=row_count_diff * 100,
                    null_percentage_difference=null_diff * 100,
                    type_compatibility_score=type_compat,
                    data_quality_score=data_quality_score,
                    business_rule_score=business_rule_score,
                    data_integrity_score=data_integrity_score,
                    performance_impact_score=performance_impact_score
                )
                
                # Prepare features for ML prediction
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
                
                # ML Prediction
                X = self.prepare_features(feature_data)
                X_scaled = self.scaler.transform(X)
                
                prediction = self.model.predict(X_scaled)[0]
                probabilities = self.model.predict_proba(X_scaled)[0]
                
                outcome = self.label_encoder.inverse_transform([prediction])[0]
                confidence = max(probabilities)
                
                # Create validation result
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                result = ValidationResult(
                    validation_result=ValidationStatus(outcome),
                    confidence=confidence,
                    metrics=metrics,
                    mapping_info=mapping_info,
                    execution_time_ms=execution_time,
                    details={
                        'prediction_probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                        'feature_values': X.iloc[0].to_dict(),
                        'source_stats': source_col_stats,
                        'target_stats': target_col_stats,
                        'validation_type': validation_type,
                        'overall_score': metrics.overall_score
                    }
                )
                
                # Add detailed issues based on analysis
                self._analyze_and_add_issues(result, source_col_stats, target_col_stats, mapping_info)
                
                # Update stats
                self.validation_stats['successful_validations'] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Column validation failed: {e}")
                self.validation_stats['failed_validations'] += 1
                
                error_result = ValidationResult(
                    validation_result=ValidationStatus.ERROR,
                    confidence=0.0,
                    metrics=ValidationMetrics(0, 0, 0),
                    mapping_info=mapping_info,
                    details={'error': str(e)}
                )
                
                error_result.add_issue(
                    ValidationSeverity.CRITICAL,
                    'system_error',
                    f"Validation failed: {str(e)}",
                    recommendation="Check system logs and retry validation"
                )
                
                return error_result
    
    def _analyze_and_add_issues(self, result: ValidationResult, 
                               source_stats: Dict[str, Any],
                               target_stats: Dict[str, Any],
                               mapping_info: Dict[str, Any]):
        """Analyze validation result and add specific issues"""
        
        metrics = result.metrics
        
        # Critical issues
        if metrics.type_compatibility_score < 0.5:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                'type_incompatibility',
                f"Low type compatibility ({metrics.type_compatibility_score:.2f}) between "
                f"{mapping_info.get('source_type', 'unknown')} and {mapping_info.get('target_type', 'unknown')}",
                recommendation="Consider using explicit type casting or alternative target type"
            )
        
        # High severity issues
        if abs(metrics.row_count_difference_pct) > 5:
            result.add_issue(
                ValidationSeverity.HIGH,
                'data_loss',
                f"Significant row count difference: {metrics.row_count_difference_pct:.1f}%",
                recommendation="Investigate data filtering or transformation logic"
            )
        
        if abs(metrics.null_percentage_difference) > 10:
            result.add_issue(
                ValidationSeverity.HIGH,
                'null_handling',
                f"Large null percentage change: {metrics.null_percentage_difference:.1f}%",
                recommendation="Review null value handling in transformation"
            )
        
        # Medium severity issues
        if metrics.data_quality_score < 0.7:
            result.add_issue(
                ValidationSeverity.MEDIUM,
                'data_quality',
                f"Low data quality score ({metrics.data_quality_score:.2f})",
                recommendation="Simplify transformation or add validation rules"
            )
        
        if metrics.performance_impact_score < 0.6:
            result.add_issue(
                ValidationSeverity.MEDIUM,
                'performance',
                f"High performance impact expected ({metrics.performance_impact_score:.2f})",
                recommendation="Consider adding indexes or optimizing transformation"
            )
        
        # Low severity/info issues
        if metrics.business_rule_score < 0.8:
            result.add_issue(
                ValidationSeverity.LOW,
                'business_rules',
                f"Business rule compliance could be improved ({metrics.business_rule_score:.2f})",
                recommendation="Add constraints, default values, or validation rules"
            )
        
        # Confidence-based issues
        if result.confidence < 0.7:
            result.add_issue(
                ValidationSeverity.MEDIUM,
                'prediction_uncertainty',
                f"Low prediction confidence ({result.confidence:.2f})",
                recommendation="Model may need more training data for this scenario"
            )
    
    def train_model(self) -> Dict[str, Any]:
        """Enhanced model training with cross-validation and hyperparameter tuning"""
        
        with self._training_lock:
            try:
                self.model_status = ModelStatus.TRAINING
                logger.info(f"Starting enhanced {self.config.model_type} model training...")
                
                # Generate comprehensive training data
                training_data = self.generate_training_data_from_mappings()
                
                if training_data.empty:
                    raise ValueError("No training data generated")
                
                logger.info(f"Generated {len(training_data)} training samples")
                
                # Prepare features and labels
                X = self.prepare_features(training_data)
                y = self.label_encoder.fit_transform(training_data['outcome'])
                
                # Enhanced train-test split with stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=self.config.validation_split,
                    random_state=self.config.random_state,
                    stratify=y
                )
                
                # Feature scaling
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Cross-validation for model selection
                cv_scores = cross_val_score(
                    self.model, X_train_scaled, y_train, 
                    cv=5, scoring='f1_weighted'
                )
                
                logger.info(f"Cross-validation scores: {cv_scores}")
                logger.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Hyperparameter tuning (simplified for demo)
                if self.config.model_type == 'random_forest':
                    param_grid = {
                        'n_estimators': [150, 200, 250],
                        'max_depth': [10, 15, 20],
                        'min_samples_split': [3, 5, 7]
                    }
                else:  # gradient_boosting
                    param_grid = {
                        'n_estimators': [100, 150, 200],
                        'max_depth': [6, 8, 10],
                        'learning_rate': [0.05, 0.1, 0.15]
                    }
                
                # Grid search with limited scope for performance
                grid_search = GridSearchCV(
                    self.model, param_grid, 
                    cv=3, scoring='f1_weighted',
                    n_jobs=-1, verbose=1
                )
                
                logger.info("Performing hyperparameter optimization...")
                grid_search.fit(X_train_scaled, y_train)
                
                # Use best model
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
                
                # Final evaluation
                y_pred = self.model.predict(X_test_scaled)
                y_pred_proba = self.model.predict_proba(X_test_scaled)
                
                # Enhanced metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Feature importance analysis
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Class distribution analysis
                unique, counts = np.unique(y_train, return_counts=True)
                class_distribution = dict(zip(
                    self.label_encoder.inverse_transform(unique),
                    counts
                ))
                
                # Store comprehensive metrics
                self.model_metrics = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': grid_search.best_params_,
                    'training_samples': len(training_data),
                    'feature_count': len(X.columns),
                    'confusion_matrix': conf_matrix.tolist(),
                    'feature_importance': feature_importance,
                    'top_features': dict(sorted_features[:5]),
                    'class_distribution': class_distribution,
                    'model_type': self.config.model_type,
                    'trained_timestamp': datetime.now().isoformat()
                }
                
                # Update training history
                self.training_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_score': cv_scores.mean(),
                    'training_samples': len(training_data),
                    'mapping_count': len(self.mapping_manager.get_all_mappings()),
                    'model_type': self.config.model_type
                })
                
                # Determine final status
                if accuracy >= self.config.accuracy_threshold and f1 >= self.config.accuracy_threshold:
                    self.model_status = ModelStatus.TRAINED
                    logger.info(f"✅ Model training successful! Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                else:
                    self.model_status = ModelStatus.ERROR
                    logger.warning(f"⚠️ Model performance below threshold. Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                
                # Enhanced training report
                self._print_training_report(accuracy, f1, cv_scores, sorted_features, class_distribution)
                
                # Save model
                self.save_model()
                
                return self.model_metrics
                
            except Exception as e:
                self.model_status = ModelStatus.ERROR
                logger.error(f"Model training failed: {e}")
                raise
    
    def _print_training_report(self, accuracy: float, f1: float, cv_scores: np.ndarray,
                              sorted_features: List[Tuple[str, float]], 
                              class_distribution: Dict[str, int]):
        """Print comprehensive training report"""
        
        print("\n" + "="*80)
        print("🎯 ENHANCED MODEL TRAINING RESULTS")
        print("="*80)
        print(f"Model Type: {self.config.model_type.upper()}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Cross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Training Samples: {self.model_metrics['training_samples']:,}")
        print(f"Features: {self.model_metrics['feature_count']}")
        print(f"Mappings: {len(self.mapping_manager.get_all_mappings())}")
        
        print(f"\n📊 Class Distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / sum(class_distribution.values())) * 100
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🎯 Top 5 Feature Importance:")
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.3f}")
        
        print(f"\n💡 Model Status: {self.model_status.value.upper()}")
        print("="*80)
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        
        diagnostics = {
            'model_info': {
                'status': self.model_status.value,
                'type': self.config.model_type,
                'is_trained': self.is_trained,
                'needs_retraining': self.needs_retraining
            },
            'performance_metrics': self.model_metrics,
            'training_history': self.training_history,
            'validation_stats': self.validation_stats,
            'configuration': {
                'max_training_samples': self.config.max_training_samples,
                'accuracy_threshold': self.config.accuracy_threshold,
                'auto_retrain': self.config.auto_retrain,
                'model_type': self.config.model_type,
                'use_class_weights': self.config.use_class_weights
            },
            'feature_info': {
                'encoders': {k: list(v.classes_) for k, v in self.feature_encoders.items()},
                'feature_count': len(self.feature_encoders),
                'scaler_type': type(self.scaler).__name__
            }
        }
        
        # Add health score
        if self.is_trained:
            last_accuracy = self.model_metrics.get('accuracy', 0)
            last_f1 = self.model_metrics.get('f1_score', 0)
            
            health_score = (last_accuracy + last_f1) / 2
            diagnostics['health_score'] = health_score
            
            if health_score >= 0.9:
                diagnostics['health_status'] = 'excellent'
            elif health_score >= 0.8:
                diagnostics['health_status'] = 'good'
            elif health_score >= 0.7:
                diagnostics['health_status'] = 'fair'
            else:
                diagnostics['health_status'] = 'poor'
        else:
            diagnostics['health_score'] = 0.0
            diagnostics['health_status'] = 'not_trained'
        
        return diagnostics
    
    def validate_mapping_batch(self, mapping_ids: List[str], 
                              source_db: DatabaseConnector,
                              target_db: DatabaseConnector,
                              parallel: bool = True) -> Dict[str, Any]:
        """Enhanced batch validation with parallel processing"""
        
        start_time = datetime.now()
        results = {
            'batch_id': f"batch_{int(start_time.timestamp())}",
            'timestamp': start_time.isoformat(),
            'mapping_count': len(mapping_ids),
            'parallel_processing': parallel,
            'results': {},
            'summary': {
                'total': len(mapping_ids),
                'successful': 0,
                'failed': 0,
                'warnings': 0,
                'errors': 0
            },
            'execution_time_seconds': 0.0,
            'performance_stats': {}
        }
        
        try:
            if parallel and len(mapping_ids) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(mapping_ids))) as executor:
                    future_to_mapping = {
                        executor.submit(self.validate_mapping, mapping_id, source_db, target_db): mapping_id
                        for mapping_id in mapping_ids
                    }
                    
                    for future in as_completed(future_to_mapping):
                        mapping_id = future_to_mapping[future]
                        try:
                            result = future.result()
                            results['results'][mapping_id] = result
                            
                            # Update summary
                            status = result.get('overall_result', 'ERROR')
                            if status == 'PASS':
                                results['summary']['successful'] += 1
                            elif status == 'FAIL':
                                results['summary']['failed'] += 1
                            elif status == 'WARNING':
                                results['summary']['warnings'] += 1
                            else:
                                results['summary']['errors'] += 1
                                
                        except Exception as e:
                            logger.error(f"Parallel validation failed for {mapping_id}: {e}")
                            results['results'][mapping_id] = {
                                'mapping_id': mapping_id,
                                'overall_result': 'ERROR',
                                'error': str(e)
                            }
                            results['summary']['errors'] += 1
            else:
                # Sequential processing
                for mapping_id in mapping_ids:
                    try:
                        result = self.validate_mapping(mapping_id, source_db, target_db)
                        results['results'][mapping_id] = result
                        
                        status = result.get('overall_result', 'ERROR')
                        if status == 'PASS':
                            results['summary']['successful'] += 1
                        elif status == 'FAIL':
                            results['summary']['failed'] += 1
                        elif status == 'WARNING':
                            results['summary']['warnings'] += 1
                        else:
                            results['summary']['errors'] += 1
                            
                    except Exception as e:
                        logger.error(f"Sequential validation failed for {mapping_id}: {e}")
                        results['results'][mapping_id] = {
                            'mapping_id': mapping_id,
                            'overall_result': 'ERROR',
                            'error': str(e)
                        }
                        results['summary']['errors'] += 1
            
            # Calculate execution time and performance stats
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            results['execution_time_seconds'] = execution_time
            
            results['performance_stats'] = {
                'avg_time_per_mapping': execution_time / len(mapping_ids),
                'validations_per_second': len(mapping_ids) / execution_time,
                'processing_mode': 'parallel' if parallel else 'sequential'
            }
            
            # Overall batch status
            if results['summary']['errors'] > 0:
                results['overall_status'] = 'ERROR'
            elif results['summary']['failed'] > 0:
                results['overall_status'] = 'FAIL'
            elif results['summary']['warnings'] > 0:
                results['overall_status'] = 'WARNING'
            else:
                results['overall_status'] = 'PASS'
            
            logger.info(f"Batch validation completed: {results['summary']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
            return results
    
    def save_model(self, filepath: str = None) -> bool:
        """Enhanced model saving with versioning"""
        try:
            if filepath is None:
                models_dir = Path('/app/models')
                models_dir.mkdir(exist_ok=True)
                
                # Create versioned filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = models_dir / f'dynamic_validator_v2_{timestamp}.pkl'
                
                # Also save as latest
                latest_path = models_dir / 'dynamic_validator_latest.pkl'
            else:
                latest_path = None
            
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_encoders': self.feature_encoders,
                'scaler': self.scaler,
                'model_status': self.model_status.value,
                'model_metrics': self.model_metrics,
                'training_history': self.training_history,
                'validation_stats': self.validation_stats,
                'configuration': {
                    'model_type': self.config.model_type,
                    'max_training_samples': self.config.max_training_samples,
                    'accuracy_threshold': self.config.accuracy_threshold,
                    'auto_retrain': self.config.auto_retrain
                },
                'mapping_count': len(self.mapping_manager.get_all_mappings()),
                'saved_timestamp': datetime.now().isoformat(),
                'version': '2.1'
            }
            
            # Save versioned model
            with open(filepath, 'wb') as f:
                joblib.dump(model_data, f, compress=3)
            
            # Save as latest
            if latest_path:
                with open(latest_path, 'wb') as f:
                    joblib.dump(model_data, f, compress=3)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """Enhanced model loading with version compatibility"""
        try:
            if filepath is None:
                # Try to load latest first
                latest_path = Path('/app/models/dynamic_validator_latest.pkl')
                if latest_path.exists():
                    filepath = latest_path
                else:
                    # Fallback to old naming convention
                    filepath = Path('/app/models/dynamic_validator_model.pkl')
            
            if not os.path.exists(filepath):
                logger.info("No existing model file found")
                return False
            
            logger.info(f"Loading model from {filepath}")
            
            with open(filepath, 'rb') as f:
                if str(filepath).endswith('.pkl'):
                    model_data = joblib.load(f)
                else:
                    model_data = pickle.load(f)
            
            # Version compatibility check
            version = model_data.get('version', '1.0')
            logger.info(f"Loading model version {version}")
            
            # Load model components
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_encoders = model_data['feature_encoders']
            self.scaler = model_data.get('scaler', RobustScaler())
            self.model_metrics = model_data.get('model_metrics', {})
            self.training_history = model_data.get('training_history', [])
            self.validation_stats = model_data.get('validation_stats', {
                'total_validations': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'average_execution_time': 0.0,
                'last_reset': datetime.now().isoformat()
            })
            
            # Update configuration if available
            if 'configuration' in model_data:
                config_data = model_data['configuration']
                if hasattr(self.config, 'model_type'):
                    self.config.model_type = config_data.get('model_type', self.config.model_type)
            
            # Update status
            self.model_status = ModelStatus.TRAINED
            
            mapping_count = model_data.get('mapping_count', 0)
            saved_time = model_data.get('saved_timestamp', 'unknown')
            
            logger.info(f"✅ Model loaded successfully (v{version}, {mapping_count} mappings, saved: {saved_time})")
            
            # Check if retraining is needed
            if self.needs_retraining:
                logger.warning("⚠️ Model may need retraining due to significant changes")
                self.model_status = ModelStatus.OUTDATED
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_status = ModelStatus.ERROR
            return False

# Pydantic models for Enhanced API
class EnhancedValidationRequest(BaseModel):
    mapping_ids: List[str] = Field(..., description="List of mapping IDs to validate")
    include_details: bool = Field(True, description="Include detailed validation information")
    include_issues: bool = Field(True, description="Include issue analysis")
    parallel: bool = Field(False, description="Run validations in parallel")
    
    @validator('mapping_ids')
    def validate_mapping_ids(cls, v):
        if not v:
            raise ValueError('At least one mapping ID is required')
        return v

class ModelTrainingRequest(BaseModel):
    force_retrain: bool = Field(False, description="Force retraining even if not needed")
    model_type: Optional[Literal['random_forest', 'gradient_boosting']] = Field(None, description="Model type to use")
    max_samples: Optional[int] = Field(None, description="Maximum training samples")

class ModelDiagnosticsResponse(BaseModel):
    model_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    training_history: List[Dict[str, Any]]
    validation_stats: Dict[str, Any]
    configuration: Dict[str, Any]
    feature_info: Dict[str, Any]
    health_score: float
    health_status: str

    # Missing methods from original implementation
    def generate_training_data_from_mappings(self, samples_per_mapping: int = None) -> pd.DataFrame:
        """Generate comprehensive training data based on all available mappings"""
        mappings = self.mapping_manager.get_all_mappings()
        if not mappings:
            raise ValueError("No mappings available for training data generation")
        
        # Auto-calculate samples per mapping
        if samples_per_mapping is None:
            samples_per_mapping = max(50, min(500, self.config.max_training_samples // len(mappings)))
        
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
        """Generate training samples for a specific mapping"""
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
        """Generate a single validation scenario with realistic parameters"""
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
    
    def _determine_scenario_outcome(self, **kwargs) -> str:
        """Determine the outcome of a validation scenario based on multiple factors"""
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
        """Prepare features for model training or prediction"""
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
    
    def validate_mapping(self, mapping_id: str, source_db: DatabaseConnector, 
                        target_db: DatabaseConnector) -> Dict[str, Any]:
        """Validate a complete mapping between source and target tables"""
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
                        
                        results['column_validations'][f"{source_col}->{target_col}"] = validation_result.to_dict()
                        
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
        """Perform table-level validation checks"""
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
        """Generate recommendations based on validation results"""
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
        
        if not recommendations:
            recommendations.append("Validation completed successfully with no major issues detected.")
        
        return recommendations

# Export enhanced components
__all__ = [
    'DynamicDataMappingValidator',
    'ValidationStatus',
    'ModelStatus',
    'ValidationSeverity',
    'ValidationMetrics',
    'ValidationResult',
    'ValidationIssue',
    'ModelConfiguration',
    'EnhancedValidationRequest',
    'ModelTrainingRequest',
    'ModelDiagnosticsResponse'
]
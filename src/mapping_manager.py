# =============================================================================
# src/mapping_manager.py - Dynamic Mapping Management
# Dynamic Data Mapping Validator
# =============================================================================

import json
import os
import glob
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class MappingStatus(Enum):
    """Enumeration for mapping validation status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    ERROR = "error"

@dataclass
class MappingValidationResult:
    """Result of mapping validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    mapping_id: str
    
class MappingConflictError(Exception):
    """Raised when mapping conflicts occur"""
    pass

class MappingValidationError(Exception):
    """Raised when mapping validation fails"""
    pass

class MappingManager:
    """
    Manages dynamic mapping configurations with advanced features:
    - Hot reloading of mappings
    - Validation and conflict detection
    - Version management
    - Backup and restore
    - Import/export functionality
    """
    
    def __init__(self, mappings_directory: str = '/app/mappings'):
        """
        Initialize the mapping manager
        
        Args:
            mappings_directory: Path to directory containing mapping files
        """
        self.mappings_directory = Path(mappings_directory)
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self.mapping_metadata: Dict[str, Dict[str, Any]] = {}
        self.watchers: List[callable] = []
        
        # Ensure directory exists
        self.mappings_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize
        self._load_all_mappings()
        
        logger.info(f"MappingManager initialized with directory: {self.mappings_directory}")
    
    def add_change_watcher(self, callback: callable):
        """
        Add a callback function to be called when mappings change
        
        Args:
            callback: Function to call when mappings change
        """
        self.watchers.append(callback)
        logger.debug(f"Added change watcher: {callback.__name__}")
    
    def _notify_watchers(self, event: str, mapping_id: str):
        """Notify all watchers of mapping changes"""
        for watcher in self.watchers:
            try:
                watcher(event, mapping_id)
            except Exception as e:
                logger.error(f"Error in watcher {watcher.__name__}: {e}")
    
    def _load_all_mappings(self):
        """Load all mapping files from the mappings directory"""
        if not self.mappings_directory.exists():
            logger.warning(f"Mappings directory does not exist: {self.mappings_directory}")
            return
        
        # Clear existing mappings
        self.mappings.clear()
        self.mapping_metadata.clear()
        
        # Find all JSON files
        pattern = str(self.mappings_directory / "*.json")
        mapping_files = glob.glob(pattern)
        
        loaded_count = 0
        error_count = 0
        
        for file_path in mapping_files:
            try:
                mapping_data = self._load_mapping_file(file_path)
                if mapping_data:
                    loaded_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to load mapping file {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} mappings successfully, {error_count} errors")
        self._notify_watchers("bulk_load", "all")
    
    def _load_mapping_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a single mapping file
        
        Args:
            file_path: Path to the mapping file
            
        Returns:
            Mapping data if successful, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            # Validate mapping structure
            validation_result = self._validate_mapping_structure(mapping_data)
            if not validation_result.is_valid:
                logger.error(f"Invalid mapping in {file_path}: {validation_result.errors}")
                return None
            
            mapping_id = mapping_data.get('mapping_id')
            if not mapping_id:
                logger.error(f"Mapping file {file_path} missing mapping_id")
                return None
            
            # Check for conflicts
            if mapping_id in self.mappings:
                logger.warning(f"Duplicate mapping_id '{mapping_id}' in {file_path}")
                return None
            
            # Store mapping
            self.mappings[mapping_id] = mapping_data
            
            # Store metadata
            file_stats = os.stat(file_path)
            self.mapping_metadata[mapping_id] = {
                'file_path': file_path,
                'file_size': file_stats.st_size,
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
                'loaded_time': datetime.now(),
                'checksum': self._calculate_checksum(mapping_data),
                'status': MappingStatus.ACTIVE,
                'validation_result': validation_result
            }
            
            logger.debug(f"Loaded mapping: {mapping_id} from {file_path}")
            return mapping_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading mapping file {file_path}: {e}")
            return None
    
    def _validate_mapping_structure(self, mapping_data: Dict[str, Any]) -> MappingValidationResult:
        """
        Validate that a mapping has the required structure and valid data
        
        Args:
            mapping_data: The mapping data to validate
            
        Returns:
            MappingValidationResult with validation details
        """
        errors = []
        warnings = []
        mapping_id = mapping_data.get('mapping_id', 'unknown')
        
        # Required top-level fields
        required_fields = ['mapping_id', 'name', 'source', 'target', 'column_mappings']
        for field in required_fields:
            if field not in mapping_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate mapping_id format
        if 'mapping_id' in mapping_data:
            mapping_id = mapping_data['mapping_id']
            if not isinstance(mapping_id, str) or not mapping_id.strip():
                errors.append("mapping_id must be a non-empty string")
            elif not mapping_id.replace('_', '').replace('-', '').isalnum():
                warnings.append("mapping_id should contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate source and target structure
        for endpoint in ['source', 'target']:
            if endpoint in mapping_data:
                endpoint_data = mapping_data[endpoint]
                if not isinstance(endpoint_data, dict):
                    errors.append(f"{endpoint} must be an object")
                    continue
                
                required_endpoint_fields = ['database', 'schema', 'table']
                for field in required_endpoint_fields:
                    if field not in endpoint_data:
                        errors.append(f"Missing required field {field} in {endpoint}")
                    elif not isinstance(endpoint_data[field], str) or not endpoint_data[field].strip():
                        errors.append(f"{endpoint}.{field} must be a non-empty string")
        
        # Validate column mappings
        if 'column_mappings' in mapping_data:
            column_mappings = mapping_data['column_mappings']
            if not isinstance(column_mappings, dict):
                errors.append("column_mappings must be an object")
            elif not column_mappings:
                warnings.append("column_mappings is empty")
            else:
                for source_col, mapping_info in column_mappings.items():
                    if not isinstance(mapping_info, dict):
                        errors.append(f"Column mapping {source_col} must be an object")
                        continue
                    
                    if 'target_column' not in mapping_info:
                        errors.append(f"Column mapping {source_col} missing target_column")
                    
                    # Validate transformation type
                    transformation = mapping_info.get('transformation', 'direct')
                    valid_transformations = ['direct', 'lookup', 'custom', 'calculated', 'split', 'concatenate']
                    if transformation not in valid_transformations:
                        warnings.append(f"Unknown transformation type '{transformation}' in {source_col}")
                    
                    # Validate data types
                    for type_field in ['source_type', 'target_type']:
                        if type_field in mapping_info:
                            data_type = mapping_info[type_field]
                            if not isinstance(data_type, str):
                                errors.append(f"{source_col}.{type_field} must be a string")
        
        # Validate validation rules
        if 'validation_rules' in mapping_data:
            validation_rules = mapping_data['validation_rules']
            if not isinstance(validation_rules, dict):
                errors.append("validation_rules must be an object")
            else:
                # Check tolerance values
                for tolerance_field in ['row_count_tolerance', 'null_value_tolerance']:
                    if tolerance_field in validation_rules:
                        tolerance = validation_rules[tolerance_field]
                        if not isinstance(tolerance, (int, float)) or tolerance < 0 or tolerance > 1:
                            errors.append(f"{tolerance_field} must be a number between 0 and 1")
        
        # Validate version if present
        if 'version' in mapping_data:
            version = mapping_data['version']
            if not isinstance(version, str):
                warnings.append("version should be a string")
        
        is_valid = len(errors) == 0
        return MappingValidationResult(is_valid, errors, warnings, mapping_id)
    
    def _calculate_checksum(self, mapping_data: Dict[str, Any]) -> str:
        """Calculate MD5 checksum of mapping data"""
        mapping_str = json.dumps(mapping_data, sort_keys=True)
        return hashlib.md5(mapping_str.encode()).hexdigest()
    
    def get_mapping(self, mapping_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific mapping by ID
        
        Args:
            mapping_id: The mapping identifier
            
        Returns:
            Mapping data if found, None otherwise
        """
        return self.mappings.get(mapping_id)
    
    def get_all_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded mappings
        
        Returns:
            Dictionary of all mappings
        """
        return self.mappings.copy()
    
    def get_mapping_metadata(self, mapping_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific mapping
        
        Args:
            mapping_id: The mapping identifier
            
        Returns:
            Mapping metadata if found, None otherwise
        """
        return self.mapping_metadata.get(mapping_id)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all mappings"""
        return self.mapping_metadata.copy()
    
    def add_mapping(self, mapping_data: Dict[str, Any], save_to_file: bool = True, 
                   overwrite: bool = False) -> bool:
        """
        Add a new mapping
        
        Args:
            mapping_data: The mapping configuration
            save_to_file: Whether to save to disk
            overwrite: Whether to overwrite existing mappings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate mapping structure
            validation_result = self._validate_mapping_structure(mapping_data)
            if not validation_result.is_valid:
                raise MappingValidationError(f"Invalid mapping: {validation_result.errors}")
            
            mapping_id = mapping_data['mapping_id']
            
            # Check for conflicts
            if mapping_id in self.mappings and not overwrite:
                raise MappingConflictError(f"Mapping {mapping_id} already exists")
            
            # Check for table conflicts
            if not overwrite:
                conflicts = self._check_table_conflicts(mapping_data)
                if conflicts:
                    raise MappingConflictError(f"Table conflicts detected: {conflicts}")
            
            # Add to memory
            old_mapping = self.mappings.get(mapping_id)
            self.mappings[mapping_id] = mapping_data
            
            # Save to file if requested
            file_path = None
            if save_to_file:
                file_path = self.mappings_directory / f"{mapping_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_data, f, indent=2)
                logger.info(f"Saved mapping {mapping_id} to {file_path}")
            
            # Update metadata
            self.mapping_metadata[mapping_id] = {
                'file_path': str(file_path) if file_path else None,
                'file_size': len(json.dumps(mapping_data)),
                'modified_time': datetime.now(),
                'loaded_time': datetime.now(),
                'checksum': self._calculate_checksum(mapping_data),
                'status': MappingStatus.ACTIVE,
                'validation_result': validation_result
            }
            
            # Notify watchers
            event = "mapping_updated" if old_mapping else "mapping_added"
            self._notify_watchers(event, mapping_id)
            
            logger.info(f"Added mapping: {mapping_id}")
            return True
            
        except (MappingValidationError, MappingConflictError) as e:
            logger.error(f"Failed to add mapping: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding mapping: {e}")
            return False
    
    def update_mapping(self, mapping_id: str, mapping_data: Dict[str, Any]) -> bool:
        """
        Update an existing mapping
        
        Args:
            mapping_id: The mapping identifier
            mapping_data: The updated mapping configuration
            
        Returns:
            True if successful, False otherwise
        """
        if mapping_id not in self.mappings:
            logger.error(f"Mapping {mapping_id} not found")
            return False
        
        # Ensure mapping_id matches
        mapping_data['mapping_id'] = mapping_id
        
        return self.add_mapping(mapping_data, save_to_file=True, overwrite=True)
    
    def delete_mapping(self, mapping_id: str, delete_file: bool = True) -> bool:
        """
        Delete a mapping
        
        Args:
            mapping_id: The mapping identifier
            delete_file: Whether to delete the file from disk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if mapping_id not in self.mappings:
                logger.warning(f"Mapping {mapping_id} not found")
                return False
            
            # Remove from memory
            del self.mappings[mapping_id]
            
            # Delete file if requested and exists
            if delete_file and mapping_id in self.mapping_metadata:
                file_path = self.mapping_metadata[mapping_id].get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted mapping file: {file_path}")
            
            # Remove metadata
            if mapping_id in self.mapping_metadata:
                del self.mapping_metadata[mapping_id]
            
            # Notify watchers
            self._notify_watchers("mapping_deleted", mapping_id)
            
            logger.info(f"Deleted mapping: {mapping_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting mapping {mapping_id}: {e}")
            return False
    
    def _check_table_conflicts(self, mapping_data: Dict[str, Any]) -> List[str]:
        """
        Check for table conflicts with existing mappings
        
        Args:
            mapping_data: The mapping to check
            
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        new_mapping_id = mapping_data['mapping_id']
        
        source = mapping_data.get('source', {})
        target = mapping_data.get('target', {})
        
        new_source_table = f"{source.get('database')}.{source.get('schema')}.{source.get('table')}"
        new_target_table = f"{target.get('database')}.{target.get('schema')}.{target.get('table')}"
        
        for existing_id, existing_mapping in self.mappings.items():
            if existing_id == new_mapping_id:
                continue
            
            existing_source = existing_mapping.get('source', {})
            existing_target = existing_mapping.get('target', {})
            
            existing_source_table = f"{existing_source.get('database')}.{existing_source.get('schema')}.{existing_source.get('table')}"
            existing_target_table = f"{existing_target.get('database')}.{existing_target.get('schema')}.{existing_target.get('table')}"
            
            # Check for same source-target combination
            if new_source_table == existing_source_table and new_target_table == existing_target_table:
                conflicts.append(f"Same source-target combination as mapping '{existing_id}'")
        
        return conflicts
    
    def get_mappings_for_table(self, database: str, schema: str, table: str, 
                              endpoint_type: str = 'source') -> List[Dict[str, Any]]:
        """
        Get all mappings that involve a specific table
        
        Args:
            database: Database name
            schema: Schema name
            table: Table name
            endpoint_type: 'source' or 'target'
            
        Returns:
            List of matching mappings
        """
        matching_mappings = []
        
        for mapping_id, mapping_data in self.mappings.items():
            endpoint_data = mapping_data.get(endpoint_type, {})
            if (endpoint_data.get('database') == database and 
                endpoint_data.get('schema') == schema and 
                endpoint_data.get('table') == table):
                matching_mappings.append(mapping_data)
        
        return matching_mappings
    
    def reload_mappings(self) -> Dict[str, Any]:
        """
        Reload all mappings from disk
        
        Returns:
            Summary of reload operation
        """
        old_count = len(self.mappings)
        old_mappings = set(self.mappings.keys())
        
        self._load_all_mappings()
        
        new_count = len(self.mappings)
        new_mappings = set(self.mappings.keys())
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'old_count': old_count,
            'new_count': new_count,
            'added': list(new_mappings - old_mappings),
            'removed': list(old_mappings - new_mappings),
            'unchanged': list(old_mappings & new_mappings)
        }
        
        logger.info(f"Reloaded mappings: {old_count} -> {new_count}")
        self._notify_watchers("bulk_reload", "all")
        
        return summary
    
    def export_mappings(self, export_path: str, include_metadata: bool = False) -> bool:
        """
        Export all mappings to a single file
        
        Args:
            export_path: Path to export file
            include_metadata: Whether to include metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0',
                'mapping_count': len(self.mappings),
                'mappings': self.mappings
            }
            
            if include_metadata:
                # Convert datetime objects to strings for JSON serialization
                serializable_metadata = {}
                for mapping_id, metadata in self.mapping_metadata.items():
                    serializable_metadata[mapping_id] = {
                        **metadata,
                        'modified_time': metadata['modified_time'].isoformat() if metadata.get('modified_time') else None,
                        'loaded_time': metadata['loaded_time'].isoformat() if metadata.get('loaded_time') else None,
                        'status': metadata['status'].value if hasattr(metadata.get('status'), 'value') else str(metadata.get('status')),
                        'validation_result': {
                            'is_valid': metadata['validation_result'].is_valid,
                            'errors': metadata['validation_result'].errors,
                            'warnings': metadata['validation_result'].warnings,
                            'mapping_id': metadata['validation_result'].mapping_id
                        } if metadata.get('validation_result') else None
                    }
                export_data['metadata'] = serializable_metadata
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.mappings)} mappings to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting mappings: {e}")
            return False
    
    def import_mappings(self, import_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Import mappings from a file
        
        Args:
            import_path: Path to import file
            overwrite: Whether to overwrite existing mappings
            
        Returns:
            Summary of import operation
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if 'mappings' not in import_data:
                raise ValueError("Invalid import file: missing 'mappings' key")
            
            imported_mappings = import_data['mappings']
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_in_file': len(imported_mappings),
                'imported': [],
                'skipped': [],
                'errors': []
            }
            
            for mapping_id, mapping_data in imported_mappings.items():
                try:
                    if mapping_id in self.mappings and not overwrite:
                        summary['skipped'].append(mapping_id)
                        continue
                    
                    if self.add_mapping(mapping_data, save_to_file=True, overwrite=overwrite):
                        summary['imported'].append(mapping_id)
                    else:
                        summary['errors'].append(f"{mapping_id}: validation failed")
                        
                except Exception as e:
                    summary['errors'].append(f"{mapping_id}: {str(e)}")
            
            logger.info(f"Import complete: {len(summary['imported'])} imported, "
                       f"{len(summary['skipped'])} skipped, {len(summary['errors'])} errors")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error importing mappings: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'errors': [str(e)]
            }
    
    def backup_mappings(self, backup_dir: str) -> str:
        """
        Create a backup of all mappings
        
        Args:
            backup_dir: Directory to store backup
            
        Returns:
            Path to backup file
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"mappings_backup_{timestamp}.json"
        
        self.export_mappings(str(backup_file), include_metadata=True)
        
        logger.info(f"Created backup: {backup_file}")
        return str(backup_file)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded mappings
        
        Returns:
            Dictionary with mapping statistics
        """
        stats = {
            'total_mappings': len(self.mappings),
            'status_counts': {},
            'table_counts': {},
            'transformation_counts': {},
            'validation_summary': {
                'valid': 0,
                'invalid': 0,
                'warnings': 0
            }
        }
        
        # Count by status
        for metadata in self.mapping_metadata.values():
            status = metadata.get('status', MappingStatus.ACTIVE)
            status_str = status.value if hasattr(status, 'value') else str(status)
            stats['status_counts'][status_str] = stats['status_counts'].get(status_str, 0) + 1
        
        # Count tables and transformations
        for mapping_data in self.mappings.values():
            # Source tables
            source = mapping_data.get('source', {})
            source_table = f"{source.get('database', '')}.{source.get('schema', '')}.{source.get('table', '')}"
            stats['table_counts'][source_table] = stats['table_counts'].get(source_table, 0) + 1
            
            # Transformations
            for col_mapping in mapping_data.get('column_mappings', {}).values():
                transformation = col_mapping.get('transformation', 'direct')
                stats['transformation_counts'][transformation] = stats['transformation_counts'].get(transformation, 0) + 1
        
        # Validation summary
        for metadata in self.mapping_metadata.values():
            validation_result = metadata.get('validation_result')
            if validation_result:
                if validation_result.is_valid:
                    stats['validation_summary']['valid'] += 1
                else:
                    stats['validation_summary']['invalid'] += 1
                
                if validation_result.warnings:
                    stats['validation_summary']['warnings'] += 1
        
        return stats

# Export main class
__all__ = ['MappingManager', 'MappingStatus', 'MappingValidationResult', 'MappingConflictError', 'MappingValidationError']

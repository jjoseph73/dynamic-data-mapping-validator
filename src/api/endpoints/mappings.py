"""
Mappings Management API Endpoints
Handles CRUD operations for mapping configurations
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from pathlib import Path as FilePath

from ..validation_models import (
    MappingListResponse,
    MappingCreateRequest,
    MappingUpdateRequest,
    MappingResponse,
    ErrorResponse
)
from ...mapping_manager import MappingManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection will be handled by main app
def get_mapping_manager() -> MappingManager:
    # This will be overridden by the main app dependency
    pass

@router.get("/", response_model=MappingListResponse)
async def list_mappings(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of mappings to return"),
    offset: int = Query(0, ge=0, description="Number of mappings to skip"),
    search: Optional[str] = Query(None, max_length=100, description="Search term for mapping names"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Get list of all mapping configurations with optional filtering
    """
    try:
        logger.info(f"Listing mappings (limit: {limit}, offset: {offset}, search: {search})")
        
        # Get all mappings
        all_mappings = manager.get_all_mappings()
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            all_mappings = {
                name: config for name, config in all_mappings.items()
                if search_lower in name.lower() or 
                   search_lower in config.get('description', '').lower()
            }
        
        # Apply tag filter
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',')]
            all_mappings = {
                name: config for name, config in all_mappings.items()
                if any(tag in config.get('tags', []) for tag in tag_list)
            }
        
        # Convert to list and apply pagination
        mappings_list = [
            {
                "name": name,
                "config": config,
                "created_at": config.get('metadata', {}).get('created_at'),
                "updated_at": config.get('metadata', {}).get('updated_at'),
                "description": config.get('description'),
                "tags": config.get('tags', [])
            }
            for name, config in all_mappings.items()
        ]
        
        # Sort by name
        mappings_list.sort(key=lambda x: x['name'])
        
        # Apply pagination
        total_count = len(mappings_list)
        paginated_mappings = mappings_list[offset:offset + limit]
        
        return MappingListResponse(
            mappings=paginated_mappings,
            total_count=total_count,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error listing mappings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list mappings: {str(e)}")

@router.get("/{mapping_name}", response_model=MappingResponse)
async def get_mapping(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Get a specific mapping configuration by name
    """
    try:
        logger.info(f"Getting mapping: {mapping_name}")
        
        mapping_config = manager.get_mapping(mapping_name)
        
        if not mapping_config:
            raise HTTPException(status_code=404, detail=f"Mapping '{mapping_name}' not found")
        
        return MappingResponse(
            success=True,
            mapping_name=mapping_name,
            message="Mapping retrieved successfully",
            mapping_config=mapping_config,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mapping {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get mapping: {str(e)}")

@router.post("/", response_model=MappingResponse, status_code=201)
async def create_mapping(
    request: MappingCreateRequest,
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Create a new mapping configuration
    """
    try:
        logger.info(f"Creating mapping: {request.name}")
        
        # Check if mapping already exists
        if manager.get_mapping(request.name):
            raise HTTPException(status_code=409, detail=f"Mapping '{request.name}' already exists")
        
        # Prepare mapping config
        mapping_config = request.config.dict()
        mapping_config['description'] = request.description
        mapping_config['tags'] = request.tags
        mapping_config['metadata'] = {
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'version': '1.0'
        }
        
        # Save mapping
        success = manager.save_mapping(request.name, mapping_config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save mapping")
        
        return MappingResponse(
            success=True,
            mapping_name=request.name,
            message="Mapping created successfully",
            mapping_config=mapping_config,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating mapping {request.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create mapping: {str(e)}")

@router.put("/{mapping_name}", response_model=MappingResponse)
async def update_mapping(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping"),
    request: MappingUpdateRequest = None,
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Update an existing mapping configuration
    """
    try:
        logger.info(f"Updating mapping: {mapping_name}")
        
        # Check if mapping exists
        existing_config = manager.get_mapping(mapping_name)
        if not existing_config:
            raise HTTPException(status_code=404, detail=f"Mapping '{mapping_name}' not found")
        
        # Update configuration
        updated_config = existing_config.copy()
        
        if request.config:
            updated_config.update(request.config.dict())
        
        if request.description is not None:
            updated_config['description'] = request.description
            
        if request.tags is not None:
            updated_config['tags'] = request.tags
        
        # Update metadata
        if 'metadata' not in updated_config:
            updated_config['metadata'] = {}
        updated_config['metadata']['updated_at'] = datetime.utcnow().isoformat()
        
        # Save updated mapping
        success = manager.save_mapping(mapping_name, updated_config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update mapping")
        
        return MappingResponse(
            success=True,
            mapping_name=mapping_name,
            message="Mapping updated successfully",
            mapping_config=updated_config,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating mapping {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update mapping: {str(e)}")

@router.delete("/{mapping_name}", response_model=MappingResponse)
async def delete_mapping(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Delete a mapping configuration
    """
    try:
        logger.info(f"Deleting mapping: {mapping_name}")
        
        # Check if mapping exists
        if not manager.get_mapping(mapping_name):
            raise HTTPException(status_code=404, detail=f"Mapping '{mapping_name}' not found")
        
        # Delete mapping
        success = manager.delete_mapping(mapping_name)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete mapping")
        
        return MappingResponse(
            success=True,
            mapping_name=mapping_name,
            message="Mapping deleted successfully",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting mapping {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete mapping: {str(e)}")

@router.post("/{mapping_name}/duplicate", response_model=MappingResponse, status_code=201)
async def duplicate_mapping(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping to duplicate"),
    new_name: str = Query(..., min_length=1, max_length=128, description="Name for the duplicated mapping"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Duplicate an existing mapping configuration
    """
    try:
        logger.info(f"Duplicating mapping: {mapping_name} -> {new_name}")
        
        # Check if source mapping exists
        source_config = manager.get_mapping(mapping_name)
        if not source_config:
            raise HTTPException(status_code=404, detail=f"Source mapping '{mapping_name}' not found")
        
        # Check if target name already exists
        if manager.get_mapping(new_name):
            raise HTTPException(status_code=409, detail=f"Mapping '{new_name}' already exists")
        
        # Create duplicate config
        duplicate_config = source_config.copy()
        duplicate_config['metadata'] = {
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'version': '1.0',
            'duplicated_from': mapping_name
        }
        
        # Update description to indicate it's a duplicate
        original_desc = duplicate_config.get('description', '')
        duplicate_config['description'] = f"Copy of {mapping_name}: {original_desc}"
        
        # Save duplicate
        success = manager.save_mapping(new_name, duplicate_config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to duplicate mapping")
        
        return MappingResponse(
            success=True,
            mapping_name=new_name,
            message=f"Mapping duplicated successfully from '{mapping_name}'",
            mapping_config=duplicate_config,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating mapping {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to duplicate mapping: {str(e)}")

@router.post("/{mapping_name}/validate", response_model=Dict[str, Any])
async def validate_mapping_syntax(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Validate the syntax and structure of a mapping configuration
    """
    try:
        logger.info(f"Validating mapping syntax: {mapping_name}")
        
        # Get mapping
        mapping_config = manager.get_mapping(mapping_name)
        if not mapping_config:
            raise HTTPException(status_code=404, detail=f"Mapping '{mapping_name}' not found")
        
        # Validate mapping syntax
        validation_result = manager.validate_mapping_syntax(mapping_config)
        
        return {
            "mapping_name": mapping_name,
            "is_valid": validation_result.get('is_valid', False),
            "errors": validation_result.get('errors', []),
            "warnings": validation_result.get('warnings', []),
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating mapping syntax {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate mapping syntax: {str(e)}")

@router.get("/{mapping_name}/export")
async def export_mapping(
    mapping_name: str = Path(..., min_length=1, max_length=128, description="Name of the mapping"),
    format: str = Query("json", regex="^(json|yaml)$", description="Export format"),
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Export a mapping configuration in the specified format
    """
    try:
        logger.info(f"Exporting mapping: {mapping_name} in {format} format")
        
        # Get mapping
        mapping_config = manager.get_mapping(mapping_name)
        if not mapping_config:
            raise HTTPException(status_code=404, detail=f"Mapping '{mapping_name}' not found")
        
        if format == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=mapping_config,
                headers={"Content-Disposition": f"attachment; filename={mapping_name}.json"}
            )
        elif format == "yaml":
            import yaml
            from fastapi.responses import PlainTextResponse
            yaml_content = yaml.dump(mapping_config, default_flow_style=False)
            return PlainTextResponse(
                content=yaml_content,
                headers={"Content-Disposition": f"attachment; filename={mapping_name}.yaml"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting mapping {mapping_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export mapping: {str(e)}")

@router.post("/import", response_model=MappingResponse, status_code=201)
async def import_mapping(
    name: str = Query(..., min_length=1, max_length=128, description="Name for the imported mapping"),
    overwrite: bool = Query(False, description="Whether to overwrite if mapping exists"),
    # In a real implementation, you'd handle file upload here
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Import a mapping configuration from file
    """
    try:
        logger.info(f"Importing mapping: {name}")
        
        # Check if mapping exists and overwrite flag
        if manager.get_mapping(name) and not overwrite:
            raise HTTPException(status_code=409, detail=f"Mapping '{name}' already exists. Use overwrite=true to replace.")
        
        # In a real implementation, you'd parse the uploaded file here
        # For now, return a placeholder response
        return MappingResponse(
            success=True,
            mapping_name=name,
            message="Mapping import functionality not yet implemented",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing mapping {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import mapping: {str(e)}")

@router.get("/tags/", response_model=Dict[str, Any])
async def get_all_tags(
    manager: MappingManager = Depends(get_mapping_manager)
):
    """
    Get all unique tags used across mappings
    """
    try:
        logger.info("Getting all mapping tags")
        
        all_mappings = manager.get_all_mappings()
        all_tags = set()
        
        for config in all_mappings.values():
            tags = config.get('tags', [])
            all_tags.update(tags)
        
        return {
            "tags": sorted(list(all_tags)),
            "total_count": len(all_tags),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting all tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")
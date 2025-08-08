# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import json
import os
from pathlib import Path

# Import your application
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.api.app import app
from src.validation_engine import ValidationEngine
from src.mapping_manager import MappingManager
from src.database_connector import DatabaseConnector

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_column_mapping():
    """Sample column mapping for testing."""
    return {
        "source_column": "USER_ID",
        "target_column": "user_id",
        "source_type": "NUMBER(10)",
        "target_type": "INTEGER"
    }

@pytest.fixture
def sample_table_mapping():
    """Sample table mapping for testing."""
    return {
        "table_name": "users",
        "columns": [
            {
                "source_column": "USER_ID",
                "target_column": "user_id",
                "source_type": "NUMBER(10)",
                "target_type": "INTEGER"
            },
            {
                "source_column": "USERNAME",
                "target_column": "username",
                "source_type": "VARCHAR2(50)",
                "target_type": "VARCHAR(50)"
            },
            {
                "source_column": "EMAIL_ADDRESS",
                "target_column": "email",
                "source_type": "VARCHAR2(255)",
                "target_type": "VARCHAR(255)"
            }
        ]
    }

@pytest.fixture
def sample_batch_mappings():
    """Sample batch mappings for testing."""
    return [
        {
            "source_column": "USER_ID",
            "target_column": "user_id",
            "source_type": "NUMBER(10)",
            "target_type": "INTEGER"
        },
        {
            "source_column": "PRODUCT_ID",
            "target_column": "product_id",
            "source_type": "NUMBER(10)",
            "target_type": "BIGINT"
        },
        {
            "source_column": "PRICE",
            "target_column": "price",
            "source_type": "NUMBER(10,2)",
            "target_type": "DECIMAL(10,2)"
        }
    ]

@pytest.fixture
def mock_validation_engine():
    """Mock validation engine for testing."""
    with patch('src.api.endpoints.validation.validation_engine') as mock:
        mock.validate_column_mapping.return_value = {
            "is_valid": True,
            "confidence": 0.95,
            "recommendation": "Column mapping is valid"
        }
        mock.validate_table_mapping.return_value = {
            "is_valid": True,
            "confidence": 0.92,
            "recommendation": "Table mapping is valid",
            "issues": []
        }
        mock.validate_batch_mappings.return_value = [
            {
                "is_valid": True,
                "confidence": 0.95,
                "recommendation": "Valid mapping"
            }
        ]
        yield mock

@pytest.fixture
def mock_mapping_manager():
    """Mock mapping manager for testing."""
    with patch('src.api.endpoints.mappings.mapping_manager') as mock:
        mock.get_all_mappings.return_value = [
            {
                "name": "test_mapping",
                "config": {
                    "table_name": "test_table",
                    "columns": []
                }
            }
        ]
        mock.get_mapping.return_value = {
            "table_name": "test_table",
            "columns": []
        }
        mock.save_mapping.return_value = True
        mock.delete_mapping.return_value = True
        yield mock
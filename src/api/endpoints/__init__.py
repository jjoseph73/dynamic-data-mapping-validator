"""
API Endpoints Package
Dynamic Data Mapping Validator
"""

# Import all endpoint modules for easy access
from . import validation
from . import mappings
from . import model
from . import health

__all__ = [
    "validation",
    "mappings", 
    "model",
    "health"
]
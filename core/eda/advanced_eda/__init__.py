"""Advanced EDA Module.

This module provides advanced exploratory data analysis capabilities with:
- Intelligent domain detection and template recommendations
- Professional visualization and export systems
- Comprehensive session management and history tracking
"""

from .routes import router as advanced_eda_router
from .services import AdvancedEDAService

__all__ = ['advanced_eda_router', 'AdvancedEDAService']

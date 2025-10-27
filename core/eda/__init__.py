"""EDA package exports."""
# Conditionally import components to avoid dependency issues during testing

try:
    from .services import EDAService
    service_available = True
except ImportError:
    # pandas/numpy not available, skip service import
    service_available = False

try:
    from .routes import router as eda_router
    router_available = True
except ImportError:
    # FastAPI not available, skip router import
    router_available = False

# Build __all__ dynamically based on what's available
__all__ = []
if service_available:
    __all__.append("EDAService")
if router_available:
    __all__.append("eda_router")
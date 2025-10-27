"""EDA API routes.

Provides a small FastAPI router that exposes EDA endpoints using the EDAService.
"""
from typing import Any, Dict
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from core.eda.services import EDAService
from core.data_ingestion.dependencies import get_data_service, require_data_access
from core.database.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eda", tags=["EDA"])


@router.get("/api/sources/{source_id}/preview")
async def eda_preview(
    source_id: str,
    sample_size: int = Query(100, ge=1, le=1000),
    mode: str = Query("head", pattern="^(head|first_last)$"),
    force_refresh: bool = Query(False, description="Force regeneration of the preview snapshot"),
    current_user: User = Depends(require_data_access),
    service: EDAService = Depends(get_data_service)
):
    """Return a small preview and simple stats for a data source."""
    try:
        # service here is actually a DataIngestionService provider; create EDAService with same session
        eda_service = EDAService(service.session)
        result = await eda_service.preview_source(
            source_id,
            sample_size=sample_size,
            mode=mode,
            force_refresh=force_refresh,
        )
        
        # Use JSONResponse for consistent handling
        return JSONResponse(
            content=result,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logger.error(f"EDA preview error for {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sources/{source_id}/quality-report")
async def eda_quality_report(
    source_id: str,
    sample_size: int = Query(500, ge=1, le=1_000_000),
    current_user: User = Depends(require_data_access),
    service: EDAService = Depends(get_data_service)
):
    """Return comprehensive quality report with text analysis for a data source."""
    try:
        # service here is actually a DataIngestionService provider; create EDAService with same session
        eda_service = EDAService(service.session)
        result = await eda_service.quality_report(source_id, sample_size=sample_size)
        
        # Use JSONResponse to handle large responses properly
        return JSONResponse(
            content=result,
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"EDA quality report error for {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
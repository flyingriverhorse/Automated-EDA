"""Advanced EDA Routes.

FastAPI routes for the advanced EDA system with two-tab architecture.
Handles domain-specific analysis and LLM-powered code generation.
"""
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.data_ingestion.dependencies import get_data_service, require_data_access
from core.data_ingestion.service import DataIngestionService
from core.database.models import User
from core.utils.logging_utils import log_data_action
from core.eda.services import EDAService
from core.eda.security import get_persistent_sandbox_manager
from .services import AdvancedEDAService
from .notebook_templates import NotebookTemplateManager
from .analysis_results import make_json_serializable

logger = logging.getLogger(__name__)

# Initialize templates (fallback if app-level templates not available)
templates = Jinja2Templates(directory="templates")


def get_template_engine(request: Request) -> Jinja2Templates:
    """Return the configured template engine, falling back to local instance."""
    return getattr(getattr(request.app, "state", object()), "templates", templates)

router = APIRouter(prefix="/advanced-eda", tags=["Advanced EDA"])


# Pydantic models for request/response
class DomainAnalysisRequest(BaseModel):
    template_key: str
    analysis_depth: str = "intermediate"
    focus_areas: List[str] = []
    dataset_info: Optional[Dict[str, Any]] = None


class LLMCodeRequest(BaseModel):
    query: str
    dataset_id: str
    analysis_style: str = "exploratory"
    use_context: bool = True
    use_templates: bool = True
    generate_comments: bool = True
    session_history: List[Dict[str, Any]] = []


class CodeExecutionRequest(BaseModel):
    code: str
    dataset_id: str
    context: str = "notebook"


class ExportRequest(BaseModel):
    session_id: Optional[str] = None
    dataset_id: str
    cells: List[Dict[str, Any]] = []


# ============================================================================
# MAIN ADVANCED EDA PAGE ROUTES
# ============================================================================

@router.get("/", response_class=HTMLResponse)
async def advanced_eda_main_page(
    request: Request,
    source_id: Optional[str] = Query(None),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Render the main Advanced EDA page with two-tab system."""
    try:
        dataset_info = None
        dataset_name = "No Dataset"
        
        if source_id:
            try:
                eda_service = AdvancedEDAService(session=service.session)
                dataset_info = await eda_service.get_source_info(source_id)
                dataset_name = dataset_info.get('name', f'Dataset {source_id}') if dataset_info else f'Dataset {source_id}'
            except Exception as e:
                logger.warning(f"Error getting source info for {source_id}: {e}")
                # Continue loading the page even if source info fails
                dataset_name = f'Dataset {source_id}'
        
        template_engine = get_template_engine(request)

        return template_engine.TemplateResponse("eda/eda_main.html", {
            "request": request,
            "source_id": source_id,
            "dataset_name": dataset_name,
            "dataset_info": dataset_info,
            "current_user": current_user,
            "user": current_user,
            "title": "Advanced Exploratory Data Analysis"
        })
        
    except Exception as e:
        logger.error(f"Advanced EDA page error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LLM-POWERED EDA ROUTES
# ============================================================================

@router.post("/api/llm-code/{source_id}")
async def execute_llm_code(
    source_id: str,
    request_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Execute LLM-generated code for data analysis."""
    try:
        code = request_data.get('code')
        context = request_data.get('context', '')
        
        if not code:
            raise HTTPException(status_code=400, detail="code is required")
            
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.execute_code(source_id, code, context)
        
        if result["success"]:
            log_data_action("LLM_CODE_EXEC", 
                           details=f"source:{source_id},user:{current_user.id}")
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Code execution failed"))
        
    except Exception as e:
        logger.error(f"LLM code execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DATASET INFORMATION API
# ============================================================================

@router.get("/api/data-sources/{source_id}/info")
async def get_dataset_info(
    source_id: str,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get dataset information including name, description, and metadata."""
    try:
        eda_service = AdvancedEDAService(service.session)
        dataset_info = await eda_service.get_source_info(source_id)
        
        if dataset_info:
            return JSONResponse(content={
                "success": True,
                **dataset_info
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": "Dataset not found"
            }, status_code=404)
            
    except Exception as e:
        logger.error(f"Dataset info error: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)


# ============================================================================
# SESSION AND EXPORT MANAGEMENT
# ============================================================================

@router.post("/api/save-session")
async def save_eda_session(
    session_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Save Advanced EDA session data."""
    try:
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.save_session(session_data, str(current_user.id))
        
        if result["success"]:
            log_data_action("ADVANCED_SESSION_SAVE", 
                           details=f"session:{result['session_id']},user:{current_user.id}")
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Session save failed"))
        
    except Exception as e:
        logger.error(f"Session save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sessions")
async def get_eda_sessions(
    source_id: Optional[str] = Query(None),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get Advanced EDA session history for the current user."""
    try:
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.get_sessions(str(current_user.id), source_id)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get sessions"))
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/export-notebook")
async def export_notebook(
    export_data: ExportRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Export Advanced EDA session as Jupyter notebook."""
    try:
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.export_notebook(export_data.dict(), str(current_user.id))
        
        if result["success"]:
            log_data_action("ADVANCED_NOTEBOOK_EXPORT", 
                           details=f"session:{export_data.session_id},user:{current_user.id}")
            
            # Return the notebook file
            return FileResponse(
                path=result["file_path"],
                filename=result["filename"],
                media_type='application/json'
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))
        
    except Exception as e:
        logger.error(f"Notebook export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NOTEBOOK CODE EXECUTION
# ============================================================================

class CodeExecutionRequest(BaseModel):
    code: str
    context: Optional[str] = "notebook"


@router.post("/api/execute/{source_id}")
async def execute_notebook_code(
    source_id: str,
    request_data: CodeExecutionRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Execute Python code in notebook context with dataset access."""
    try:
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.execute_code(
            source_id=source_id, 
            code=request_data.code,
            context=request_data.context
        )
        
        if result["success"]:
            log_data_action("NOTEBOOK_EXECUTION", 
                           details=f"source:{source_id},lines:{len(request_data.code.split())},user:{current_user.id}")
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Code execution failed"))
        
    except Exception as e:
        logger.error(f"Notebook code execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COLUMN INSIGHTS API
# ============================================================================

@router.api_route("/api/column-insights/{source_id}", methods=["GET", "POST"])
async def get_column_insights(
    source_id: str,
    request: Request,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get column insights for column selection and filtering."""
    try:
        logger.info(f"Column insights called for source_id: {source_id}")
        preprocessing_payload = None
        base_mode = "auto"

        if request.method == "POST":
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    preprocessing_payload = payload.get("preprocessing")
                    base_mode = payload.get("base", "auto") or "auto"
            except Exception as exc:  # pragma: no cover - defensive parsing
                logger.warning("Invalid preprocessing payload for column insights: %s", exc)
        
        eda_service = AdvancedEDAService(service.session)
        result = await eda_service.get_column_insights(
            source_id,
            preprocessing_config=preprocessing_payload,
            base=base_mode,
        )
        
        if result.get('success'):
            return JSONResponse(content=result)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": result.get('error', 'Column insights failed'),
                    "message": result.get('message', 'Unable to get column insights')
                }
            )
    except Exception as e:
        logger.error(f"Column insights error for {source_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "message": "Failed to process column insights request"
            }
        )


# ============================================================================
# DOMAIN DETECTION AND ANALYSIS
# ============================================================================

@router.get("/api/detect-domain/{source_id}")
async def detect_dataset_domain(
    source_id: str,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Detect the domain/type of the dataset for template suggestions."""
    try:
        logger.info(f"Domain detection called for source_id: {source_id}")
        
        # Use the advanced EDA service with domain analyzer
        eda_service = AdvancedEDAService(service.session)
        
        # Use the sophisticated domain detection
        result = await eda_service.detect_domain(source_id)
        
        if result.get('success'):
            return JSONResponse(content={
                "success": True,
                "domain": result.get('primary_domain'),  # Frontend expects 'domain' key
                "primary_domain": result.get('primary_domain'),
                "primary_confidence": result.get('primary_confidence', result.get('confidence', 0.5)),
                "confidence": result.get('confidence', 0.5),  # Keep for backward compatibility
                "secondary_domains": result.get('secondary_domains', []),
                "all_scores": result.get('domain_scores', {}),
                "domain_scores": result.get('domain_scores', {}),  # Add explicit domain_scores
                "patterns": result.get('detected_patterns', {}),
                "recommendations": result.get('recommendations', []),
                "detected_at": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Domain detection failed'))
        
    except FileNotFoundError as e:
        logger.warning(f"Data source not found: {source_id}")
        return JSONResponse(content={
            "success": False,
            "error": f"Data source {source_id} not found"
        }, status_code=404)
    except Exception as e:
        logger.error(f"Domain detection error: {e}")
        # Check if it's a data source not found error
        if "not found" in str(e).lower() or "filenotfound" in str(e).lower():
            return JSONResponse(content={
                "success": False,
                "error": f"Data source {source_id} not found"
            }, status_code=404)
        else:
            return JSONResponse(content={
                "success": False,
                "error": str(e)
            }, status_code=500)


@router.get("/api/domain-analysis/{source_id}")
async def get_domain_analysis(
    source_id: str,
    template_name: str = Query(default="data_quality_structure", description="Analysis template to use"),
    analysis_depth: str = Query(default="intermediate", description="Analysis depth: basic, intermediate, advanced"),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get domain-specific analysis insights for recommendations."""
    try:
        logger.info(f"Domain analysis called for source_id: {source_id}, template: {template_name}")
        
        eda_service = AdvancedEDAService(service.session)
        
        # Generate domain analysis 
        result = await eda_service.generate_domain_analysis(
            source_id=source_id,
            template_name=template_name,
            analysis_depth=analysis_depth
        )
        
        if result.get('success'):
            return JSONResponse(content={
                "success": True,
                "template_name": result.get('template_name'),
                "analysis_depth": result.get('analysis_depth'),
                "template_summary": result.get('template_summary', {}),
                "template_sections": result.get('template_sections', []),
                "generated_at": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": result.get('error', 'Domain analysis failed')
            }, status_code=500)
        
    except Exception as e:
        logger.error(f"Domain analysis error: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

# ============================================================================
# ANALYSIS GENERATION AND EXECUTION
# ============================================================================

class GenerateAndRunRequest(BaseModel):
    analysis_type: str

class CustomCodeRequest(BaseModel):
    code: str

@router.post("/api/generate-and-run/{source_id}")
async def generate_and_run_analysis(
    source_id: str,
    request_data: GenerateAndRunRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Generate and execute analysis code for a specific analysis type."""
    try:
        eda_service = AdvancedEDAService(service.session)
        analysis_type = request_data.analysis_type
        
        # Map analysis types to template approaches or generate with LLM
        analysis_mapping = {
            'data_quality_structure': 'data_quality_structure',
            'univariate_numeric': 'univariate_numeric',
            'univariate_categorical': 'univariate_categorical',
            'bivariate_multivariate': 'bivariate_multivariate',
            'advanced_outlier_detection': 'advanced_outlier_detection',
            'time_series_exploration': 'time_series_exploration',
            'relationship_exploration': 'relationship_exploration',
            'data_profiling': 'data_profiling',
            'missing_data_analysis': 'missing_data_analysis',
            'correlation_analysis': 'correlation_analysis',
            'distribution_analysis': 'distribution_analysis',
            'outlier_detection': 'outlier_detection',
            'network_analysis': 'relationship_exploration',
            'entity_relationship_network': 'relationship_exploration',
            'custom_code': 'custom_code'
        }
        
        template_name = analysis_mapping.get(analysis_type, 'data_profiling')
        
        # Use the separate function for code generation and execution
        result = await eda_service.generate_and_execute_analysis_code(
            source_id=source_id,
            template_name=template_name
        )
        
        # Extract code and results from the analysis
        if result.get('success'):
            return JSONResponse(content={
                "success": True,
                "analysis_type": analysis_type,
                "template_name": template_name,
                "code": result.get('code', ''),
                "execution_result": result.get('execution_result', {}),
                "output": result.get('output', ''),
                "generated_at": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": result.get('error', 'Analysis generation failed'),
                "analysis_type": analysis_type
            }, status_code=500)
        
    except FileNotFoundError as e:
        logger.warning(f"Data source not found for analysis generation: {source_id}")
        return JSONResponse(content={
            "success": False,
            "error": f"Data source {source_id} not found",
            "analysis_type": request_data.analysis_type if 'request_data' in locals() else None
        }, status_code=404)
    except Exception as e:
        logger.error(f"Generate and run analysis error: {e}")
        # Check if it's a data source not found error
        if "not found" in str(e).lower() or "filenotfound" in str(e).lower():
            return JSONResponse(content={
                "success": False,
                "error": f"Data source {source_id} not found",
                "analysis_type": request_data.analysis_type if 'request_data' in locals() else None
            }, status_code=404)
        else:
            return JSONResponse(content={
                "success": False,
                "error": str(e)
            }, status_code=500)


@router.get("/api/system-stats")
async def get_system_stats(
    current_user: User = Depends(require_data_access)
):
    """Get detailed system statistics for monitoring."""
    try:
        from core.eda.security.rate_limiter import get_rate_limiter
        from core.eda.security import get_persistent_sandbox_manager
        import psutil
        
        rate_limiter = get_rate_limiter()
        global_stats = rate_limiter.get_global_stats()
        
        # Get system resource information
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval for faster response
        
        # Return in the format expected by frontend
        return JSONResponse(content={
            "memory_usage": memory.percent,
            "cpu_usage": cpu_percent,
            "active_users": global_stats.get("total_active_users", 1),
            "total_executions": global_stats.get("total_concurrent_executions", 0),
            "system_load": round((memory.percent + cpu_percent) / 2, 1),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        # Return default values if system stats fail
        return JSONResponse(content={
            "memory_usage": 25.0,
            "cpu_usage": 15.0,
            "active_users": 1,
            "total_executions": 0,
            "system_load": 20.0,
            "timestamp": datetime.now().isoformat(),
            "error": "System stats unavailable"
        })


@router.get("/api/rate-limit-status")
async def get_rate_limit_status(
    current_user: User = Depends(require_data_access)
):
    """Get current rate limit status for the user."""
    try:
        from core.eda.security.rate_limiter import get_rate_limiter
        
        rate_limiter = get_rate_limiter()
        user_stats = rate_limiter.get_user_stats(str(current_user.id))
        
        # Return the user stats directly for frontend compatibility
        return JSONResponse(content=user_stats)
        
    except Exception as e:
        logger.error(f"Rate limit status error: {e}")
        # Return default values if rate limiter fails
        return JSONResponse(content={
            "max_executions_per_minute": 15,
            "remaining_executions": 15,
            "max_concurrent_executions": 3,
            "concurrent_executions": 0,
            "window_start": datetime.now().isoformat(),
            "error": "Rate limiter unavailable"
        })


@router.post("/api/execute-custom-code/{source_id}")
async def execute_custom_code(
    source_id: str,
    request_data: CustomCodeRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Execute custom user code with access to dataset only."""
    try:
        eda_service = AdvancedEDAService(service.session)
        
        # Execute the custom code with dataset access
        result = await eda_service.execute_code(
            source_id=source_id,
            code=request_data.code,
            context="custom_notebook",
            user_id=str(current_user.id)
        )
        
        if result.get('success'):
            # Extract outputs from sandbox response
            outputs = result.get('outputs', [])
            output_text = '\n'.join([output.get('text', '') for output in outputs]) if outputs else ''
            
            return JSONResponse(content={
                "success": True,
                "code": request_data.code,
                "execution_result": result.get('execution_result', {}),
                "output": output_text,
                "outputs": outputs,
                "plots": result.get('plots', []),
                "executed_at": datetime.now().isoformat()
            })
        else:
            # Check if this is a rate limit error
            rate_limit_info = result.get('rate_limit_info')
            if rate_limit_info:
                return JSONResponse(content={
                    "success": False,
                    "error": result.get('error', 'Rate limit exceeded'),
                    "code": request_data.code,
                    "rate_limit_info": rate_limit_info
                }, status_code=429)  # Too Many Requests
            else:
                return JSONResponse(content={
                    "success": False,
                    "error": result.get('error', 'Code execution failed'),
                    "code": request_data.code,
                    "outputs": result.get('outputs', []),
                    "execution_result": result.get('execution_result', {})
                }, status_code=400)
        
    except Exception as e:
        logger.error(f"Custom code execution error: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)


@router.post("/api/reset-custom-workspace/{source_id}")
async def reset_custom_workspace(
    source_id: str,
    current_user: User = Depends(require_data_access)
):
    """Reset the persistent sandbox session for the current user and dataset."""
    try:
        manager = get_persistent_sandbox_manager()
        manager.reset_session(str(current_user.id), source_id)

        return JSONResponse(content={
            "success": True,
            "message": "Custom workspace has been reset.",
            "source_id": source_id
        })
    except Exception as e:
        logger.error("Custom workspace reset error: %s", e)
        return JSONResponse(content={
            "success": False,
            "error": f"Failed to reset custom workspace: {e}"
        }, status_code=500)


# ============================================================================
# GRANULAR ANALYSIS COMPONENT ROUTES  
# ============================================================================

@router.get("/components/available")
async def get_available_components(
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get all available granular analysis components."""
    try:
        eda_service = AdvancedEDAService(session=service.session)
        components = eda_service.get_analyses_by_category()
        
        return JSONResponse({
            "success": True,
            "components": components
        })
    except Exception as e:
        logger.error(f"Error getting available components: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/{component_id}/info")
async def get_component_info(
    component_id: str,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get detailed information about a specific component."""
    try:
        eda_service = AdvancedEDAService(session=service.session)
        info = eda_service.get_component_info(component_id)
        
        if not info:
            raise HTTPException(status_code=404, detail="Component not found")
        
        return JSONResponse({
            "success": True,
            "component": info
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/recommendations")
async def get_analysis_recommendations(
    source_id: str = Query(..., description="Source ID for the dataset"),
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Get recommended analyses based on data characteristics."""
    try:
        eda_service = AdvancedEDAService(session=service.session)
        recommendations = await eda_service.get_analysis_recommendations(source_id)
        
        return JSONResponse({
            "success": True,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GranularAnalysisRequest(BaseModel):
    analysis_ids: List[str]
    source_id: str
    selected_columns: Optional[List[str]] = None
    column_mapping: Optional[Dict[str, str]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    analysis_metadata: Optional[Dict[str, Any]] = None


@router.post("/components/run-analysis")
async def run_granular_analysis(
    request: GranularAnalysisRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Run granular analysis components - generates and executes code."""
    try:
        logger.info(f"🚀 Run analysis route called with analysis_ids={request.analysis_ids}, source_id='{request.source_id}'")
        logger.info(f"📋 Request details: selected_columns={request.selected_columns}, column_mapping={request.column_mapping}")
        
        # Validate request
        if not request.source_id:
            logger.error("❌ Missing source_id in request")
            return JSONResponse({
                "success": False,
                "error": "source_id is required"
            }, status_code=400)
            
        if not request.analysis_ids or len(request.analysis_ids) == 0:
            logger.error("❌ Missing or empty analysis_ids in request")
            return JSONResponse({
                "success": False,
                "error": "analysis_ids is required and cannot be empty"
            }, status_code=400)
        
        logger.info(f"✅ Request validation passed")
        
        eda_service = AdvancedEDAService(session=service.session)
        logger.info(f"✅ EDA service created")
        
        # Run the granular analysis (generate + execute)
        logger.info(f"🔄 Calling run_granular_analysis...")
        result = await eda_service.run_granular_analysis(
            request.source_id, 
            request.analysis_ids, 
            selected_columns=request.selected_columns,
            column_mapping=request.column_mapping,
            preprocessing_config=request.preprocessing,
            analysis_metadata=request.analysis_metadata,
        )
        
        logger.info(f"✅ Analysis completed with success={result.get('success', False)}")
        
        # Ensure result is JSON serializable
        serializable_result = make_json_serializable(result)
        return JSONResponse(serializable_result)
        
    except Exception as e:
        logger.error(f"❌ Error running granular analysis: {e}")
        logger.error(f"📋 Request was: analysis_ids={getattr(request, 'analysis_ids', 'N/A')}, source_id='{getattr(request, 'source_id', 'N/A')}'")
        
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # Return detailed error for debugging
        error_response = {
            "success": False,
            "error": f"Analysis execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "error_details": str(e)
        }
        
        # Include request details in error for debugging
        try:
            error_response["debug_info"] = {
                "analysis_ids": getattr(request, 'analysis_ids', None),
                "source_id": getattr(request, 'source_id', None),
                "has_selected_columns": bool(getattr(request, 'selected_columns', None)),
                "has_column_mapping": bool(getattr(request, 'column_mapping', None))
            }
        except:
            pass
        
        return JSONResponse(error_response, status_code=500)

@router.get("/debug/test")
async def debug_test():
    """Simple test endpoint to verify the API is working."""
    return JSONResponse({
        "success": True,
        "message": "Advanced EDA API is working",
        "timestamp": datetime.now().isoformat()
    })

@router.delete("/api/preprocessing/{source_id}")
async def reset_preprocessing_state(
    source_id: str,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Clear cached preprocessing state for a source."""

    eda_service = AdvancedEDAService(service.session)
    eda_service.clear_preprocessing_state(source_id)

    return JSONResponse(
        content={
            "success": True,
            "message": "Preprocessing state cleared",
            "source_id": source_id,
        }
    )

@router.get("/debug/data/{data_path:path}")
async def debug_data_file(data_path: str):
    """Debug endpoint to check data file status and basic info."""
    try:
        import os
        import pandas as pd
        
        logger.info(f"🔍 Debug: Checking data file: {data_path}")
        
        # Check if file exists
        file_exists = os.path.exists(data_path)
        file_size = os.path.getsize(data_path) if file_exists else None
        
        result = {
            "success": True,
            "file_path": data_path,
            "file_exists": file_exists,
            "file_size": file_size
        }
        
        if file_exists:
            try:
                # Try to load the data with different encodings
                df = None
                encoding_used = None
                
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        df = pd.read_csv(data_path, encoding=encoding, nrows=5)  # Just first 5 rows for debug
                        encoding_used = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to read with {encoding}: {e}")
                        continue
                
                if df is not None:
                    result.update({
                        "data_shape": [len(df), len(df.columns)],  # Only first 5 rows were read
                        "data_types": df.dtypes.astype(str).to_dict(),
                        "columns": df.columns.tolist(),
                        "encoding_used": encoding_used,
                        "sample_data": df.head(2).to_dict('records')  # First 2 rows as sample
                    })
                else:
                    result.update({
                        "error": "Could not read file with any encoding",
                        "encodings_tried": ['utf-8', 'latin1', 'cp1252']
                    })
                    
            except Exception as e:
                result.update({
                    "error": f"Error reading file: {str(e)}",
                    "error_type": type(e).__name__
                })
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Debug data file check failed: {e}")
        return JSONResponse({
            "success": False,
            "error": f"Debug failed: {str(e)}",
            "error_type": type(e).__name__
        }, status_code=500)


# New simplified analyze endpoint for direct execution
class AnalyzeRequest(BaseModel):
    source_id: str
    data_path: str  # Path to data file
    analysis_type: str  # Type of analysis to run


@router.post("/api/analyze")
async def analyze_data(request: AnalyzeRequest):
    """Simplified endpoint for direct analysis execution."""
    try:
        logger.info(f"🔄 Running analysis: {request.analysis_type} on {request.data_path}")
        
        # Import the runtime components
        from .granular_runtime import GranularAnalysisRuntime, AnalysisContext
        from .data_manager import EDADataManager
        
        # Create a mock database session for testing
        # In production, this should use proper session handling
        class MockSession:
            def close(self):
                pass
        
        session = MockSession()
        data_manager = EDADataManager(session)
        runtime = GranularAnalysisRuntime()
        
        # Load the data directly from file
        import pandas as pd
        import os
        
        if not os.path.exists(request.data_path):
            return JSONResponse({
                "success": False,
                "error": f"Data file not found: {request.data_path}"
            }, status_code=404)
        
        # Load data with encoding fallbacks
        df = None
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(request.data_path, encoding=encoding)
                logger.info(f"✅ Data loaded with {encoding}: {df.shape}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error with {encoding}: {e}")
                break
        
        if df is None:
            return JSONResponse({
                "success": False,
                "error": "Could not load data file with any encoding"
            }, status_code=500)
        
        # Create analysis context
        context = AnalysisContext(source_id=request.source_id)
        
        # Run the analysis
        result = runtime.run_analysis(request.analysis_type, df, context)
        
        # Convert to dict and ensure JSON serialization
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else {
            "result_type": request.analysis_type,
            "metrics": result.metrics if hasattr(result, 'metrics') else [],
            "insights": result.insights if hasattr(result, 'insights') else [],
            "tables": result.tables if hasattr(result, 'tables') else [],
            "charts": result.charts if hasattr(result, 'charts') else []
        }
        
        serializable_result = make_json_serializable(result_dict)
        
        logger.info(f"✅ Analysis completed: {request.analysis_type}")
        return JSONResponse({
            "success": True,
            **serializable_result
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        return JSONResponse({
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "error_type": type(e).__name__
        }, status_code=500)


@router.post("/debug/test-data-loading")
async def debug_test_data_loading(
    request: dict,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Debug endpoint to test data loading without running analysis."""
    try:
        source_id = request.get("source_id")
        if not source_id:
            return JSONResponse({
                "success": False,
                "error": "source_id is required"
            }, status_code=400)
        
        logger.info(f"🔍 Debug: Testing data loading for source_id: {source_id}")
        
        eda_service = AdvancedEDAService(session=service.session)
        
        # Try to load the data
        data_info = await eda_service.data_manager.prepare_for_eda(source_id, user_id="debug_user")
        df = data_info.get("data")
        
        result = {
            "success": True,
            "message": "Data loaded successfully",
            "data_info": {
                "shape": df.shape if df is not None else None,
                "columns": df.columns.tolist() if df is not None else None,
                "dtypes": df.dtypes.astype(str).to_dict() if df is not None else None,
                "strategy": data_info.get("strategy"),
                "file_path": data_info.get("file_path"),
                "memory_usage": data_info.get("memory_usage")
            }
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"❌ Debug data loading failed: {e}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        return JSONResponse({
            "success": False,
            "error": f"Data loading failed: {str(e)}",
            "error_type": type(e).__name__
        }, status_code=500)

@router.post("/components/generate-code")
async def generate_granular_analysis_code(
    request: GranularAnalysisRequest,
    current_user: User = Depends(require_data_access),
    service: DataIngestionService = Depends(get_data_service)
):
    """Generate code for selected granular analysis components."""
    try:
        logger.info(f"🚀 Route called with analysis_ids={request.analysis_ids}, source_id='{request.source_id}'")
        
        eda_service = AdvancedEDAService(session=service.session)
        
        # Validate that all components exist and are compatible
        valid_analyses = []
        incompatible_analyses = []
        
        # Get available analyses for debugging
        available_analyses = eda_service.get_available_analyses()
        logger.info(f"📊 Available analyses count: {len(available_analyses)}")
        logger.info(f"🔍 First 5 available: {available_analyses[:5]}")
        
        for analysis_id in request.analysis_ids:
            logger.info(f"🔸 Processing analysis_id: '{analysis_id}'")
            
            if analysis_id in available_analyses:
                logger.info(f"   ✅ '{analysis_id}' found in available analyses")
                
                is_compatible = await eda_service.validate_analysis_compatibility(
                    analysis_id, request.source_id
                )
                logger.info(f"   🎯 Compatibility result: {is_compatible}")
                
                if is_compatible:
                    valid_analyses.append(analysis_id)
                    logger.info(f"   ✅ '{analysis_id}' added to valid_analyses")
                else:
                    incompatible_analyses.append(analysis_id)
                    logger.info(f"   ❌ '{analysis_id}' added to incompatible_analyses")
            else:
                logger.warning(f"   ❌ '{analysis_id}' NOT FOUND in available analyses")
                incompatible_analyses.append(analysis_id)
        
        logger.info(f"📋 Final results: valid={len(valid_analyses)}, incompatible={len(incompatible_analyses)}")
        logger.info(f"   Valid analyses: {valid_analyses}")
        logger.info(f"   Incompatible analyses: {incompatible_analyses}")
        
        if not valid_analyses:
            logger.warning("❌ No valid analyses selected - returning error response")
            return JSONResponse({
                "success": False,
                "error": "No valid analyses selected",
                "incompatible_analyses": incompatible_analyses,
                "debug_info": {
                    "requested_analyses": request.analysis_ids,
                    "available_count": len(available_analyses),
                    "available_sample": available_analyses[:10]
                }
            })
        
        # Generate code for valid analyses
        result = await eda_service.generate_analysis_code(valid_analyses, request.source_id)
        
        if incompatible_analyses:
            result["warnings"] = f"Skipped incompatible analyses: {', '.join(incompatible_analyses)}"
            result["incompatible_analyses"] = incompatible_analyses
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Error generating analysis code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

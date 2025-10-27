"""Advanced EDA Service.

Service class for the advanced EDA system with enhanced functionality.
Integrates with existing data managers, export systems, and template engines.
"""
from typing import Any, Dict, List, Optional, Tuple
from time import perf_counter
from pathlib import Path
import logging
import json
import asyncio
import copy
import hashlib
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from config import get_settings

from sqlalchemy.ext.asyncio import AsyncSession
from core.data_ingestion.service import DataIngestionService
from core.data_ingestion.serialization import JSONSafeSerializer
from core.eda.text_analysis import get_text_insights
from core.utils.logging_utils import log_data_action

# Import advanced EDA components
from .data_manager import EDADataManager
from .domain_analyzer import DomainAnalyzer
from .session_manager import SessionManager
from .export_manager import ExportManager
from .notebook_templates import NotebookTemplateManager
from .granular_generators import granular_generators
from .granular_runtime import GranularAnalysisRuntime, AnalysisContext
from .granular_runtime.preprocessing import (
    PreprocessingOptions,
    PreprocessingReport,
    apply_preprocessing,
    preprocessing_state_store,
    run_preprocessing_with_state,
)
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class AdvancedEDAService:
    """Advanced service for comprehensive exploratory data analysis.
    
    Features:
    - Intelligent domain detection and template recommendation
    - Advanced session management with history tracking
    - Professional notebook export with multiple formats
    - Integrated data quality analysis and visualization
    """

    COLUMN_INSIGHTS_CACHE_TTL = 300
    GRANULAR_ANALYSIS_CACHE_TTL = 180

    _COMMON_DATETIME_FORMATS: Tuple[str, ...] = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y%m%d",
        "%d.%m.%Y",
        "%Y.%m.%d",
    )

    def __init__(self, session: AsyncSession):
        self.session = session
        try:
            self.settings = get_settings()
        except Exception:  # pragma: no cover - defensive fallback
            self.settings = None

        self.data_service = DataIngestionService(session)
        
        # Initialize advanced EDA components
        self.data_manager = EDADataManager(session)
        self.domain_analyzer = DomainAnalyzer()
        self.session_manager = SessionManager(session)
        self.export_manager = ExportManager()
        self.template_manager = NotebookTemplateManager()
        self.analysis_generators = granular_generators
        self.analysis_runtime = GranularAnalysisRuntime()

        cache_size = getattr(self.settings, "GRANULAR_CACHE_MAX_SIZE", 256) if self.settings else 256
        self.cache = CacheManager(max_size=cache_size)
        self.column_insights_cache_ttl = getattr(
            self.settings, "GRANULAR_COLUMN_INSIGHTS_CACHE_TTL", self.COLUMN_INSIGHTS_CACHE_TTL
        ) if self.settings else self.COLUMN_INSIGHTS_CACHE_TTL
        self.granular_analysis_cache_ttl = getattr(
            self.settings, "GRANULAR_ANALYSIS_CACHE_TTL", self.GRANULAR_ANALYSIS_CACHE_TTL
        ) if self.settings else self.GRANULAR_ANALYSIS_CACHE_TTL

    @classmethod
    def _coerce_datetime_series(cls, values: pd.Series) -> pd.Series:
        """Convert a series of strings to datetimes while suppressing noisy warnings."""

        if values is None:
            return pd.to_datetime(pd.Series([], dtype=object), errors="coerce")

        if not isinstance(values, pd.Series):
            values = pd.Series(values)

        if values.empty:
            return pd.to_datetime(values, errors="coerce")

        for fmt in cls._COMMON_DATETIME_FORMATS:
            try:
                return pd.to_datetime(values, format=fmt, errors="raise")
            except (ValueError, TypeError):
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(values, errors="coerce")

    @staticmethod
    def _dataset_signature(data_info: Dict[str, Any]) -> str:
        """Generate a stable signature representing the dataset backing data_info."""

        file_path = data_info.get("file_path")
        if file_path:
            try:
                stat = os.stat(file_path)
                return f"{os.path.abspath(file_path)}::{stat.st_mtime_ns}::{stat.st_size}"
            except OSError:
                return os.path.abspath(file_path)

        metadata = data_info.get("metadata") or {}
        shape = data_info.get("shape") or data_info.get("estimated_shape")
        descriptor = {"source": metadata.get("source_id"), "shape": shape}
        return json.dumps(descriptor, sort_keys=True, default=str)

    @staticmethod
    def _make_cache_key(scope: str, descriptor: Dict[str, Any]) -> str:
        """Return a hashed cache key for a given scope and descriptor."""

        serialized = json.dumps(descriptor, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{scope}:{digest}"

    async def _load_dataframe_for_insights(
        self,
        source_id: str,
        *,
        user_scope: str = "column_insights",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset for column insights with caching via the data manager."""

        data_info = await self.data_manager.prepare_for_eda(source_id, user_id=user_scope)
        df = data_info.get("data")

        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Dataset unavailable for column insights")
        if df.empty:
            raise ValueError("Dataset is empty and cannot be analyzed")

        return df, data_info

    def _compute_column_quality_details(
        self, df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compute per-column quality metrics and potential issues."""

        column_details: List[Dict[str, Any]] = []
        potential_issues: List[Dict[str, Any]] = []
        total_rows = len(df)

        for column in df.columns:
            series = df[column]
            non_null_count = int(series.notna().sum())
            null_count = int(series.isna().sum())
            null_percentage = float((null_count / total_rows * 100) if total_rows else 0.0)
            unique_count = int(series.nunique(dropna=False))
            unique_percentage = float((unique_count / total_rows * 100) if total_rows else 0.0)

            detail: Dict[str, Any] = {
                "name": column,
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "null_count": null_count,
                "null_percentage": null_percentage,
                "unique_count": unique_count,
                "unique_percentage": unique_percentage,
                "memory_usage": int(series.memory_usage(deep=True)),
                "data_category": "unknown",
            }

            if pd.api.types.is_bool_dtype(series):
                detail["data_category"] = "boolean"
                true_count = int(series.dropna().sum())
                false_count = int(non_null_count - true_count)
                detail.update(
                    {
                        "true_count": true_count,
                        "false_count": false_count,
                        "true_percentage": float(true_count / non_null_count * 100) if non_null_count else 0.0,
                    }
                )
            elif pd.api.types.is_numeric_dtype(series):
                detail["data_category"] = "numeric"
                non_null_series = series.dropna()
                if not non_null_series.empty:
                    min_value = float(non_null_series.min())
                    max_value = float(non_null_series.max())
                    detail.update(
                        {
                            "min_value": min_value,
                            "max_value": max_value,
                            "mean_value": float(non_null_series.mean()),
                            "has_negative": bool((non_null_series < 0).any()),
                            "has_zero": bool((non_null_series == 0).any()),
                        }
                    )
                    if "age" in column.lower() and max_value > 150:
                        potential_issues.append(
                            {
                                "type": "warning",
                                "column": column,
                                "message": f"Column '{column}' has values > 150. Verify age data.",
                            }
                        )
            elif pd.api.types.is_datetime64_any_dtype(series):
                detail["data_category"] = "datetime"
                potential_issues.append(
                    {
                        "type": "info",
                        "column": column,
                        "message": f"Column '{column}' detected as datetime.",
                    }
                )
            else:
                detail["data_category"] = "text"
                try:
                    text_insights = get_text_insights(series, column)
                except Exception as exc:  # pragma: no cover - defensive safety
                    logger.debug("Text insights failed for column %s: %s", column, exc)
                    text_insights = {}

                detail["text_category"] = text_insights.get("text_category", "unknown")
                detail["avg_text_length"] = text_insights.get("avg_text_length", 0)

                if text_insights.get("text_category") == "free_text":
                    potential_issues.append(
                        {
                            "type": "info",
                            "column": column,
                            "message": f"Column '{column}' contains free-text data suitable for NLP analysis.",
                        }
                    )

                recommendations = text_insights.get("eda_recommendations", [])
                for rec in recommendations:
                    potential_issues.append(
                        {
                            "type": "recommendation",
                            "column": column,
                            "message": f"Text Analysis: {rec}",
                        }
                    )

                if detail["text_category"] == "categorical" and unique_count > 50:
                    potential_issues.append(
                        {
                            "type": "warning",
                            "column": column,
                            "message": f"Column '{column}' has {unique_count} categories. Consider grouping or encoding.",
                        }
                    )

                if detail["avg_text_length"] and detail["avg_text_length"] > 200:
                    potential_issues.append(
                        {
                            "type": "info",
                            "column": column,
                            "message": f"Column '{column}' contains long text (avg: {detail['avg_text_length']:.0f} chars).",
                        }
                    )

            if detail["null_percentage"] > 30:
                potential_issues.append(
                    {
                        "type": "warning",
                        "column": column,
                        "message": f"Column '{column}' has {detail['null_percentage']:.1f}% missing values.",
                    }
                )

            if detail["unique_percentage"] > 95:
                potential_issues.append(
                    {
                        "type": "warning",
                        "column": column,
                        "message": f"Column '{column}' has {detail['unique_percentage']:.1f}% unique values. May be an identifier.",
                    }
                )

            column_details.append(detail)

        return column_details, potential_issues

    def _assemble_column_insights(
        self,
        column_details: List[Dict[str, Any]],
        potential_issues: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """Convert quality metrics into frontend-ready column insights."""

        column_insights: List[Dict[str, Any]] = []
        total_columns = len(column_details)
        problematic_columns = 0

        for col_data in column_details:
            col_name = col_data.get("name", "unknown")
            data_type = col_data.get("dtype", "unknown")
            null_percentage = col_data.get("null_percentage", 0.0) or 0.0
            unique_percentage = col_data.get("unique_percentage", 0.0) or 0.0
            data_category = col_data.get("data_category", "unknown")

            has_issues = False
            issue_messages: List[str] = []
            issue_types: List[str] = []

            if null_percentage > 30:
                has_issues = True
                issue_messages.append(f"{null_percentage:.1f}% missing")
                issue_types.append("missing_data")

            if unique_percentage > 95:
                has_issues = True
                issue_messages.append("High cardinality")
                issue_types.append("high_cardinality")

            col_issues = [issue for issue in potential_issues if issue.get("column") == col_name]
            for issue in col_issues:
                issue_type = issue.get("type")
                if issue_type in {"warning", "error"}:
                    has_issues = True
                if issue_type:
                    issue_types.append(issue_type)
                message = issue.get("message")
                if message:
                    issue_messages.append(message)

            if has_issues:
                problematic_columns += 1

            type_display = data_type
            if data_category == "numeric":
                type_display = f"{data_type} • numeric"
            elif data_category == "text":
                type_display = f"text • {col_data.get('text_category', 'text')}"
            elif data_category == "datetime":
                type_display = f"{data_type} • datetime"
            elif data_category == "boolean":
                type_display = f"{data_type} • boolean"

            status_info: List[str] = []
            non_null_count = col_data.get("non_null_count", 0) or 0
            unique_count = col_data.get("unique_count", 0) or 0

            if null_percentage > 0:
                status_info.append(f"{non_null_count:,} values")
                status_info.append(f"{null_percentage:.1f}% missing")
            else:
                status_info.append(f"{non_null_count:,} values")
                status_info.append("no missing")

            if unique_count > 0:
                if unique_count == non_null_count and null_percentage == 0:
                    status_info.append("all unique")
                elif unique_count == 1:
                    status_info.append("constant")
                elif unique_percentage > 80:
                    status_info.append(f"{unique_count:,} unique ({unique_percentage:.0f}%)")
                else:
                    status_info.append(f"{unique_count:,} unique")

            status_display = " • ".join(status_info)

            column_insight: Dict[str, Any] = {
                "name": col_name,
                "data_type": data_type,
                "type_display": type_display,
                "status_display": status_display,
                "has_issues": has_issues,
                "issue_messages": issue_messages,
                "issue_types": issue_types,
                "null_percentage": null_percentage,
                "unique_percentage": unique_percentage,
                "non_null_count": non_null_count,
                "unique_count": unique_count,
                "data_category": data_category,
                "selected": True,
                "memory_usage": col_data.get("memory_usage", 0),
            }

            if data_category == "numeric":
                column_insight.update(
                    {
                        "min_value": col_data.get("min_value"),
                        "max_value": col_data.get("max_value"),
                        "mean_value": col_data.get("mean_value"),
                        "has_negative": col_data.get("has_negative", False),
                        "has_zero": col_data.get("has_zero", False),
                    }
                )
            elif data_category == "text":
                column_insight.update(
                    {
                        "text_category": col_data.get("text_category", "unknown"),
                        "avg_text_length": col_data.get("avg_text_length", 0),
                    }
                )

            column_insights.append(column_insight)

        column_insights.sort(key=lambda x: (not x["has_issues"], x["null_percentage"]), reverse=False)

        summary_stats = {
            "total_columns": total_columns,
            "selected_columns": total_columns,
            "problematic_columns": problematic_columns,
            "numeric_columns": len([c for c in column_insights if c["data_category"] == "numeric"]),
            "text_columns": len([c for c in column_insights if c["data_category"] == "text"]),
            "datetime_columns": len([c for c in column_insights if c["data_category"] == "datetime"]),
            "boolean_columns": len([c for c in column_insights if c["data_category"] == "boolean"]),
        }

        column_recommendations: List[Dict[str, Any]] = []
        if problematic_columns > 0:
            column_recommendations.append(
                {
                    "type": "warning",
                    "title": "Review Problematic Columns",
                    "description": f"{problematic_columns} columns have potential issues that may affect analysis quality.",
                }
            )

        numeric_cols = summary_stats["numeric_columns"]
        text_cols = summary_stats["text_columns"]

        if numeric_cols > 0 and text_cols > 0:
            column_recommendations.append(
                {
                    "type": "info",
                    "title": "Mixed Data Types Detected",
                    "description": f"Dataset has {numeric_cols} numeric and {text_cols} text columns - good for comprehensive analysis.",
                }
            )

        if text_cols > numeric_cols:
            column_recommendations.append(
                {
                    "type": "info",
                    "title": "Text-Heavy Dataset",
                    "description": "Consider text preprocessing and NLP analysis techniques for optimal insights.",
                }
            )

        return column_insights, summary_stats, column_recommendations

    def _build_column_insights_payload(
        self,
        df: pd.DataFrame,
        source_id: str,
        data_info: Dict[str, Any],
        preprocessing_report: Optional[PreprocessingReport],
    ) -> Dict[str, Any]:
        """Build the full API payload for column insights."""

        random_state = getattr(self.settings, "ML_RANDOM_STATE", 42) if self.settings else 42
        sample_limit = 5000
        if self.settings:
            sample_limit = getattr(self.settings, "EDA_COLUMN_INSIGHTS_SAMPLE_LIMIT", 5000)
            helper = hasattr(self.settings, "is_field_set")
            if sample_limit is None and helper and not self.settings.is_field_set("EDA_COLUMN_INSIGHTS_SAMPLE_LIMIT"):
                sample_limit = 5000

        if sample_limit is not None and len(df) > sample_limit:
            sample_df = df.sample(sample_limit, random_state=random_state)
        else:
            sample_df = df
        column_details, potential_issues = self._compute_column_quality_details(sample_df)
        column_insights, summary_stats, recommendations = self._assemble_column_insights(column_details, potential_issues)

        payload: Dict[str, Any] = {
            "success": True,
            "column_insights": JSONSafeSerializer.clean_for_json(column_insights),
            "summary_stats": summary_stats,
            "recommendations": recommendations,
            "source_id": source_id,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        metadata = data_info.get("metadata") or {}
        dataset_name = (
            metadata.get("dataset_name")
            or metadata.get("name")
            or metadata.get("title")
        )
        if dataset_name:
            payload["dataset_name"] = dataset_name

        if preprocessing_report:
            payload["preprocessing_report"] = preprocessing_report.to_dict()

        return payload

    def clear_preprocessing_state(self, source_id: str) -> None:
        """Remove any cached preprocessing artifacts for the given source."""

        preprocessing_state_store.clear(source_id)

    async def get_source_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a data source."""
        try:
            source = await self.data_service.get_data_source_by_source_id(source_id)
            if source:
                # Try to get the actual filename from name field first
                name = getattr(source, 'name', None)
                
                # If no name, try to extract from source_metadata (JSON field)
                if not name and hasattr(source, 'source_metadata'):
                    metadata = source.source_metadata
                    if isinstance(metadata, dict):
                        name = metadata.get('filename') or metadata.get('original_filename') or metadata.get('name')
                
                # If still no name, try to extract from config
                if not name and hasattr(source, 'config'):
                    config = source.config
                    if isinstance(config, dict):
                        name = config.get('filename') or config.get('original_filename') or config.get('name')
                
                # If still no name, try to get it from the data file path
                if not name:
                    uploads_dir = Path(__file__).parent.parent.parent.parent / "uploads" / "data"
                    if uploads_dir.exists():
                        for file_path in uploads_dir.iterdir():
                            if file_path.is_file() and file_path.name.startswith(f"{source_id}_"):
                                # Extract original filename after the source_id prefix
                                filename = file_path.name
                                if "_" in filename:
                                    name = filename.split("_", 1)[1]  # Get part after first underscore
                                else:
                                    name = filename
                                break
                
                # Fallback to source name or descriptive name
                if not name:
                    name = getattr(source, 'name', None) or f'Dataset {source_id[:8]}'
                
                return {
                    'id': source_id,
                    'name': name,
                    'description': getattr(source, 'description', ''),
                    'created_at': getattr(source, 'created_at').isoformat() if getattr(source, 'created_at') else None,
                    'status': getattr(source, 'status', 'unknown')
                }
            return {
                'id': source_id,
                'name': f'Dataset {source_id[:8]}',  # Fallback if source not found
                'description': '',
                'created_at': None,
                'status': 'unknown'
            }
        except Exception as e:
            logger.error(f"Error getting source info for {source_id}: {e}")
            return {
                'id': source_id,
                'name': f'Dataset {source_id[:8]}',  # Fallback on error
                'description': '',
                'created_at': None,
                'status': 'error'
            }

    async def get_preview(self, source_id: str, sample_size: int = 100, mode: str = "head") -> Dict[str, Any]:
        """Get data preview using the advanced data manager."""
        try:
            log_data_action("ADVANCED_PREVIEW", details=f"source:{source_id},size:{sample_size},mode:{mode}")
            
            result = await self.data_manager.load_data_preview(
                source_id=source_id,
                user_id="system",  # Will be updated with proper user context
                sample_size=sample_size,
                mode=mode
            )
            
            return {
                "success": True,
                "preview": result.get("data", {}),
                "metadata": result.get("metadata", {}),
                "source_id": source_id
            }
            
        except Exception as e:
            logger.error(f"Advanced preview error: {e}")
            return {
                "success": False,
                "error": f"Preview failed: {str(e)}"
            }

    async def get_quality_report(self, source_id: str, sample_size: int = 500) -> Dict[str, Any]:
        """Get comprehensive quality report using the basic EDA service."""
        try:
            log_data_action("ADVANCED_QUALITY", details=f"source:{source_id},size:{sample_size}")
            
            # Use the basic EDA service quality_report method which already works
            from core.eda.services import EDAService
            basic_eda_service = EDAService(self.session)
            
            # Get quality report from the basic service
            quality_result = await basic_eda_service.quality_report(source_id, sample_size)
            
            if quality_result.get("success"):
                return quality_result
            else:
                return {
                    "success": False,
                    "error": quality_result.get("error", "Quality report failed"),
                    "message": quality_result.get("message", "Unable to generate quality report")
                }
            
        except Exception as e:
            logger.error(f"Advanced quality report error: {e}")
            return {
                "success": False,
                "error": f"Quality report failed: {str(e)}"
            }

    async def detect_domain(self, source_id: str) -> Dict[str, Any]:
        """Detect dataset domain using the advanced domain analyzer."""
        try:
            # Get data preview for domain analysis
            preview_result = await self.get_preview(source_id, sample_size=200)
            if not preview_result["success"]:
                return preview_result
                
            preview_data = preview_result["preview"]
            columns = preview_data.get("columns", [])
            sample_data = preview_data.get("data", [])
            
            # Use domain analyzer
            domain_result = await asyncio.to_thread(
                self.domain_analyzer.analyze_dataset_domain,
                columns,
                sample_data,
                preview_result.get("metadata", {}),
            )
            
            return {
                "success": True,
                "primary_domain": domain_result["primary_domain"],
                "domain": domain_result["primary_domain"],  # Keep both for compatibility
                "primary_confidence": domain_result["primary_confidence"],
                "confidence": domain_result["confidence"],  # Keep for backward compatibility
                "secondary_domains": domain_result.get("secondary_domains", []),
                "detected_patterns": domain_result["patterns"],
                "recommendations": domain_result["recommendations"],
                "domain_scores": domain_result.get("domain_scores", {}),
                "source_id": source_id
            }
            
        except Exception as e:
            logger.error(f"Domain detection error: {e}")
            return {
                "success": False,
                "error": f"Domain detection failed: {str(e)}"
            }

    async def generate_domain_analysis(self, source_id: str, template_name: str, analysis_depth: str = "intermediate") -> Dict[str, Any]:
        """Generate domain-specific analysis using templates."""
        try:
            # Get data for analysis - use full dataset for Advanced EDA (up to 100,000 rows)
            preview_result = await self.get_preview(source_id, sample_size=100000, mode="sample")
            if not preview_result["success"]:
                return preview_result
                
            # Generate analysis using template manager with specified depth
            analysis_result = self.template_manager.generate_template_analysis(
                template_name=template_name,
                data_preview=preview_result["preview"],
                metadata=preview_result.get("metadata", {}),
                source_id=source_id,
                analysis_depth=analysis_depth
            )
            
            # Return results in format expected by frontend 
            result = {
                "success": True,
                "template_name": template_name,
                "analysis_depth": analysis_depth,
                "source_id": source_id
            }
            
            # Add template analysis data if available
            if isinstance(analysis_result, dict):
                result.update({
                    "template_summary": analysis_result.get("summary", {}),
                    "template_visualizations": analysis_result.get("visualizations", []),
                    "template_sections": analysis_result.get("sections", [])
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Domain analysis error: {e}")
            return {
                "success": False,
                "error": f"Domain analysis failed: {str(e)}"
            }

    async def generate_and_execute_analysis_code(self, source_id: str, template_name: str) -> Dict[str, Any]:
        """Generate and execute Python code for specific analysis types - separate from domain analysis."""
        try:
            logger.info(f"Starting analysis generation for source_id: {source_id}, template: {template_name}")
            
            # Get full dataset context for code generation
            try:
                logger.info(f"Getting dataset context for {source_id}")
                data_preview = await self._get_full_dataset_context(source_id)
                logger.info(f"Dataset context retrieved: {data_preview is not None}")
            except Exception as e:
                logger.warning(f"Failed to get dataset context for {source_id}: {e}")
                data_preview = None

            # Generate Python code for the specific analysis type
            try:
                logger.info(f"Generating analysis code for template: {template_name}")
                code = self._generate_analysis_code(template_name, source_id, data_preview)
                logger.info(f"Analysis code generated: {len(code) if code else 0} characters")
            except Exception as e:
                logger.error(f"Code generation failed for {template_name}: {e}")
                return {
                    "success": False,
                    "error": f"Code generation failed: {str(e)}",
                    "template_name": template_name
                }
            
            # Execute the generated code
            try:
                logger.info(f"Executing generated code for {source_id}")
                execution_result = await self.execute_code(source_id, code, "notebook")
                logger.info(f"Code execution completed: {execution_result.get('success', False)}")
            except Exception as e:
                logger.error(f"Code execution failed: {e}")
                return {
                    "success": False,
                    "error": f"Code execution failed: {str(e)}",
                    "code": code,
                    "template_name": template_name
                }
            
            # Return results with code and execution details
            if execution_result.get("success"):
                result = {
                    "success": True,
                    "code": code,
                    "execution_result": execution_result.get("result", {}),
                    "output": execution_result.get("output", {}),
                    "template_name": template_name,
                    "source_id": source_id
                }
                
                return result
            else:
                return {
                    "success": False,
                    "error": execution_result.get("error", "Code execution failed"),
                    "code": code,
                    "template_name": template_name
                }
            
        except Exception as e:
            logger.error(f"Code generation and execution error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Code generation and execution failed: {str(e)}"
            }

    def _generate_analysis_code(self, template_name: str, source_id: str, data_preview: Dict = None) -> str:
        """Generate Python code for specific analysis types."""
        return self.analysis_generators.generate_code(template_name, data_preview)

    async def _get_full_dataset_context(self, source_id: str) -> Dict[str, Any]:
        """Get complete dataset information for analysis code generation - uses full data, no row limiting."""
        try:
            # Get data source information
            source = await self.data_service.get_data_source_by_source_id(source_id)
            if not source:
                logger.warning(f"Data source not found for source_id: {source_id}")
                return None
                
            # Get the file path
            config = source.config or {}
            file_path = config.get('file_path')
            
            if not file_path:
                logger.warning(f"No file path found for source_id: {source_id}")
                return None
            
            # Load full dataset information for analysis (no row limiting)
            try:
                # First get total row count without loading all data
                from pathlib import Path

                file_path_obj = Path(file_path)
                file_ext = file_path_obj.suffix.lower()
                sample_row_count = 200

                if file_ext == '.csv':
                    # Get column info and total rows for CSV
                    df_sample = pd.read_csv(file_path, nrows=sample_row_count)
                    columns = list(df_sample.columns)
                    dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}

                    # Count total rows efficiently
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for _ in f) - 1  # Subtract header row

                elif file_ext in ['.xlsx', '.xls']:
                    df_sample = pd.read_excel(file_path, nrows=sample_row_count)
                    columns = list(df_sample.columns)
                    dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}

                    # For Excel, we need to load to count (less efficient but necessary)
                    df_full = pd.read_excel(file_path)
                    total_rows = len(df_full)

                elif file_ext == '.json':
                    df_sample = pd.read_json(file_path, lines=True, nrows=sample_row_count)
                    columns = list(df_sample.columns)
                    dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}

                    # For JSON, count lines
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for _ in f)
                else:
                    logger.warning(f"Unsupported file type: {file_ext}")
                    return None

                # Get sample data for context (limit to first 20 rows for payload size)
                sample_preview = df_sample.head(20)
                sample_data = sample_preview.fillna("").to_dict('records')

                # Identify column types for compatibility checks
                numeric_columns = list(df_sample.select_dtypes(include=[np.number]).columns)
                string_like_dtypes = ['object', 'string']
                categorical_columns = list(df_sample.select_dtypes(include=string_like_dtypes).columns)
                datetime_columns = list(
                    df_sample.select_dtypes(include=['datetime', 'datetimetz', 'datetime64[ns]']).columns
                )
                boolean_columns = list(df_sample.select_dtypes(include=['bool']).columns)

                # Enhanced datetime detection - check string-like columns for potential dates
                potential_datetime_columns: List[str] = []
                if not df_sample.empty:
                    for col in categorical_columns:
                        if col in datetime_columns:
                            continue

                        series = df_sample[col].dropna()
                        if series.empty:
                            continue

                        str_values = series.astype(str)
                        converted = self._coerce_datetime_series(str_values)
                        valid_ratio = converted.notna().mean() if len(converted) else 0

                        if valid_ratio >= 0.6 and converted.notna().sum() >= 3:
                            potential_datetime_columns.append(col)
                            logger.info(
                                "🕐 Detected potential datetime column: '%s' (valid_ratio=%.2f)",
                                col,
                                valid_ratio,
                            )

                # Add detected datetime columns to the main list
                all_datetime_columns = sorted(set(datetime_columns + potential_datetime_columns))

                # Update categorical columns to exclude detected datetime columns
                categorical_columns = [col for col in categorical_columns if col not in all_datetime_columns]

                return {
                    "columns": columns,
                    "sample_data": sample_data,  # Limited preview for structure context
                    "dtypes": dtypes,
                    "total_rows": total_rows,
                    "shape": [total_rows, len(columns)],
                    "file_path": file_path,  # Include file path for analysis code
                    "file_type": file_ext,
                    # Add the expected column type lists for compatibility checks
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "object_columns": categorical_columns,  # Alias for categorical (excluding detected datetime)
                    "datetime_columns": all_datetime_columns,  # Include both native and detected datetime columns
                    "boolean_columns": boolean_columns,
                    "potential_datetime_columns": potential_datetime_columns,  # For debugging
                }
                
            except Exception as e:
                logger.error(f"Error loading data information for {source_id}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting data information for {source_id}: {e}")
            return None

    async def execute_code(self, source_id: str, code: str, context: str = None, user_id: str = None) -> Dict[str, Any]:
        """Execute Python code in secure sandbox with dataset access only."""
        try:
            logger.info(f"Starting secure code execution for source_id: {source_id}, user: {user_id}")
            
            # Check rate limits for custom code execution
            if context == "custom_notebook" and user_id:
                from core.eda.security.rate_limiter import get_rate_limiter
                rate_limiter = get_rate_limiter()
                
                rate_check = rate_limiter.check_rate_limit(user_id)
                if not rate_check["allowed"]:
                    return {
                        "success": False,
                        "error": rate_check["message"],
                        "rate_limit_info": rate_check
                    }
                
                # Record execution start
                rate_limiter.record_execution_start(user_id)
                
                try:
                    result = await self._execute_with_monitoring(source_id, code, context, user_id)
                finally:
                    # Always record execution end
                    rate_limiter.record_execution_end(user_id)
                    
                return result
            else:
                # For non-custom code (generated analysis), no rate limiting
                return await self._execute_with_monitoring(source_id, code, context, user_id)
                
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }
    
    
    async def _execute_with_monitoring(self, source_id: str, code: str, context: str = None, user_id: str = None) -> Dict[str, Any]:
        """Execute Python code in secure sandbox with dataset access only."""
        try:
            logger.info(f"Starting secure code execution for source_id: {source_id}")
            
            # Get data file path for code execution
            uploads_dir = Path(__file__).parent.parent.parent.parent / "uploads" / "data"
            logger.info(f"Looking for data files in: {uploads_dir}")
            data_file_path = None
            
            if uploads_dir.exists():
                logger.info(f"Uploads directory exists, scanning for files with prefix: {source_id}_")
                for file_path in uploads_dir.iterdir():
                    logger.info(f"Checking file: {file_path.name}")
                    if file_path.is_file() and file_path.name.startswith(f"{source_id}_"):
                        data_file_path = file_path
                        logger.info(f"Found data file: {data_file_path}")
                        break
            else:
                logger.error(f"Uploads directory does not exist: {uploads_dir}")
            
            if not data_file_path:
                logger.error(f"No data file found for source_id: {source_id}")
                return {
                    "success": False,
                    "error": f"Data source not found for execution. Looked in {uploads_dir}"
                }

            # Use secure sandbox for custom code execution
            if context == "custom_notebook":
                logger.info("Using persistent sandbox session for custom notebook execution")
                from core.eda.security import (
                    SandboxExecutionError,
                    get_persistent_sandbox_manager,
                )

                manager = get_persistent_sandbox_manager()
                try:
                    session = manager.get_or_create_session(
                        user_id or "default",
                        source_id,
                        str(data_file_path),
                        max_execution_time=30,
                        max_memory_mb=512,
                        max_cpu_percent=50,
                    )
                    return session.execute_code(code)
                except SandboxExecutionError as exc:
                    logger.error("Persistent sandbox session error: %s", exc)
                    manager.reset_session(user_id or "default", source_id)
                    return {
                        "success": False,
                        "error": "Sandbox session unavailable. Please rerun the cell.",
                        "outputs": [
                            {
                                "type": "error",
                                "text": "Sandbox session error encountered. Session has been reset.",
                            }
                        ],
                    }

            # For generated analysis code, use the existing secure execution method
            # Load dataset into context
            data_context = {
                "df": f"pd.read_csv('{data_file_path}')",
                "data_path": str(data_file_path)
            }

            # Prepare code with data loading and automatic output handling
            data_file_path_str = str(data_file_path).replace('\\', '/')
            
            # For generated analysis code, DON'T indent - it's already properly formatted
            processed_code = code
            
            # For generated analysis code, use standard execution with basic security
            # Combine dataset loading and analysis code
            full_code = """# -*- coding: utf-8 -*-
# Load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure proper UTF-8 output on Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Load and prepare dataset
df = pd.read_csv('{}')
print("Dataset loaded successfully with shape:", df.shape)

# Execute analysis code
{}
""".format(data_file_path_str, processed_code)
            
            # Use simple execution instead of complex LLM engine for now
            # This avoids the kernel initialization issues
            try:
                import subprocess
                import tempfile
                import json
                import sys
                import os
                
                logger.info("Starting subprocess execution setup")
                
                # Get the current Python executable (should be in the virtual environment)
                python_exe = sys.executable
                logger.info(f"Using Python executable: {python_exe}")
                
                # Write code to temporary file with UTF-8 encoding
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(full_code)
                    temp_file = f.name
                    logger.info(f"Created temporary file: {temp_file}")
                
                logger.info("Executing Python code via subprocess")
                # Execute the Python code using the same interpreter
                # Set up environment with UTF-8 encoding for Windows
                subprocess_env = os.environ.copy()
                subprocess_env['PYTHONIOENCODING'] = 'utf-8'
                subprocess_env['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
                subprocess_env['PYTHONUTF8'] = '1'
                
                result = subprocess.run(
                    [python_exe, temp_file], 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8',
                    errors='replace',  # Replace problematic characters instead of failing
                    timeout=60,
                    cwd=os.getcwd(),  # Use current working directory
                    env=subprocess_env
                )
                
                logger.info(f"Subprocess completed with return code: {result.returncode}")
                if result.stdout:
                    logger.info(f"Subprocess stdout: {result.stdout[:500]}...")  # Log first 500 chars
                if result.stderr:
                    logger.error(f"Subprocess stderr: {result.stderr}")
                
                # Clean up
                os.unlink(temp_file)
                logger.info("Temporary file cleaned up")
                
                execution_result = {
                    "success": result.returncode == 0,
                    "outputs": [
                        {
                            "type": "stream",
                            "name": "stdout",
                            "text": result.stdout
                        }
                    ] if result.stdout else [],
                    "error": result.stderr if result.stderr else None,
                    "execution_count": 1
                }
                
                if result.stderr:
                    execution_result["outputs"].append({
                        "type": "stream", 
                        "name": "stderr",
                        "text": result.stderr
                    })
                    
            except Exception as exec_error:
                logger.error(f"Subprocess execution exception: {exec_error}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                execution_result = {
                    "success": False,
                    "error": f"Code execution failed: {str(exec_error)}",
                    "outputs": [],
                    "execution_count": 0
                }

            # Format output for notebook display
            if execution_result["success"]:
                # Process outputs for notebook display
                formatted_output = {
                    "stdout": "",
                    "stderr": "",
                    "plots": [],
                    "data_frames": []
                }

                # Extract stdout/stderr from outputs
                for output in execution_result.get("outputs", []):
                    if output["type"] == "stream":
                        if output["name"] == "stdout":
                            formatted_output["stdout"] += output["text"]
                        elif output["name"] == "stderr":
                            formatted_output["stderr"] += output["text"]
                    elif output["type"] in ["display_data", "execute_result"]:
                        data = output.get("data", {})
                        # Handle matplotlib plots
                        if "image/png" in data:
                            formatted_output["plots"].append(data["image/png"])
                        # Handle dataframes
                        if "text/html" in data and "<table" in data["text/html"]:
                            formatted_output["data_frames"].append({
                                "html": data["text/html"]
                            })

                # Add matplotlib figures
                for figure in execution_result.get("figures", []):
                    formatted_output["plots"].append(figure)

                return {
                    "success": True,
                    "output": formatted_output,
                    "execution_count": execution_result.get("execution_count", 0),
                    "timestamp": execution_result.get("timestamp")
                }
            else:
                return {
                    "success": False,
                    "error": execution_result.get("error", "Unknown execution error"),
                    "execution_count": execution_result.get("execution_count", 0)
                }
            
        except Exception as e:
            logger.error(f"Notebook code execution error: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }

    async def save_session(self, session_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Save EDA session using the session manager."""
        try:
            session_result = await self.session_manager.save_session(
                user_id=user_id,
                session_data=session_data,
                session_type="advanced_eda"
            )
            
            return {
                "success": True,
                "session_id": session_result["session_id"],
                "saved_at": session_result["saved_at"]
            }
            
        except Exception as e:
            logger.error(f"Session save error: {e}")
            return {
                "success": False,
                "error": f"Session save failed: {str(e)}"
            }

    async def get_sessions(self, user_id: str, source_id: str = None) -> Dict[str, Any]:
        """Get EDA sessions using the session manager."""
        try:
            sessions_result = await self.session_manager.get_user_sessions(
                user_id=user_id,
                session_type="advanced_eda",
                source_id=source_id
            )
            
            return {
                "success": True,
                "sessions": sessions_result["sessions"],
                "total": len(sessions_result["sessions"])
            }
            
        except Exception as e:
            logger.error(f"Get sessions error: {e}")
            return {
                "success": False,
                "error": f"Get sessions failed: {str(e)}"
            }

    async def export_notebook(self, export_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Export session as Jupyter notebook using the export manager."""
        try:
            export_result = await self.export_manager.export_to_notebook(
                session_data=export_data,
                user_id=user_id,
                export_format="jupyter",
                include_outputs=True
            )
            
            return {
                "success": True,
                "file_path": export_result["file_path"],
                "filename": export_result["filename"],
                "export_type": "jupyter_notebook"
            }
            
        except Exception as e:
            logger.error(f"Notebook export error: {e}")
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }

    # ===== GRANULAR ANALYSIS METHODS =====
    
    def get_available_analyses(self) -> List[str]:
        """Get list of all available granular analysis components."""
        return self.analysis_generators.get_available_analyses()
    
    def get_analyses_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get analyses grouped by category."""
        return self.analysis_generators.get_grouped_analyses()
    
    def get_analysis_metadata(self, analysis_id: str) -> Dict[str, Any]:
        """Get metadata for a specific analysis."""
        return self.analysis_generators.get_analysis_metadata(analysis_id)
    
    async def get_analysis_recommendations(self, source_id: str) -> List[str]:
        """Get recommended analyses based on data characteristics."""
        try:
            # Get data preview for recommendations
            preview_data = await self._get_full_dataset_context(source_id)
            if not preview_data:
                return []
            
            return self.analysis_generators.get_analysis_recommendations(preview_data)
            
        except Exception as e:
            logger.error(f"Error getting analysis recommendations: {e}")
            return self.analysis_generators.get_available_analyses()[:10]
    
    async def validate_analysis_compatibility(self, analysis_id: str, source_id: str) -> bool:
        """Check if analysis is compatible with the data."""
        try:
            logger.info(f"🔍 Validating compatibility for analysis_id='{analysis_id}', source_id='{source_id}'")
            
            # Get data preview for validation
            preview_data = await self._get_full_dataset_context(source_id)
            
            if not preview_data:
                logger.warning(f"⚠️ No preview data returned for source_id='{source_id}', defaulting to compatible=True")
                return True  # Default to compatible if can't check
            
            logger.info(f"✅ Got preview data with keys: {list(preview_data.keys()) if preview_data else 'None'}")
            
            # Debug datetime columns specifically for time series
            if analysis_id in ['seasonality_detection', 'temporal_trends']:
                datetime_cols = preview_data.get('datetime_columns', [])
                object_cols = preview_data.get('object_columns', [])
                logger.info(f"🕐 DEBUG for {analysis_id}: datetime_columns={datetime_cols}, object_columns={object_cols}")
            
            result = self.analysis_generators.validate_analysis_compatibility(analysis_id, preview_data)
            logger.info(f"🎯 Compatibility result for '{analysis_id}': {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error validating analysis compatibility for '{analysis_id}': {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return True  # Default to compatible if validation fails
    
    async def run_granular_analysis(
        self,
        source_id: str,
        analysis_ids: List[str],
        selected_columns: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        analysis_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run granular analyses using the structured runtime engine."""

        try:
            logger.info(
                "🚀 Running granular analysis via runtime for source_id=%s analyses=%s selected_columns=%s",
                source_id,
                analysis_ids,
                selected_columns,
            )

            available_analyses = self.get_available_analyses()
            valid_analyses: List[str] = []
            incompatible_analyses: List[str] = []

            for analysis_id in analysis_ids:
                if analysis_id not in available_analyses:
                    incompatible_analyses.append(analysis_id)
                    continue

                is_compatible = await self.validate_analysis_compatibility(analysis_id, source_id)
                if is_compatible:
                    valid_analyses.append(analysis_id)
                else:
                    incompatible_analyses.append(analysis_id)

            if not valid_analyses:
                return {
                    "success": False,
                    "error": "No valid analyses selected",
                    "incompatible_analyses": incompatible_analyses,
                }

            # Load dataset (sampled if necessary) for analysis
            try:
                logger.info(f"🔄 Loading data for source_id: {source_id}")
                data_info = await self.data_manager.prepare_for_eda(source_id, user_id="analysis_runtime")
                df = data_info.get("data")
                logger.info(f"📊 Loaded data shape: {df.shape if df is not None else 'None'}")
            except Exception as e:
                logger.error(f"❌ Data loading failed: {e}")
                return {
                    "success": False,
                    "error": f"Failed to load dataset: {str(e)}",
                }
                
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                logger.error("❌ Dataset is None, not DataFrame, or empty")
                return {
                    "success": False,
                    "error": "Dataset unavailable or empty for analysis",
                }

            preprocessing_report = None
            preprocessing_options: Optional[PreprocessingOptions] = None

            if preprocessing_config:
                try:
                    preprocessing_options = PreprocessingOptions.from_payload(preprocessing_config)
                    logger.info(
                        "🧹 Preprocessing configuration received: %s",
                        preprocessing_options.to_payload(),
                    )
                except Exception as exc:  # pragma: no cover - defensive parsing
                    logger.warning("⚠️ Invalid preprocessing configuration ignored: %s", exc)

            if preprocessing_options:
                df, preprocessing_report = run_preprocessing_with_state(
                    source_id,
                    df,
                    preprocessing_options,
                    base="auto",
                )
                logger.info(
                    "✅ Applied preprocessing operations: %s",
                    preprocessing_report.applied_operations,
                )

                if preprocessing_report.dropped_columns:
                    logger.info(
                        "🗂️ Columns dropped during preprocessing: %s",
                        ", ".join(preprocessing_report.dropped_columns),
                    )

            # Limit to avoid excessive processing (configurable)
            max_rows: Optional[int]
            if self.settings:
                max_rows = getattr(self.settings, "GRANULAR_RUNTIME_MAX_ROWS", None)
                helper = hasattr(self.settings, "is_field_set")
                if max_rows is None:
                    if helper and self.settings.is_field_set("GRANULAR_RUNTIME_MAX_ROWS"):
                        pass  # Explicit unlimited
                    elif helper and self.settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT"):
                        max_rows = self.settings.EDA_GLOBAL_SAMPLE_LIMIT
                    else:
                        max_rows = 50_000
            else:
                max_rows = 50_000
            if max_rows is not None and len(df) > max_rows:
                random_state = getattr(self.settings, "GRANULAR_RUNTIME_RANDOM_STATE", 42) if self.settings else 42
                df = df.sample(max_rows, random_state=random_state)

            if selected_columns:
                filtered_selected = [col for col in selected_columns if col in df.columns]
                if len(filtered_selected) != len(selected_columns):
                    missing = set(selected_columns) - set(filtered_selected)
                    logger.info("ℹ️ Selected columns removed by preprocessing: %s", ", ".join(missing))
                selected_columns = filtered_selected

            metadata = dict(data_info.get("metadata", {})) if isinstance(data_info.get("metadata"), dict) else {}
            if preprocessing_report:
                metadata["preprocessing_report"] = preprocessing_report.to_dict()
            if analysis_metadata:
                metadata.update(analysis_metadata)

            context = AnalysisContext(
                source_id=source_id,
                selected_columns=selected_columns,
                column_mapping=column_mapping,
                metadata=metadata,
            )

            dataset_signature = self._dataset_signature(data_info)
            preprocess_marker: Optional[Dict[str, Any]]
            if preprocessing_report:
                preprocess_marker = {
                    "applied_operations": preprocessing_report.applied_operations,
                    "dropped_columns": preprocessing_report.dropped_columns,
                }
            else:
                preprocess_marker = preprocessing_config

            metadata_descriptor = None
            if analysis_metadata:
                try:
                    metadata_descriptor = json.dumps(analysis_metadata, sort_keys=True, default=str)
                except TypeError:
                    metadata_descriptor = str(sorted(analysis_metadata.items(), key=lambda kv: kv[0]))

            cache_descriptor = {
                "dataset": dataset_signature,
                "analyses": sorted(valid_analyses),
                "selected_columns": sorted(selected_columns or []),
                "column_mapping": sorted(column_mapping.items()) if column_mapping else None,
                "preprocessing": preprocess_marker,
                "incompatible": sorted(incompatible_analyses),
                "analysis_metadata": metadata_descriptor,
            }
            cache_key = self._make_cache_key("granular_analysis", cache_descriptor)

            cached_payload = await self.cache.get(cache_key)
            if cached_payload:
                response_copy = copy.deepcopy(cached_payload)
                response_copy["cache_hit"] = True
                response_copy["incompatible_analyses"] = incompatible_analyses
                return response_copy

            analysis_results: List[Dict[str, Any]] = []
            
            # Run analyses in thread pool to avoid blocking other users
            loop = asyncio.get_event_loop()
            
            for analysis_id in valid_analyses:
                try:
                    logger.info(f"🔄 Running analysis: {analysis_id}")
                    # Run each analysis in a thread pool to avoid blocking
                    runtime_result = await loop.run_in_executor(
                        None, 
                        self.analysis_runtime.run_analysis, 
                        analysis_id, 
                        df, 
                        context
                    )
                    logger.info(f"✅ Completed analysis: {analysis_id} (status: {runtime_result.status})")
                    analysis_results.append(runtime_result.to_dict())
                except Exception as e:
                    logger.error(f"❌ Analysis {analysis_id} failed: {e}")
                    # Create error result
                    error_result = {
                        "analysis_id": analysis_id,
                        "title": analysis_id.replace("_", " ").title(),
                        "status": "error",
                        "summary": f"Analysis failed: {str(e)}",
                        "metrics": [],
                        "insights": [{"level": "danger", "text": f"Analysis failed: {str(e)}"}],
                        "tables": [],
                        "charts": [],
                        "details": {"error": str(e)}
                    }
                    analysis_results.append(error_result)

            output_payload = {
                "analysis_results": analysis_results,
                "structured": True,
            }

            response: Dict[str, Any] = {
                "success": True,
                "analysis_ids": valid_analyses,
                "analysis_count": len(valid_analyses),
                "analysis_results": analysis_results,
                "selected_columns": selected_columns,
                "column_mapping": column_mapping,
                "metadata": context.metadata,
                "output": output_payload,
            }

            if preprocessing_report:
                response["preprocessing_report"] = preprocessing_report.to_dict()

            if incompatible_analyses:
                response["warnings"] = f"Skipped incompatible analyses: {', '.join(incompatible_analyses)}"
                response["incompatible_analyses"] = incompatible_analyses

            response["cache_hit"] = False
            await self.cache.set(cache_key, copy.deepcopy(response), ttl=self.granular_analysis_cache_ttl)

            return response

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"❌ Error in run_granular_analysis: {e}")
            import traceback

            logger.error("Full traceback: %s", traceback.format_exc())
            return {
                "success": False,
                "error": f"Granular analysis failed: {str(e)}",
            }

    async def generate_analysis_code(self, analysis_ids: List[str], source_id: str, selected_columns: Optional[List[str]] = None, column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate code for selected granular analyses.
        
        Args:
            analysis_ids: List of analysis component IDs
            source_id: String identifier for the data source
            selected_columns: Optional list of column names to include in analysis
            column_mapping: Optional mapping of metric names to column names for domain analyses
        """
        try:
            logger.info(f"🚀 Generating code for analysis_ids={analysis_ids}, source_id='{source_id}', selected_columns={selected_columns}")
            
            # Get data preview for code generation
            preview_data = await self._get_full_dataset_context(source_id)
            if not preview_data:
                logger.warning(f"⚠️ No preview data for source_id='{source_id}', using empty dict")
                preview_data = {}  # Use empty dict if can't get data
            else:
                logger.info(f"✅ Got preview data with keys: {list(preview_data.keys())}")
            
            # Filter columns if selected_columns is provided
            if selected_columns:
                logger.info(f"🔍 Filtering data context for selected columns: {selected_columns}")
                
                # Store original columns for reference
                all_columns = preview_data.get('columns', [])
                preview_data['all_columns'] = all_columns  # Preserve original columns list
                
                # Filter the different column type lists
                filtered_columns = [col for col in all_columns if col in selected_columns]
                
                # Update column lists in preview_data
                original_numeric = preview_data.get('numeric_columns', [])
                original_categorical = preview_data.get('categorical_columns', [])
                original_object = preview_data.get('object_columns', [])
                original_datetime = preview_data.get('datetime_columns', [])
                original_boolean = preview_data.get('boolean_columns', [])
                
                preview_data['columns'] = filtered_columns
                preview_data['numeric_columns'] = [col for col in original_numeric if col in selected_columns]
                preview_data['categorical_columns'] = [col for col in original_categorical if col in selected_columns]
                preview_data['object_columns'] = [col for col in original_object if col in selected_columns]
                preview_data['datetime_columns'] = [col for col in original_datetime if col in selected_columns]
                preview_data['boolean_columns'] = [col for col in original_boolean if col in selected_columns]
                
                # Filter sample_data if it exists
                if 'sample_data' in preview_data and isinstance(preview_data['sample_data'], list):
                    filtered_sample_data = []
                    for row in preview_data['sample_data']:
                        if isinstance(row, dict):
                            filtered_row = {col: row[col] for col in selected_columns if col in row}
                            filtered_sample_data.append(filtered_row)
                    preview_data['sample_data'] = filtered_sample_data
                
                logger.info(f"✅ Filtered preview data - original columns: {len(all_columns)}, selected columns: {len(filtered_columns)}, numeric: {len(preview_data['numeric_columns'])}, categorical: {len(preview_data['categorical_columns'])}")
            else:
                # No column filtering, but still preserve all columns for reference
                all_columns = preview_data.get('columns', [])
                preview_data['all_columns'] = all_columns
            
            # Add column mapping to preview_data for domain analyses
            if column_mapping:
                logger.info(f"🔗 Adding column mapping to preview data: {column_mapping}")
                preview_data['column_mapping'] = column_mapping
            
            # Generate code for multiple analyses
            code = self.analysis_generators.generate_multiple_analyses_code(analysis_ids, preview_data)
            logger.info(f"✅ Generated {len(code)} characters of code for {len(analysis_ids)} analyses")
            
            return {
                "success": True,
                "code": code,
                "analysis_count": len(analysis_ids),
                "analysis_ids": analysis_ids,
                "selected_columns": selected_columns
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating analysis code: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Code generation failed: {str(e)}",
                "code": f"# Error generating code: {str(e)}"
            }
    
    def get_component_info(self, analysis_id: str) -> Dict[str, Any]:
        """Get complete information about a component."""
        return self.analysis_generators.get_component_info(analysis_id)

    async def get_column_insights(
        self,
        source_id: str,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        base: str = "auto",
    ) -> Dict[str, Any]:
        """Get column insights using the most recent preprocessing state when available."""

        try:
            log_data_action("GET_COLUMN_INSIGHTS", details=f"source_id:{source_id}")
            timer_start = perf_counter()

            load_start = perf_counter()
            df, data_info = await self._load_dataframe_for_insights(source_id)
            load_elapsed = perf_counter() - load_start

            dataset_signature = self._dataset_signature(data_info)

            cached_state = None
            state_version: Optional[str] = None
            if not preprocessing_config:
                cached_state = preprocessing_state_store.snapshot(source_id)
                if cached_state:
                    state_version = cached_state.updated_at.isoformat()

            cache_descriptor = {
                "source_id": source_id,
                "dataset": dataset_signature,
                "base": base,
                "preprocessing": preprocessing_config or ({"state_version": state_version} if state_version else None),
            }
            cache_key = self._make_cache_key("column_insights", cache_descriptor)

            cached_payload = await self.cache.get(cache_key)
            if cached_payload:
                payload_copy = copy.deepcopy(cached_payload)
                payload_copy["analysis_timestamp"] = datetime.now().isoformat()
                payload_copy["cache_hit"] = True
                return payload_copy

            preprocessing_report: Optional[PreprocessingReport] = None

            if preprocessing_config:
                try:
                    options = PreprocessingOptions.from_payload(preprocessing_config)
                except Exception as exc:  # pragma: no cover - defensive sanitation
                    logger.warning("Invalid preprocessing configuration for column insights: %s", exc)
                    options = PreprocessingOptions()

                preprocess_start = perf_counter()
                df, preprocessing_report = run_preprocessing_with_state(
                    source_id,
                    df,
                    options,
                    base=base,
                )
                preprocess_elapsed = perf_counter() - preprocess_start
            else:
                if cached_state:
                    df = cached_state.dataframe
                    preprocessing_report = cached_state.report
                preprocess_elapsed = 0.0

            build_start = perf_counter()
            payload = self._build_column_insights_payload(df, source_id, data_info, preprocessing_report)
            build_elapsed = perf_counter() - build_start

            payload["cache_hit"] = False
            await self.cache.set(cache_key, copy.deepcopy(payload), ttl=self.column_insights_cache_ttl)

            total_elapsed = perf_counter() - timer_start
            logger.debug(
                "Column insights timings for %s — load: %.2f ms, preprocess: %.2f ms, build: %.2f ms, total: %.2f ms",
                source_id,
                load_elapsed * 1000,
                preprocess_elapsed * 1000,
                build_elapsed * 1000,
                total_elapsed * 1000,
            )
            return payload

        except Exception as e:
            log_data_action("GET_COLUMN_INSIGHTS", success=False, details=str(e))
            logger.error("Error getting column insights: %s", e)
            return {
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "message": "Unable to process column insights request.",
            }

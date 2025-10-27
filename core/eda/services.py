"""Small EDA service module.

Provides a lightweight service class for exploratory data analysis tasks.
This is a scaffold that integrates with the project's async DB session and
existing DataIngestionService where appropriate.
"""
from typing import Any, Dict, List, Optional
from collections import defaultdict
from pathlib import Path
import asyncio
import logging
import pandas as pd
import json
import numpy as np
import time
import hashlib


PREVIEW_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "preview_samples"
PREVIEW_CACHE_TTL_SECONDS = 300

from sqlalchemy.ext.asyncio import AsyncSession
from core.data_ingestion.service import DataIngestionService
from core.data_ingestion.serialization import JSONSafeSerializer
from core.utils.logging_utils import log_data_action
from core.eda.text_analysis import get_text_insights
from config import get_settings

logger = logging.getLogger(__name__)

PII_PATTERN_KEYWORDS = (
    "email",
    "phone",
    "ssn",
    "social_security",
    "credit",
    "card",
    "person",
    "name",
    "passport",
    "bank",
    "iban",
    "account",
    "ip",
)


def _pattern_type_is_pii(pattern_type: str) -> bool:
    normalized = (pattern_type or "").lower()
    return any(keyword in normalized for keyword in PII_PATTERN_KEYWORDS)


RECOMMENDATION_PRIORITY_META: Dict[str, Dict[str, Any]] = {
    "critical": {"score": 100, "label": "Critical"},
    "high": {"score": 85, "label": "High"},
    "medium": {"score": 70, "label": "Medium"},
    "strategic": {"score": 60, "label": "Strategic"},
    "advanced": {"score": 50, "label": "Advanced"},
    "low": {"score": 40, "label": "Low"},
    "general": {"score": 60, "label": "General"},
}


RECOMMENDATION_CATEGORY_META: Dict[str, Dict[str, Any]] = {
    "data_quality": {
        "label": "Data Quality",
        "default_tags": {"data_quality", "missing_data"},
    },
    "privacy": {
        "label": "Privacy & Governance",
        "default_tags": {"privacy"},
    },
    "text_preprocessing": {
        "label": "Text Preprocessing",
        "default_tags": {"text", "nlp", "text_quality"},
    },
    "categorical_encoding": {
        "label": "Categorical Encoding",
        "default_tags": {"categorical", "encoding"},
    },
    "feature_engineering": {
        "label": "Feature Engineering",
        "default_tags": {"feature_engineering"},
    },
    "pattern_detection": {
        "label": "Pattern Detection",
        "default_tags": {"pattern_detection"},
    },
    "text_quality": {
        "label": "Text Quality",
        "default_tags": {"text_quality"},
    },
    "advanced_analysis": {
        "label": "Advanced Analysis",
        "default_tags": {"advanced_analysis"},
    },
    "project_roadmap": {
        "label": "Project Roadmap",
        "default_tags": {"project_roadmap"},
    },
    "general": {
        "label": "General Guidance",
        "default_tags": {"general"},
    },
}


RECOMMENDATION_SIGNAL_TAGS: Dict[str, List[str]] = {
    "missing_data": ["missing_data", "nulls"],
    "empty_column": ["missing_data", "empty_column"],
    "low_variance": ["constant_feature", "low_variance"],
    "outlier": ["outlier", "distribution"],
    "skewness": ["skewness", "distribution"],
    "multicollinearity": ["multicollinearity", "correlation"],
    "pii": ["pii", "privacy"],
    "text_quality": ["text_quality", "nlp"],
    "nlp": ["nlp", "text"],
    "pattern_detection": ["pattern_detection"],
    "high_cardinality": ["high_cardinality", "categorical"],
    "project_roadmap": ["project_roadmap"],
    "feature_engineering": ["feature_engineering"],
}


def _normalize_slug(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"})
    return normalized or None


class EDAService:
    """Service responsible for lightweight exploratory data operations.

    Responsibilities (scaffold):
    - Load a small preview for a source
    - Compute basic column statistics (counts, nulls, unique)
    - Return metadata useful for EDA UI
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.data_service = DataIngestionService(session)
        try:
            self.settings = get_settings()
        except Exception:  # pragma: no cover - defensive fallback
            self.settings = None

    # ------------------------------------------------------------------
    # Preview caching helpers
    # ------------------------------------------------------------------
    def _ensure_preview_cache_dir(self) -> None:
        try:
            PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("Could not ensure preview cache directory exists: %s", exc)

    def _build_preview_cache_path(self, file_path: Path, sample_size: int, mode: str) -> Optional[Path]:
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            return None

        signature = f"{file_path.resolve()}|{stat.st_mtime}|{stat.st_size}|{sample_size}|{mode}"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
        return PREVIEW_CACHE_DIR / f"preview_{digest}.json"

    def _load_cached_preview(self, cache_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not cache_path or not cache_path.exists():
            return None

        if PREVIEW_CACHE_TTL_SECONDS > 0:
            age = time.time() - cache_path.stat().st_mtime
            if age > PREVIEW_CACHE_TTL_SECONDS:
                return None

        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
                if isinstance(cached, dict):
                    cached["meta"] = cached.get("meta", {})
                    cached["meta"]["from_cache"] = True
                    return cached
        except json.JSONDecodeError:
            logger.warning("Corrupted preview cache at %s; ignoring", cache_path)
        except Exception as exc:
            logger.debug("Failed to load preview cache %s: %s", cache_path, exc)
        return None

    def _store_cached_preview(self, cache_path: Optional[Path], payload: Dict[str, Any]) -> None:
        if not cache_path:
            return

        try:
            self._ensure_preview_cache_dir()
            payload_with_meta = dict(payload)
            meta = dict(payload_with_meta.get("meta") or {})
            meta["cached_at"] = time.time()
            payload_with_meta["meta"] = meta

            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload_with_meta, handle, ensure_ascii=False)
        except Exception as exc:
            logger.debug("Unable to persist preview cache %s: %s", cache_path, exc)

    def _preview_source_sync(
        self,
        source_id: str,
        sample_size: int,
        mode: str,
        force_refresh: bool,
        initial_file_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Blocking implementation of preview generation executed in a worker thread."""

        uploads_dir = Path(__file__).parent.parent.parent / "uploads" / "data"
        file_path = initial_file_path

        if not file_path and uploads_dir.exists():
            file_path = next(
                (fp for fp in uploads_dir.iterdir() if fp.is_file() and fp.name.startswith(f"{source_id}_")),
                None,
            )

        if not file_path:
            return {
                "success": False,
                "error": f"Data source '{source_id}' not found",
                "message": "Cannot preview - data source does not exist.",
            }

        cache_path = self._build_preview_cache_path(file_path, sample_size, mode)
        if not force_refresh:
            cached_response = self._load_cached_preview(cache_path)
            if cached_response:
                cached_response.setdefault("source_id", source_id)
                return cached_response

        df = None
        total_rows = 0
        file_extension = file_path.suffix.lower()

        try:
            if file_extension == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_rows = max(sum(1 for line in f) - 1, 0)

                if mode == "first_last" and sample_size < total_rows:
                    half_size = max(sample_size // 2, 1)
                    df_head = pd.read_csv(file_path, nrows=half_size)
                    skip_start = max(total_rows - half_size, 0)
                    df_tail = pd.read_csv(file_path, skiprows=range(1, skip_start + 1))
                    df = pd.concat([df_head, df_tail], ignore_index=True)
                else:
                    df = pd.read_csv(file_path, nrows=sample_size)

            elif file_extension in ['.xlsx', '.xls']:
                df_full = pd.read_excel(file_path)
                total_rows = len(df_full)

                if mode == "first_last" and sample_size < total_rows:
                    half_size = max(sample_size // 2, 1)
                    df = pd.concat([
                        df_full.head(half_size),
                        df_full.tail(half_size)
                    ], ignore_index=True)
                else:
                    df = df_full.head(sample_size)
                del df_full

            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_rows = len(data)
                        if mode == "first_last" and sample_size < total_rows:
                            half_size = max(sample_size // 2, 1)
                            selected_data = data[:half_size] + data[-half_size:]
                        else:
                            selected_data = data[:sample_size]
                        df = pd.DataFrame(selected_data)
                    else:
                        total_rows = 1
                        df = pd.DataFrame([data])
            else:
                return {
                    "success": True,
                    "preview": {
                        "columns": [],
                        "sample_data": [],
                        "total_rows": 0,
                        "sample_size": 0,
                        "estimated_total_rows": 0,
                        "message": f"Preview not available for {file_extension} files"
                    },
                    "source_id": source_id,
                    "meta": {"from_cache": False},
                }

        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to preview file: {str(exc)}",
                "message": "Unable to read or parse the uploaded file.",
            }

        if df is None or df.empty:
            return {
                "success": False,
                "error": f"Data source '{source_id}' not found",
                "message": "Cannot preview - data source does not exist.",
            }

        sample_data = JSONSafeSerializer.clean_for_json(df.to_dict('records'))

        preview_data = {
            "columns": df.columns.tolist(),
            "sample_data": sample_data,
            "total_rows": total_rows,
            "sample_size": len(df),
            "estimated_total_rows": total_rows,
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        safe_preview = JSONSafeSerializer.clean_for_json(preview_data)

        response_payload = {
            "success": True,
            "preview": safe_preview,
            "source_id": source_id,
            "meta": {"from_cache": False},
        }

        self._store_cached_preview(cache_path, response_payload)
        return response_payload

    async def preview_source(
        self,
        source_id: str,
        sample_size: int = 100,
        mode: str = "head",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Return a small data preview with complex file handling and sampling modes."""

        try:
            cap = None
            if self.settings and hasattr(self.settings, "is_field_set") and self.settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT"):
                cap = self.settings.EDA_GLOBAL_SAMPLE_LIMIT
            else:
                cap = 1000
            if cap is not None:
                sample_size = min(sample_size, cap)

            log_data_action(
                "GET_PREVIEW",
                details=f"source_id:{source_id}, sample_size:{sample_size}, mode:{mode}",
            )

            file_path: Optional[Path] = await self.get_source_file_path(source_id)

            return await asyncio.to_thread(
                self._preview_source_sync,
                source_id,
                sample_size,
                mode,
                force_refresh,
                file_path,
            )

        except Exception as e:
            log_data_action("GET_PREVIEW", success=False, details=str(e))
            return {
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "message": "Unable to process preview request."
            }


    async def get_source_file_path(self, source_id: str) -> Optional[Path]:
        """Return the file path for a source if available.

        Useful for initiating heavier EDA operations that require loading the full file.
        """
        try:
            source = await self.data_service.get_data_source(source_id)
        except Exception as exc:
            logger.debug("Failed to load data source %s from database: %s", source_id, exc)
            source = None

        if source:
            cfg = getattr(source, "config", {}) or {}
            for key in ("file_path", "filepath", "path"):
                candidate = cfg.get(key)
                if candidate:
                    expanded = Path(candidate)
                    if expanded.exists():
                        return expanded

        uploads_dir = Path(__file__).parent.parent.parent / "uploads" / "data"
        if uploads_dir.exists():
            possible = next(
                (fp for fp in uploads_dir.iterdir() if fp.is_file() and fp.name.startswith(f"{source_id}_")),
                None,
            )
            if possible:
                return possible

        return None


    async def quality_report(self, source_id: str, sample_size: int = 500) -> Dict[str, Any]:
        """Generate comprehensive quality report with text analysis without blocking the event loop."""

        requested_sample_size = sample_size
        cap = None
        if self.settings and hasattr(self.settings, "is_field_set") and self.settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT"):
            cap = self.settings.EDA_GLOBAL_SAMPLE_LIMIT
        if cap is not None:
            sample_size = min(sample_size, cap)

        try:
            return await asyncio.to_thread(
                self._quality_report_sync,
                source_id,
                sample_size,
                requested_sample_size,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            log_data_action("GET_QUALITY_REPORT", success=False, details=str(exc))
            return {"success": False, "error": f"Internal server error: {str(exc)}"}

    def _quality_report_sync(
        self,
        source_id: str,
        sample_size: int,
        requested_sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Synchronous quality report generation executed in a worker thread."""

        try:
            log_details = f"source_id:{source_id}, sample_size:{sample_size}"
            if requested_sample_size is not None and requested_sample_size != sample_size:
                log_details += f" (requested:{requested_sample_size})"
            log_data_action("GET_QUALITY_REPORT", details=log_details)

            text_sample_limit = None
            random_state = 42
            if self.settings:
                text_sample_limit = getattr(self.settings, "EDA_QUALITY_REPORT_TEXT_SAMPLE_LIMIT", None)
                random_state = getattr(self.settings, "ML_RANDOM_STATE", random_state)

            uploads_dir = Path(__file__).parent.parent.parent / "uploads" / "data"
            df = None
            file_path = None

            if uploads_dir.exists():
                for fp in uploads_dir.iterdir():
                    if fp.is_file() and fp.name.startswith(f"{source_id}_"):
                        file_path = fp
                        try:
                            file_extension = fp.suffix.lower()

                            total_rows = 0
                            if file_extension == ".csv":
                                with open(fp, "r", encoding="utf-8") as f:
                                    total_rows = sum(1 for line in f) - 1
                                df = pd.read_csv(fp, nrows=sample_size)
                            elif file_extension in [".xlsx", ".xls"]:
                                df_full = pd.read_excel(fp)
                                total_rows = len(df_full)
                                df = df_full.head(sample_size)
                                del df_full
                            elif file_extension == ".json":
                                with open(fp, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    if isinstance(data, list):
                                        total_rows = len(data)
                                        df = pd.DataFrame(data[:sample_size])
                                    else:
                                        total_rows = 1
                                        df = pd.DataFrame([data])
                            break
                        except Exception as exc:  # pragma: no cover - defensive fallback
                            log_data_action("LOAD_DATA_ERROR", success=False, details=str(exc))
                            return {"success": False, "error": f"Error loading data: {str(exc)}"}

            if df is None or df.empty:
                return {"success": False, "error": "Data source not found or empty"}

            total_cols = len(df.columns)
            file_size = file_path.stat().st_size if file_path else None

            dtype_info: Dict[str, List[str]] = {}
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                dtype_info.setdefault(dtype_str, []).append(col)

            column_quality: List[Dict[str, Any]] = []
            potential_issues: List[Dict[str, Any]] = []
            text_pattern_tracker: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "columns": set()})
            text_quality_flag_tracker: Dict[str, set] = defaultdict(set)
            pii_columns = set()

            for col in df.columns:
                try:
                    series = df[col]
                    col_data: Dict[str, Any] = {
                        "name": col,
                        "dtype": str(series.dtype),
                        "non_null_count": int(series.notna().sum()),
                        "null_count": int(series.isna().sum()),
                        "null_percentage": float(series.isna().sum() / len(series) * 100),
                        "unique_count": int(series.nunique(dropna=False)),
                        "unique_percentage": float(series.nunique(dropna=False) / len(series) * 100)
                        if len(series) > 0
                        else 0,
                        "memory_usage": int(series.memory_usage(deep=True)),
                    }

                    if pd.api.types.is_bool_dtype(series):
                        col_data.update(
                            {
                                "data_category": "boolean",
                                "true_count": int(series.sum()),
                                "false_count": int((~series).sum()),
                                "true_percentage": float(series.sum() / len(series) * 100) if len(series) > 0 else 0,
                            }
                        )
                    elif pd.api.types.is_numeric_dtype(series):
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0:
                            col_data.update(
                                {
                                    "min_value": float(non_null_series.min()),
                                    "max_value": float(non_null_series.max()),
                                    "mean_value": float(non_null_series.mean()),
                                    "has_negative": bool((non_null_series < 0).any()),
                                    "has_zero": bool((non_null_series == 0).any()),
                                    "data_category": "numeric",
                                }
                            )

                            if col_data["max_value"] > 150 and "age" in col.lower():
                                potential_issues.append(
                                    {
                                        "type": "warning",
                                        "column": col,
                                        "message": (
                                            "Column '{col}' has values > 150. Please verify if this is correct for age data."
                                        ).format(col=col),
                                    }
                                )

                    elif pd.api.types.is_datetime64_any_dtype(series):
                        col_data["data_category"] = "datetime"
                        potential_issues.append(
                            {
                                "type": "info",
                                "column": col,
                                "message": f"Column '{col}' was detected as datetime.",
                            }
                        )

                    elif series.dtype == "object":
                        try:
                            text_insights = get_text_insights(series, col)
                            col_data.update(text_insights)

                            patterns = text_insights.get("patterns") or []
                            if patterns:
                                col_data["text_patterns"] = patterns
                                for pattern in patterns:
                                    pattern_type = pattern.get("type", "unknown")
                                    normalized_type = (pattern_type or "unknown").lower()
                                    pattern_info = text_pattern_tracker[normalized_type]
                                    pattern_info["count"] += int(pattern.get("count", 0) or 0)
                                    pattern_info["columns"].add(col)

                                    potential_issues.append(
                                        {
                                            "type": "info",
                                            "column": col,
                                            "message": (
                                                f"Column '{col}' contains {pattern_type} patterns (~"
                                                f"{pattern.get('percentage', 0):.1f}% of rows)."
                                            ),
                                        }
                                    )

                                    if _pattern_type_is_pii(pattern_type):
                                        pii_columns.add(col)
                                        potential_issues.append(
                                            {
                                                "type": "warning",
                                                "column": col,
                                                "message": (
                                                    f"Column '{col}' may contain PII ({pattern_type}). "
                                                    "Ensure appropriate masking or governance."
                                                ),
                                            }
                                        )

                            quality_flags = text_insights.get("quality_flags") or []
                            if quality_flags:
                                col_data["text_quality_flags"] = quality_flags
                                for flag in quality_flags:
                                    text_quality_flag_tracker[flag].add(col)

                                potential_issues.append(
                                    {
                                        "type": "info",
                                        "column": col,
                                        "message": (
                                            f"Text quality flags detected: {', '.join(quality_flags)}."
                                        ),
                                    }
                                )

                            if text_insights.get("text_category") == "free_text":
                                potential_issues.append(
                                    {
                                        "type": "info",
                                        "column": col,
                                        "message": (
                                            f"Column '{col}' contains free-text data suitable for NLP analysis."
                                        ),
                                    }
                                )
                                for rec in text_insights.get("eda_recommendations", []):
                                    potential_issues.append(
                                        {
                                            "type": "recommendation",
                                            "column": col,
                                            "message": f"Text Analysis: {rec}",
                                        }
                                    )
                            elif text_insights.get("text_category") == "categorical":
                                if text_insights.get("unique_count", 0) > 50:
                                    potential_issues.append(
                                        {
                                            "type": "warning",
                                            "column": col,
                                            "message": (
                                                f"Column '{col}' has {text_insights.get('unique_count', 0)} categories. "
                                                "Consider grouping or encoding."
                                            ),
                                        }
                                    )

                            if text_insights.get("avg_text_length", 0) > 200:
                                potential_issues.append(
                                    {
                                        "type": "info",
                                        "column": col,
                                        "message": (
                                            f"Column '{col}' contains long text (avg: {text_insights.get('avg_text_length', 0):.0f} chars). "
                                            "Consider text preprocessing."
                                        ),
                                    }
                                )
                        except Exception:  # pragma: no cover - defensive fallback
                            col_data.update(
                                {
                                    "data_category": "text",
                                    "text_category": "unknown",
                                    "avg_text_length": 0,
                                }
                            )

                    if col_data["unique_percentage"] > 95:
                        potential_issues.append(
                            {
                                "type": "warning",
                                "column": col,
                                "message": (
                                    f"Column '{col}' has {col_data['unique_percentage']:.2f}% unique values. It might be an identifier."
                                ),
                            }
                        )

                    if col_data["null_percentage"] > 30:
                        potential_issues.append(
                            {
                                "type": "warning",
                                "column": col,
                                "message": (
                                    f"Column '{col}' has {col_data['null_percentage']:.2f}% missing values."
                                ),
                            }
                        )

                    column_quality.append(col_data)

                except Exception:  # pragma: no cover - defensive fallback
                    column_quality.append(
                        {
                            "name": col,
                            "dtype": str(df[col].dtype),
                            "non_null_count": 0,
                            "null_count": len(df),
                            "null_percentage": 100.0,
                            "unique_count": 0,
                            "unique_percentage": 0.0,
                            "memory_usage": 0,
                            "data_category": "unknown",
                        }
                    )

            try:
                column_quality.sort(key=lambda x: x.get("null_percentage", 0), reverse=True)
            except (KeyError, TypeError) as exc:  # pragma: no cover - defensive fallback
                log_data_action("SORT_COLUMNS_ERROR", success=False, details=str(exc))

            detected_patterns_summary = [
                {
                    "pattern_type": pattern_type,
                    "total_count": details["count"],
                    "columns": sorted(details["columns"]),
                }
                for pattern_type, details in text_pattern_tracker.items()
                if details["count"] > 0
            ]

            text_quality_flags_summary = [
                {"flag": flag, "columns": sorted(columns)}
                for flag, columns in text_quality_flag_tracker.items()
                if columns
            ]

            try:
                overall_completeness = (
                    df.notna().sum().sum() / (len(df) * len(df.columns)) * 100
                ) if len(df) > 0 and len(df.columns) > 0 else 100
            except (ZeroDivisionError, TypeError, ValueError):
                overall_completeness = 0.0

            quality_checks = {
                "no_duplicate_rows": bool(df.duplicated().sum() == 0),
                "no_empty_columns": bool(not any(df[col].isnull().all() for col in df.columns)),
                "reasonable_missing_data": bool(all(col.get("null_percentage", 0) < 50 for col in column_quality)),
                "consistent_data_types": bool(len(dtype_info) < len(df.columns) * 0.8),
                "no_constant_columns": bool(all(col["unique_count"] > 1 for col in column_quality)),
            }

            try:
                overall_quality_score = sum(int(v) for v in quality_checks.values()) / len(quality_checks) * 100
            except (ZeroDivisionError, TypeError, ValueError):
                overall_quality_score = 50.0

            text_columns = [col for col in column_quality if col.get("data_category") == "text"]
            free_text_columns = [col for col in text_columns if col.get("text_category") in ["free_text", "descriptive_text"]]
            categorical_columns = [col for col in text_columns if col.get("text_category") == "categorical"]

            def build_recommendation(
                title: str,
                description: str,
                *,
                priority: str = "medium",
                category: str = "general",
                columns: Optional[List[str]] = None,
                action: Optional[str] = None,
                why_it_matters: Optional[str] = None,
                feature_impact: Optional[str] = None,
                tags: Optional[List[str]] = None,
                metrics: Optional[List[Dict[str, Any]]] = None,
                focus_areas: Optional[List[str]] = None,
                signal_type: Optional[str] = None,
                code: Optional[str] = None,
                confidence: Optional[float] = None,
                references: Optional[List[str]] = None,
            ) -> Dict[str, Any]:
                normalized_priority = _normalize_slug(priority) or "medium"
                priority_meta = RECOMMENDATION_PRIORITY_META.get(
                    normalized_priority,
                    RECOMMENDATION_PRIORITY_META["medium"],
                )
                normalized_category = _normalize_slug(category) or "general"
                category_meta = RECOMMENDATION_CATEGORY_META.get(
                    normalized_category,
                    {
                        "label": normalized_category.replace("_", " ").title(),
                        "default_tags": {normalized_category},
                    },
                )

                recommendation: Dict[str, Any] = {
                    "title": title,
                    "description": description,
                    "priority": normalized_priority,
                    "priority_score": priority_meta["score"],
                    "priority_label": priority_meta["label"],
                    "category": normalized_category,
                    "category_label": category_meta["label"],
                }

                if action:
                    recommendation["action"] = action
                if code:
                    recommendation["code"] = code
                if why_it_matters:
                    recommendation["why_it_matters"] = why_it_matters
                if feature_impact:
                    recommendation["feature_impact"] = feature_impact
                if confidence is not None:
                    try:
                        recommendation["confidence"] = float(confidence)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        pass
                if references:
                    cleaned_refs = [str(ref).strip() for ref in references if ref]
                    if cleaned_refs:
                        recommendation["references"] = cleaned_refs

                normalized_columns = sorted({str(col) for col in (columns or []) if col})
                if normalized_columns:
                    recommendation["columns"] = normalized_columns

                normalized_signal = _normalize_slug(signal_type)
                if normalized_signal:
                    recommendation["signal_type"] = normalized_signal

                normalized_focus = {normalized_category}
                if normalized_signal:
                    normalized_focus.add(normalized_signal)
                for focus in focus_areas or []:
                    normalized_focus_value = _normalize_slug(focus)
                    if normalized_focus_value:
                        normalized_focus.add(normalized_focus_value)
                if normalized_focus:
                    recommendation["focus_areas"] = sorted(normalized_focus)

                aggregated_tags = set(category_meta.get("default_tags", set()))
                if normalized_signal:
                    aggregated_tags.update(RECOMMENDATION_SIGNAL_TAGS.get(normalized_signal, []))
                if tags:
                    for tag in tags:
                        normalized_tag = _normalize_slug(tag)
                        if normalized_tag:
                            aggregated_tags.add(normalized_tag)
                aggregated_tags.update(recommendation.get("focus_areas", []))
                if aggregated_tags:
                    recommendation["tags"] = sorted(aggregated_tags)

                cleaned_metrics: List[Dict[str, Any]] = []
                if metrics:
                    for metric in metrics:
                        if not isinstance(metric, dict):
                            continue
                        metric_name = metric.get("name")
                        if not metric_name:
                            continue
                        cleaned_metric: Dict[str, Any] = {"name": str(metric_name)}
                        if "value" in metric:
                            cleaned_metric["value"] = metric["value"]
                        if metric.get("unit"):
                            cleaned_metric["unit"] = str(metric["unit"])
                        if metric.get("description"):
                            cleaned_metric["description"] = str(metric["description"])
                        if metric.get("reference"):
                            cleaned_metric["reference"] = str(metric["reference"])
                        cleaned_metrics.append(cleaned_metric)
                if cleaned_metrics:
                    recommendation["metrics"] = cleaned_metrics

                recommendation["meta"] = {
                    "generated_by": "quality_report",
                    "version": 2,
                }
                if normalized_signal:
                    recommendation["meta"]["signal_type"] = normalized_signal

                return recommendation

            eda_recommendations: List[Dict[str, Any]] = []

            high_missing_cols = [col for col in column_quality if col.get("null_percentage", 0) > 30]
            if high_missing_cols:
                missing_columns = [col["name"] for col in high_missing_cols]
                worst_missing = max((col.get("null_percentage", 0) for col in high_missing_cols), default=0)
                missing_priority = "critical" if worst_missing >= 90 else "high" if worst_missing >= 70 else "medium"
                eda_recommendations.append(
                    build_recommendation(
                        "Handle Missing Data",
                        (
                            "Consider targeted imputations or column removal for high-missing columns: "
                            f"{', '.join(missing_columns[:5])}"
                        ),
                        priority=missing_priority,
                        category="data_quality",
                        columns=missing_columns,
                        action=(
                            "Profile missingness (df[cols].isna().mean()) and drop columns exceeding acceptable thresholds, "
                            "or apply domain-appropriate imputers."
                        ),
                        why_it_matters=(
                            "Significant missingness introduces bias and fragile pipelines. Addressing it early keeps feature"
                            " engineering steps consistent across training and production."
                        ),
                        feature_impact=(
                            "Stabilizes downstream models by ensuring each engineered feature has reliable coverage and lowers the"
                            " risk of runtime null explosions."
                        ),
                        tags=["missing_data", "nulls"],
                        metrics=[
                            {"name": "columns_affected", "value": len(missing_columns)},
                            {"name": "max_missing_pct", "value": round(float(worst_missing), 2), "unit": "%"},
                        ],
                        focus_areas=["data_quality", "missing_data"],
                        signal_type="missing_data",
                    )
                )

            empty_columns = [col["name"] for col in column_quality if col.get("non_null_count", 0) == 0]
            if empty_columns:
                eda_recommendations.append(
                    build_recommendation(
                        "Drop Empty Columns",
                        (
                            f"The following columns are empty in the sampled data: {', '.join(empty_columns[:6])}. "
                            "Retaining them will add noise without signal."
                        ),
                        priority="high",
                        category="data_quality",
                        columns=empty_columns,
                        action="Remove empty columns or validate upstream ingestion to avoid schema mismatches.",
                        why_it_matters="Empty columns consume maintenance time and can break automated feature stores.",
                        feature_impact="Keeps feature sets lean and avoids null-filled placeholders in production.",
                        tags=["empty_column", "missing_data"],
                        metrics=[{"name": "empty_columns", "value": len(empty_columns)}],
                        focus_areas=["data_quality", "missing_data"],
                        signal_type="empty_column",
                    )
                )

            constant_columns = [
                col["name"]
                for col in column_quality
                if col.get("unique_count", 0) <= 1 and col.get("non_null_count", 0) > 0
            ]
            if constant_columns:
                eda_recommendations.append(
                    build_recommendation(
                        "Remove Constant Features",
                        (
                            "Columns with only a single value provide no predictive signal and should be removed before"
                            " modeling: " + ", ".join(constant_columns[:6])
                        ),
                        priority="medium",
                        category="data_quality",
                        columns=constant_columns,
                        action="Drop low-variance columns to simplify models and prevent redundant feature engineering.",
                        why_it_matters="Constant columns inflate feature space and can mask true variance during scaling.",
                        feature_impact="Improves model training speed and keeps feature registries cleaner.",
                        tags=["low_variance", "constant_feature"],
                        metrics=[{"name": "low_variance_columns", "value": len(constant_columns)}],
                        focus_areas=["data_quality", "feature_engineering"],
                        signal_type="low_variance",
                    )
                )

            if free_text_columns:
                free_text_names = [col["name"] for col in free_text_columns]
                eda_recommendations.append(
                    build_recommendation(
                        "NLP Analysis Opportunity",
                        (
                            f"Columns like {', '.join(free_text_names[:4])} contain free text suitable for sentiment analysis, "
                            "topic modeling, or classification."
                        ),
                        priority="medium",
                        category="text_preprocessing",
                        columns=free_text_names,
                        action="Establish a text preprocessing pipeline (tokenization, stop-word removal, embeddings).",
                        why_it_matters=(
                            "Free-text features often hold qualitative signals that classic numeric features miss. Preparing them"
                            " thoughtfully unlocks richer feature engineering options."
                        ),
                        feature_impact="Creates high-value NLP embeddings and document-level features for downstream models.",
                        tags=["nlp", "text"],
                        metrics=[{"name": "text_columns", "value": len(free_text_names)}],
                        focus_areas=["text_preprocessing", "feature_engineering"],
                        signal_type="nlp",
                    )
                )

            if categorical_columns:
                high_cardinality_cat = [col for col in categorical_columns if col.get("unique_count", 0) > 50]
                if high_cardinality_cat:
                    cat_names = [col["name"] for col in high_cardinality_cat]
                    eda_recommendations.append(
                        build_recommendation(
                            "High Cardinality Categories",
                            (
                                f"Categorical columns {', '.join(cat_names[:3])} have many distinct levels. "
                                "Group rare categories or use target encoding."
                            ),
                            priority="medium",
                            category="categorical_encoding",
                            columns=cat_names,
                            action="Apply frequency / target encoding or bucket infrequent labels before modeling.",
                            why_it_matters=(
                                "High-cardinality categoricals can explode feature space and overfit linear models. Selecting the"
                                " right encoding keeps engineered features stable."
                            ),
                            feature_impact="Reduces sparsity and improves generalization for encoded categorical features.",
                            tags=["high_cardinality", "categorical", "encoding"],
                            metrics=[{"name": "columns_affected", "value": len(cat_names)}],
                            focus_areas=["categorical_encoding", "feature_engineering"],
                            signal_type="high_cardinality",
                        )
                    )

            if detected_patterns_summary:
                for pattern in detected_patterns_summary:
                    pattern_type = pattern["pattern_type"]
                    pattern_columns = pattern["columns"]
                    if _pattern_type_is_pii(pattern_type):
                        continue
                    normalized_pattern_signal = _normalize_slug(pattern_type) or "pattern_detection"
                    eda_recommendations.append(
                        build_recommendation(
                            f"Detected {pattern_type.title()} Patterns",
                            (
                                f"Column(s) {', '.join(pattern_columns)} contain {pattern_type} patterns. "
                                "Consider dedicated parsing or validation logic."
                            ),
                            priority="medium",
                            category="pattern_detection",
                            columns=pattern_columns,
                            action="Create validators or extract structured features from the detected patterns.",
                            why_it_matters=(
                                "Recognizing structured patterns early lets you engineer features (e.g., domain splits, regex"
                                " extracts) before training pipelines require them."
                            ),
                            feature_impact="Enables creation of specialized features and validation checkpoints for patterned data.",
                            tags=[pattern_type, "pattern_detection"],
                            metrics=[
                                {"name": "columns_affected", "value": len(pattern_columns)},
                                {"name": "total_pattern_hits", "value": pattern.get("total_count", 0)},
                            ],
                            focus_areas=["pattern_detection", "feature_engineering"],
                            signal_type=normalized_pattern_signal,
                        )
                    )

            if pii_columns:
                pii_list = sorted(pii_columns)
                eda_recommendations.append(
                    build_recommendation(
                        "Protect Sensitive Data",
                        (
                            f"Columns {', '.join(pii_list)} may contain PII (names, contacts, IDs, or financial data). "
                            "Mask, hash, or remove before sharing or modeling."
                        ),
                        priority="critical",
                        category="privacy",
                        columns=pii_list,
                        action="Implement masking or tokenization and restrict access in downstream workflows.",
                        why_it_matters="PII leakage creates compliance risk and blocks productionization of ML features.",
                        feature_impact="Ensures features derived from sensitive identifiers comply with governance policies.",
                        tags=["pii", "privacy"],
                        metrics=[{"name": "pii_columns", "value": len(pii_list)}],
                        focus_areas=["privacy", "data_quality"],
                        signal_type="pii",
                    )
                )

            if text_quality_flags_summary:
                for flag_summary in text_quality_flags_summary:
                    columns_for_flag = flag_summary["columns"]
                    flag_name = flag_summary["flag"]
                    eda_recommendations.append(
                        build_recommendation(
                            f"Address Text Quality: {flag_name}",
                            (
                                f"Columns {', '.join(columns_for_flag)} exhibit the '{flag_name}' flag. "
                                "Plan preprocessing to resolve this."
                            ),
                            priority="high",
                            category="text_quality",
                            columns=columns_for_flag,
                            action="Apply cleaning steps (normalization, whitespace fixes, casing, de-duplication).",
                            why_it_matters="Poor text hygiene balloons categorical space and weakens NLP feature extraction.",
                            feature_impact="Improves embedding quality and reduces noise in downstream pipelines.",
                            tags=["text_quality", flag_name],
                            metrics=[{"name": "columns_affected", "value": len(columns_for_flag)}],
                            focus_areas=["text_quality", "text_preprocessing"],
                            signal_type="text_quality",
                        )
                    )

            numeric_cols = [col for col in column_quality if col.get("data_category") == "numeric"]
            outlier_analysis: List[Dict[str, Any]] = []
            for col in numeric_cols:
                try:
                    col_name = col["name"]
                    series = pd.to_numeric(df[col_name], errors="coerce")
                    if not series.isna().all():
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            lower_bound = float(Q1 - 1.5 * IQR)
                            upper_bound = float(Q3 + 1.5 * IQR)
                            outliers = series[(series < lower_bound) | (series > upper_bound)]
                            outlier_percentage = len(outliers) / len(series.dropna()) * 100
                            outlier_analysis.append(
                                {
                                    "column": col_name,
                                    "outlier_count": len(outliers),
                                    "outlier_percentage": outlier_percentage,
                                }
                            )
                except Exception:  # pragma: no cover - defensive fallback
                    continue

            if any(item["outlier_percentage"] > 10 for item in outlier_analysis):
                outlier_columns = [item["column"] for item in outlier_analysis if item["outlier_percentage"] > 10]
                max_outlier_pct = max(
                    (item["outlier_percentage"] for item in outlier_analysis if item["outlier_percentage"] > 10),
                    default=0,
                )
                eda_recommendations.append(
                    build_recommendation(
                        "Investigate Outliers",
                        "Some numeric columns have significant outliers. Review these during EDA to understand data patterns.",
                        priority="advanced",
                        category="advanced_analysis",
                        columns=outlier_columns,
                        action="Visualize distributions (boxplots, z-scores) and decide on capping or transformations.",
                        why_it_matters="Extreme values can dominate model training and skew engineered aggregations.",
                        feature_impact="Determines whether to cap, transform, or separate outlier-driven features.",
                        tags=["outlier", "distribution"],
                        metrics=[
                            {"name": "columns_affected", "value": len(outlier_columns)},
                            {"name": "max_outlier_pct", "value": round(float(max_outlier_pct), 2), "unit": "%"},
                        ],
                        focus_areas=["advanced_analysis", "feature_engineering"],
                        signal_type="outlier",
                    )
                )

            skew_threshold = 1.0
            skewed_features: List[Dict[str, Any]] = []
            for col in numeric_cols:
                col_name = col["name"]
                try:
                    series = pd.to_numeric(df[col_name], errors="coerce").dropna()
                except Exception:  # pragma: no cover - defensive fallback
                    continue
                if len(series) < 3:
                    continue
                skew_value = series.skew()
                if pd.isna(skew_value) or abs(skew_value) < skew_threshold:
                    continue
                skewed_features.append({"column": col_name, "skewness": float(skew_value)})

            if skewed_features:
                top_skewed = sorted(skewed_features, key=lambda x: abs(x["skewness"]), reverse=True)[:5]
                eda_recommendations.append(
                    build_recommendation(
                        "Normalize Skewed Distributions",
                        "Several numeric columns are highly skewed; consider transformations before modeling.",
                        priority="advanced",
                        category="advanced_analysis",
                        columns=[item["column"] for item in top_skewed],
                        action="Evaluate log, Box-Cox, or yeo-johnson transforms to stabilize variance.",
                        why_it_matters="Skewed inputs violate model assumptions and reduce feature usefulness.",
                        feature_impact="Improves model calibration and makes derived features more robust.",
                        tags=["skewness", "distribution"],
                        metrics=[
                            {"name": "columns_affected", "value": len(skewed_features)},
                            {
                                "name": "max_abs_skew",
                                "value": round(max(abs(item["skewness"]) for item in skewed_features), 3),
                            },
                        ],
                        focus_areas=["advanced_analysis", "feature_engineering"],
                        signal_type="skewness",
                    )
                )

            multicollinear_pairs: List[Dict[str, Any]] = []
            if len(numeric_cols) >= 2:
                numeric_df = df[[col["name"] for col in numeric_cols]].apply(pd.to_numeric, errors="coerce")
                numeric_df = numeric_df.dropna(axis=1, how="all")
                numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
                if numeric_df.shape[1] >= 2:
                    try:
                        corr_matrix = numeric_df.corr().abs()
                        if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
                            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                            corr_threshold = 0.92
                            for row_name in upper_tri.index:
                                for col_name in upper_tri.columns:
                                    corr_value = upper_tri.at[row_name, col_name]
                                    if pd.notna(corr_value) and corr_value >= corr_threshold:
                                        multicollinear_pairs.append(
                                            {
                                                "columns": (row_name, col_name),
                                                "correlation": float(corr_value),
                                            }
                                        )
                    except Exception:  # pragma: no cover - defensive fallback
                        multicollinear_pairs = []

            if multicollinear_pairs:
                top_pairs = sorted(multicollinear_pairs, key=lambda x: x["correlation"], reverse=True)[:5]
                involved_columns = sorted({col for pair in top_pairs for col in pair["columns"]})
                eda_recommendations.append(
                    build_recommendation(
                        "High Multi-collinearity Detected",
                        (
                            "Strongly correlated features were found (ρ ≥ 0.92). Consider pruning or combining them to"
                            " avoid redundant signals."
                        ),
                        priority="high",
                        category="feature_engineering",
                        columns=involved_columns,
                        action="Inspect correlated pairs, compute VIF, and drop or blend redundant features before modeling.",
                        why_it_matters="Highly correlated inputs inflate variance and distort feature importance metrics.",
                        feature_impact="Clarifies feature attribution and supports more stable coefficients.",
                        tags=["multicollinearity", "correlation"],
                        metrics=[
                            {"name": "correlated_pairs", "value": len(multicollinear_pairs)},
                            {
                                "name": "top_pair_correlation",
                                "value": round(top_pairs[0]["correlation"], 3),
                            },
                        ],
                        focus_areas=["feature_engineering", "advanced_analysis"],
                        signal_type="multicollinearity",
                    )
                )

            eda_recommendations.extend(
                [
                    build_recommendation(
                        "Start with Data Overview",
                        "Begin with descriptive statistics and distributions for numeric, categorical, and text features.",
                        priority="strategic",
                        category="project_roadmap",
                        action="Review df.describe(), value counts, and text summaries to align stakeholders.",
                        why_it_matters="Shared exploratory context keeps teams aligned before heavy feature work begins.",
                        feature_impact="Surfaces early data quirks that influence later feature pipelines.",
                        tags=["project_roadmap", "data_quality"],
                        focus_areas=["project_roadmap"],
                        signal_type="project_roadmap",
                    ),
                    build_recommendation(
                        "Text Preprocessing Strategy",
                        "For text columns, plan cleaning steps such as lowercasing, punctuation removal, stemming, and stop word removal.",
                        priority="medium",
                        category="text_preprocessing",
                        action="Document preprocessing pipeline components and automate them in feature engineering scripts.",
                        why_it_matters="Consistent text normalization prevents subtle bugs between training and production pipelines.",
                        feature_impact="Improves quality of text-derived embeddings and categorical encodings.",
                        tags=["text_preprocessing", "nlp", "feature_engineering"],
                        focus_areas=["text_preprocessing", "feature_engineering"],
                        signal_type="nlp",
                    ),
                    build_recommendation(
                        "Feature Engineering",
                        "Create auxiliary features from text (length, word count, sentiment) and encode categorical variables appropriately.",
                        priority="strategic",
                        category="feature_engineering",
                        action="Prototype feature transformations and record their impact on downstream model performance.",
                        why_it_matters="Feature experiments turn exploratory insights into measurable business lift.",
                        feature_impact="Encourages systematic tracking of feature value and reuse across models.",
                        tags=["feature_engineering"],
                        focus_areas=["feature_engineering"],
                        signal_type="feature_engineering",
                    ),
                ]
            )

            quality_report = {
                "basic_metadata": {
                    "sample_rows": len(df),
                    "total_columns": total_cols,
                    "estimated_total_rows": total_rows,
                    "file_size_bytes": file_size,
                    "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
                    "data_types": dtype_info,
                },
                "quality_metrics": {
                    "overall_completeness": float(overall_completeness),
                    "columns_with_missing": sum(1 for col in column_quality if col.get("null_percentage", 0) > 0),
                    "high_cardinality_columns": sum(1 for col in column_quality if col.get("unique_percentage", 0) > 95),
                    "column_details": column_quality,
                },
                "quality_checks": quality_checks,
                "overall_quality_score": overall_quality_score,
                "missing_data_summary": [
                    {
                        "column": col["name"],
                        "missing_count": col["null_count"],
                        "missing_percentage": col.get("null_percentage", 0),
                    }
                    for col in sorted(column_quality, key=lambda x: x.get("null_percentage", 0), reverse=True)
                ],
                "outlier_analysis": outlier_analysis,
                "data_types": {dtype: len(cols) for dtype, cols in dtype_info.items()},
                "potential_issues": potential_issues,
                "sample_preview": {
                    "columns": list(df.columns),
                    "sample_data": JSONSafeSerializer.clean_for_json(df.head(10)),
                },
                "text_analysis_summary": {
                    "total_text_columns": len(text_columns),
                    "free_text_columns": len(free_text_columns),
                    "categorical_text_columns": len(categorical_columns),
                    "text_column_names": [col["name"] for col in text_columns],
                    "nlp_candidates": [col["name"] for col in free_text_columns],
                    "detected_patterns": detected_patterns_summary,
                    "text_quality_flags": text_quality_flags_summary,
                    "pii_columns": sorted(pii_columns),
                },
                "recommendations": eda_recommendations,
            }

            return {
                "success": True,
                "quality_report": JSONSafeSerializer.clean_for_json(quality_report),
                "source_id": source_id,
                "actual_sample_size": len(df),
                "sample_rows": JSONSafeSerializer.clean_for_json(df.head(10).to_dict("records")),
                "basic_metadata": quality_report["basic_metadata"],
                "quality_metrics": quality_report["quality_metrics"],
            }

        except Exception as exc:  # pragma: no cover - defensive fallback
            log_data_action("GET_QUALITY_REPORT", success=False, details=str(exc))
            return {"success": False, "error": f"Internal server error: {str(exc)}"}
from __future__ import annotations

"""Shared utilities for granular runtime EDA modules."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..analysis_results import (
    AnalysisChart,
    AnalysisInsight,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    dataframe_to_table,
    fig_to_base64,
)


@dataclass
class AnalysisContext:
    """Runtime context for executing analyses."""

    source_id: str
    selected_columns: Optional[List[str]] = None
    column_mapping: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    max_rows_for_analysis: Optional[int] = None
    sampling_strategy: Optional[str] = None
    random_state: Optional[int] = None
    cache_key: Optional[str] = None
    cache_ttl_seconds: Optional[int] = None

    def display_name(self, column: str) -> str:
        mapping = self.column_mapping or {}
        return mapping.get(column, column)

    def should_sample(self, row_count: int) -> bool:
        return self.max_rows_for_analysis is not None and row_count > self.max_rows_for_analysis

    def stratify_by(self) -> Optional[List[str]]:
        if not self.metadata:
            return None
        stratify = self.metadata.get("stratify_by")
        if isinstance(stratify, str):
            return [stratify]
        if isinstance(stratify, (list, tuple)):
            return [str(value) for value in stratify if value]
        return None

    def is_caching_enabled(self) -> bool:
        if self.cache_key is None:
            return False
        if self.cache_ttl_seconds is None:
            return True
        return self.cache_ttl_seconds > 0


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return the numeric columns of the dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return the categorical columns of the dataframe."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return datetime-like columns, attempting to parse object columns as needed."""
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz", "datetime64"]).columns.tolist()
    if datetime_cols:
        return datetime_cols

    parsed_columns: List[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(5)
            if sample.empty:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                if parsed.notna().any():
                    parsed_columns.append(col)
            except Exception:  # pragma: no cover - defensive parsing
                continue
    return parsed_columns


def text_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that appear to contain free-form text."""

    text_like_columns: List[str] = []

    for column in df.columns:
        series = df[column]

        dtype = series.dtype

        if is_bool_dtype(dtype):
            continue

        categorical_like = isinstance(dtype, pd.CategoricalDtype)

        if not (is_string_dtype(dtype) or is_object_dtype(dtype) or categorical_like):
            continue

        sample = series.dropna().head(12)
        if sample.empty:
            continue

        textish = 0
        for value in sample:
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    continue
                if any(char.isalpha() for char in stripped):
                    textish += 1
            else:
                try:
                    text_value = str(value)
                except Exception:  # pragma: no cover - defensive guard
                    continue
                if any(char.isalpha() for char in text_value):
                    textish += 1

        if textish >= max(1, len(sample) // 2):
            text_like_columns.append(column)

    return text_like_columns


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def metric(
    label: str,
    value: Any,
    unit: Optional[str] = None,
    description: Optional[str] = None,
    trend: Optional[str] = None,
) -> AnalysisMetric:
    """Create an analysis metric with safe JSON serialization."""

    return AnalysisMetric(label=label, value=value, unit=unit, description=description, trend=trend)


def insight(level: str, text: str) -> AnalysisInsight:
    """Create an analysis insight."""

    return AnalysisInsight(level=level, text=text)


__all__ = [
    "AnalysisContext",
    "AnalysisChart",
    "AnalysisInsight",
    "AnalysisMetric",
    "AnalysisResult",
    "AnalysisTable",
    "categorical_columns",
    "datetime_columns",
    "dataframe_to_table",
    "fig_to_base64",
    "insight",
    "metric",
    "numeric_columns",
    "text_columns",
]

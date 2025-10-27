from __future__ import annotations

"""Utilities for structured analysis results returned by granular EDA runtime."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import base64
from io import BytesIO
import json
import math
import numpy as np

import matplotlib

# Use a non-interactive backend to avoid display issues when running on servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _normalize_float(value: float) -> Optional[float]:
    """Normalize float-like values to be JSON safe."""
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def make_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return _normalize_float(float(obj))
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif hasattr(obj, '__module__') and 'pandas' in str(obj.__module__):
        # Handle pandas extension dtypes and other pandas objects
        return str(obj)
    elif hasattr(obj, 'dtype') and hasattr(obj.dtype, 'name'):
        # Handle objects with dtype attribute
        return str(obj)
    elif isinstance(obj, type):
        # Handle type objects (like dtype classes)
        return str(obj)
    elif str(type(obj)).startswith("<class 'pandas"):
        # Catch-all for pandas objects
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, float):
        return _normalize_float(obj)
    elif pd.isna(obj):
        return None
    else:
        # Last resort: try to convert to string if all else fails
        try:
            # Test if it's already JSON serializable
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


@dataclass
class AnalysisMetric:
    label: str
    value: Any
    unit: Optional[str] = None
    description: Optional[str] = None
    trend: Optional[str] = None  # 'up', 'down', or None
    
    def __post_init__(self):
        """Ensure value is JSON serializable."""
        self.value = make_json_serializable(self.value)


@dataclass
class AnalysisInsight:
    level: str  # info | warning | success | danger
    text: str


@dataclass
class AnalysisTable:
    title: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    description: Optional[str] = None
    
    def __post_init__(self):
        """Ensure all data is JSON serializable."""
        self.columns = [str(col) for col in self.columns]
        self.rows = [
            {k: make_json_serializable(v) for k, v in row.items()}
            for row in self.rows
        ]


@dataclass
class AnalysisChart:
    title: str
    image: str  # Base64 encoded PNG data URI
    chart_type: str = "matplotlib"
    description: Optional[str] = None


@dataclass
class AnalysisResult:
    analysis_id: str
    title: str
    summary: Optional[str] = None
    status: str = "success"
    metrics: List[AnalysisMetric] = field(default_factory=list)
    insights: List[AnalysisInsight] = field(default_factory=list)
    tables: List[AnalysisTable] = field(default_factory=list)
    charts: List[AnalysisChart] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        return make_json_serializable({
            "analysis_id": self.analysis_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "metrics": [metric.__dict__ for metric in self.metrics],
            "insights": [insight.__dict__ for insight in self.insights],
            "tables": [
                {
                    "title": table.title,
                    "columns": table.columns,
                    "rows": table.rows,
                    "description": table.description,
                }
                for table in self.tables
            ],
            "charts": [chart.__dict__ for chart in self.charts],
            "details": self.details,
        })


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to base64 PNG data URI."""
    buffer = BytesIO()
    fig.tight_layout()
    facecolor = fig.get_facecolor() if fig.get_facecolor() is not None else "white"
    fig.savefig(
        buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor=facecolor,
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def dataframe_to_table(
    df: pd.DataFrame,
    title: str,
    description: Optional[str] = None,
    max_rows: Optional[int] = None,
    round_decimals: Optional[int] = 3,
) -> AnalysisTable:
    display_df = df.copy()
    if round_decimals is not None:
        for column in display_df.select_dtypes(include=["float", "float64", "float32"]).columns:
            display_df[column] = display_df[column].round(round_decimals)

    if max_rows is not None and len(display_df) > max_rows:
        display_df = display_df.head(max_rows)
        if description:
            description = f"{description} (showing first {max_rows} rows)"
        else:
            description = f"Showing first {max_rows} rows"

    # Convert to records and ensure JSON serialization
    rows = display_df.to_dict(orient="records")
    serializable_rows = [
        {k: make_json_serializable(v) for k, v in row.items()}
        for row in rows
    ]

    return AnalysisTable(
        title=title,
        columns=[str(col) for col in display_df.columns],
        rows=serializable_rows,
        description=description,
    )

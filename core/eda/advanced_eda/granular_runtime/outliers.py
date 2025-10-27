from __future__ import annotations

"""Outlier detection and visualization analyses."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    dataframe_to_table,
    fig_to_base64,
    insight,
    numeric_columns,
)


def iqr_outlier_detection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="iqr_outlier_detection",
            title="IQR Outlier Detection",
            summary="No numeric columns available for outlier detection.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to detect outliers.")],
        )

    rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append(
            {
                "Column": col,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": len(outliers),
                "Outlier %": (len(outliers) / len(series) * 100) if len(series) else 0,
            }
        )

    outlier_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        outlier_df,
        title="IQR Outlier Summary",
        description="Outliers detected using the interquartile range criterion.",
        round_decimals=3,
    )

    insights: List = []
    high_outliers = outlier_df[outlier_df["Outlier %"] > 10]
    if not high_outliers.empty:
        insights.append(
            insight(
                "warning",
                "Columns with >10% IQR outliers: " + ", ".join(high_outliers["Column"].tolist()),
            )
        )
    else:
        insights.append(insight("success", "No significant IQR outlier presence detected."))

    return AnalysisResult(
        analysis_id="iqr_outlier_detection",
        title="IQR Outlier Detection",
        summary="Detects outliers based on the interquartile range (1.5×IQR).",
        tables=[table],
        insights=insights,
    )


def zscore_outlier_detection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="zscore_outlier_detection",
            title="Z-Score Outlier Detection",
            summary="No numeric columns available for outlier detection.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to detect z-score outliers.")],
        )

    rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.std(ddof=0) == 0 or series.empty:
            continue
        zscores = np.abs((series - series.mean()) / series.std(ddof=0))
        outliers = series[zscores > 3]
        rows.append(
            {
                "Column": col,
                "Outlier Count": len(outliers),
                "Outlier %": (len(outliers) / len(series) * 100) if len(series) else 0,
                "Max Z-Score": float(zscores.max()) if len(zscores) else None,
            }
        )

    outlier_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        outlier_df,
        title="Z-Score Outlier Summary",
        description="Outliers defined as |z| > 3 for each numeric column.",
        round_decimals=3,
    )

    return AnalysisResult(
        analysis_id="zscore_outlier_detection",
        title="Z-Score Outlier Detection",
        summary="Highlights extreme values using standardized scores.",
        tables=[table],
        insights=[insight("info", "Consider capping or transforming features with high z-score outliers.")],
    )


def visual_outlier_inspection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="visual_outlier_inspection",
            title="Visual Outlier Inspection",
            summary="No numeric columns available for visual inspection.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to visualize outliers.")],
        )

    charts: List[AnalysisChart] = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.boxplot(y=df[col], ax=ax, color="#F56565")
        ax.set_title(f"Outlier Box Plot: {col}")
        charts.append(
            AnalysisChart(
                title=f"Box Plot: {col}",
                image=fig_to_base64(fig),
                description="Box plot visualizing quartiles and potential outliers.",
            )
        )

    return AnalysisResult(
        analysis_id="visual_outlier_inspection",
        title="Visual Outlier Inspection",
        summary="Box plots for each numeric column to spot extreme values.",
        charts=charts,
        insights=[insight("info", "Use box plots to inspect spread and extreme values." )],
    )


def outlier_distribution_visualization(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="outlier_distribution_visualization",
            title="Outlier Distribution Visualization",
            summary="No numeric columns available to visualize outliers.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to visualize outlier distributions.")],
        )

    charts: List[AnalysisChart] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.histplot(series, bins=30, color="#4299E1", ax=ax)
        ax.axvline(lower, color="red", linestyle="--", label="Lower Bound")
        ax.axvline(upper, color="red", linestyle="--", label="Upper Bound")
        ax.legend()
        ax.set_title(f"Outlier Bounds: {col}")
        charts.append(
            AnalysisChart(
                title=f"Outlier Bounds: {col}",
                image=fig_to_base64(fig),
                description="Histogram with IQR-based outlier thresholds.",
            )
        )

    return AnalysisResult(
        analysis_id="outlier_distribution_visualization",
        title="Outlier Distribution Visualization",
        summary="Highlights outlier thresholds on numeric distributions.",
        charts=charts,
        insights=[insight("info", "Investigate observations beyond the red thresholds." )],
    )


def outlier_scatter_matrix(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="outlier_scatter_matrix",
            title="Outlier Scatter Matrix",
            summary="At least two numeric columns are required to build a scatter matrix.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns to generate a scatter matrix.")],
        )

    subset = df[numeric_cols].dropna()
    if subset.empty:
        return AnalysisResult(
            analysis_id="outlier_scatter_matrix",
            title="Outlier Scatter Matrix",
            summary="No valid data available after removing missing values.",
            status="warning",
            insights=[insight("warning", "All selected numeric columns contain only missing values.")],
        )

    if len(subset) < 5:
        return AnalysisResult(
            analysis_id="outlier_scatter_matrix",
            title="Outlier Scatter Matrix",
            summary="Insufficient data for scatter matrix generation.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    f"Only {len(subset)} valid rows available. Need at least 5 rows for meaningful scatter matrix.",
                )
            ],
        )

    try:
        axes = pd.plotting.scatter_matrix(subset, figsize=(6, 6), diagonal="kde", color="#4A5568")
        fig = axes[0, 0].figure
        fig.suptitle(f"Scatter Matrix ({len(subset)} rows)")
        chart = AnalysisChart(
            title="Scatter Matrix",
            image=fig_to_base64(fig),
            description="Scatter matrix aids in spotting multivariate outliers.",
        )

        insights = [
            insight("info", "Use diagonal KDE panels to evaluate distributions."),
            insight("info", f"Analysis based on {len(subset)} valid data points."),
        ]

        return AnalysisResult(
            analysis_id="outlier_scatter_matrix",
            title="Outlier Scatter Matrix",
            summary=f"Scatter matrix for multivariate outlier detection ({len(subset)} rows).",
            charts=[chart],
            insights=insights,
        )
    except Exception as exc:
        return AnalysisResult(
            analysis_id="outlier_scatter_matrix",
            title="Outlier Scatter Matrix",
            summary="Failed to generate scatter matrix.",
            status="error",
            insights=[insight("danger", f"Error generating scatter matrix: {str(exc)}")],
        )


__all__ = [
    "iqr_outlier_detection",
    "zscore_outlier_detection",
    "visual_outlier_inspection",
    "outlier_distribution_visualization",
    "outlier_scatter_matrix",
]

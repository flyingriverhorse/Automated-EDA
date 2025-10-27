from __future__ import annotations

"""Numeric-focused runtime analyses."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    dataframe_to_table,
    fig_to_base64,
    insight,
    metric,
    numeric_columns,
)


def numeric_frequency_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="numeric_frequency_analysis",
            title="Numeric Frequency Analysis",
            summary="No numeric columns available for frequency analysis.",
            status="warning",
            insights=[
                insight("warning", "Select numeric columns to review dominant values or ranges."),
            ],
        )

    rows: List[Dict[str, Any]] = []
    insights: List = []
    metrics = [metric("Numeric Columns", len(numeric_cols)), metric("Rows Analysed", len(df))]

    for col in numeric_cols:
        series = df[col]
        total_count = int(series.size)
        missing_count = int(series.isna().sum())
        non_null_series = series.dropna()
        non_null_count = int(non_null_series.size)

        if non_null_count == 0:
            insights.append(
                insight("warning", f"Column '{col}' only contains missing values and was skipped."),
            )
            continue

        unique_count = int(non_null_series.nunique())
        coverage_mode = "exact values"

        if unique_count <= 25:
            counts = non_null_series.value_counts().head(25)
        else:
            bin_count = min(12, max(5, int(np.sqrt(non_null_count))))
            try:
                categories = pd.cut(non_null_series, bins=bin_count, duplicates="drop")
                counts = categories.value_counts().sort_index()
                coverage_mode = f"{len(counts)} bins"
            except Exception:  # pragma: no cover - defensive fallback
                counts = non_null_series.value_counts().head(20)
                coverage_mode = "top unique values"

        if counts.empty:
            insights.append(
                insight("warning", f"Column '{col}' had no measurable frequency counts."),
            )
            continue

        for bucket, count in counts.items():
            if isinstance(bucket, pd.Interval):
                label = f"{bucket.left:.2f} – {bucket.right:.2f}"
            elif isinstance(bucket, (float, np.floating)):
                label = f"{bucket:.4g}"
            else:
                label = str(bucket)

            percent_total = (count / total_count * 100) if total_count else 0.0
            rows.append(
                {
                    "Column": col,
                    "Value / Bin": label,
                    "Count": int(count),
                    "Percent": round(percent_total, 2),
                    "Coverage": coverage_mode,
                }
            )

        top_share = counts.iloc[0] / non_null_count * 100 if non_null_count else 0.0
        message = (
            f"{col}: top {'bin' if 'bin' in coverage_mode else 'value'} covers {top_share:.1f}% "
            f"of {non_null_count:,} valid values ({coverage_mode})."
        )
        if missing_count:
            missing_pct = missing_count / total_count * 100 if total_count else 0.0
            message += f" Missing values: {missing_count:,} ({missing_pct:.1f}%)."
            rows.append(
                {
                    "Column": col,
                    "Value / Bin": "<Missing>",
                    "Count": missing_count,
                    "Percent": round(missing_pct, 2),
                    "Coverage": "missing",
                }
            )

        insights.append(insight("info", message))

    if not rows:
        return AnalysisResult(
            analysis_id="numeric_frequency_analysis",
            title="Numeric Frequency Analysis",
            summary="Numeric frequency analysis could not be generated for the provided columns.",
            status="warning",
            insights=insights or [insight("warning", "No numeric frequencies available to display.")],
        )

    freq_df = pd.DataFrame(rows)
    freq_df.sort_values(by=["Column", "Count"], ascending=[True, False], inplace=True)

    table = dataframe_to_table(
        freq_df,
        title="Numeric Frequencies",
        description=(
            "Frequency counts for numeric columns. High-cardinality columns are summarised using adaptive bins."
        ),
    )

    return AnalysisResult(
        analysis_id="numeric_frequency_analysis",
        title="Numeric Frequency Analysis",
        summary="Frequency counts and dominant ranges for numeric variables.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


def summary_statistics(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="summary_statistics",
            title="Summary Statistics",
            summary="No numeric columns available for summary statistics.",
            status="warning",
            insights=[
                insight("warning", "Select columns with numeric data to compute summary statistics."),
            ],
        )

    summary_df = df[numeric_cols].describe().transpose()
    table = dataframe_to_table(
        summary_df.reset_index().rename(columns={"index": "Column"}),
        title="Summary Statistics",
        description="Standard descriptive statistics for numeric columns.",
    )

    metrics = [
        metric("Numeric Columns", len(numeric_cols)),
        metric("Observations", len(df)),
    ]

    insights = [
        insight(
            "info",
            "Use median and IQR for skewed distributions; compare mean and median to understand skewness.",
        )
    ]

    return AnalysisResult(
        analysis_id="summary_statistics",
        title="Summary Statistics",
        summary="Descriptive statistics for numeric variables.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


def skewness_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="skewness_analysis",
            title="Skewness Analysis",
            summary="No numeric columns available to compute skewness.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to analyze skewness.")],
        )

    skewness = df[numeric_cols].skew()
    skew_df = pd.DataFrame({"Column": skewness.index, "Skewness": skewness.values})
    skew_df["Skew Category"] = skew_df["Skewness"].apply(
        lambda x: "Approx Normal" if abs(x) < 0.5 else ("Moderate" if abs(x) < 1 else "Strong")
    )

    table = dataframe_to_table(
        skew_df.sort_values("Skewness", key=lambda s: s.abs(), ascending=False),
        title="Skewness by Column",
        description="Skewness values categorized by severity.",
    )

    insights: List = []
    strong_skew = skew_df[skew_df["Skew Category"] == "Strong"]
    if not strong_skew.empty:
        insights.append(
            insight(
                "warning",
                "Columns with strong skewness: " + ", ".join(strong_skew["Column"].tolist()),
            )
        )
    else:
        insights.append(insight("success", "No strongly skewed columns detected."))

    return AnalysisResult(
        analysis_id="skewness_analysis",
        title="Skewness Analysis",
        summary="Identifies skewness levels across numeric columns.",
        tables=[table],
        insights=insights,
    )


def skewness_statistics(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="skewness_statistics",
            title="Skewness Statistics",
            summary="No numeric columns available to compute skewness statistics.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to compute skewness statistics.")],
        )

    stats_df = pd.DataFrame(
        {
            "Column": numeric_cols,
            "Skewness": df[numeric_cols].skew().values,
            "Kurtosis": df[numeric_cols].kurtosis().values,
        }
    )

    table = dataframe_to_table(
        stats_df,
        title="Skewness and Kurtosis",
        description="Higher kurtosis indicates heavier tails compared to a normal distribution.",
    )

    insights = [
        insight(
            "info",
            "Skewness outside ±1 suggests strong asymmetry; kurtosis above 3 indicates heavy tails.",
        )
    ]

    return AnalysisResult(
        analysis_id="skewness_statistics",
        title="Skewness & Kurtosis",
        summary="Statistical measures of asymmetry and tail behavior.",
        tables=[table],
        insights=insights,
    )


def skewness_visualization(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="skewness_visualization",
            title="Skewness Visualization",
            summary="No numeric columns available to visualize skewness.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to visualize skewness.")],
        )

    charts: List[AnalysisChart] = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#5A67D8")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        charts.append(
            AnalysisChart(
                title=f"Distribution: {col}",
                image=fig_to_base64(fig),
                description="Histogram with KDE highlighting skewness.",
            )
        )

    insights = [
        insight("info", "Visualize the skewness of each numeric column to assess transformations."),
    ]

    return AnalysisResult(
        analysis_id="skewness_visualization",
        title="Skewness Visualization",
        summary="Histograms with density curves for each numeric column.",
        charts=charts,
        insights=insights,
    )


def normality_test(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = [col for col in numeric_columns(df) if df[col].dropna().size >= 8]
    if not numeric_cols:
        return AnalysisResult(
            analysis_id="normality_test",
            title="Normality Test",
            summary="Insufficient numeric data to perform normality tests.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Ensure numeric columns contain at least 8 non-null observations to run statistical tests.",
                )
            ],
        )

    rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 8:
            continue

        entry: Dict[str, Any] = {"Column": col, "Sample Size": len(series)}

        if len(series) <= 5000:
            try:
                _, p_value = stats.shapiro(series)
                entry["Shapiro p"] = float(p_value)
            except Exception:
                entry["Shapiro p"] = None
        else:
            entry["Shapiro p"] = None

        if len(series) >= 20:
            try:
                _, p_value = stats.normaltest(series)
                entry["D'Agostino p"] = float(p_value)
            except Exception:
                entry["D'Agostino p"] = None
        else:
            entry["D'Agostino p"] = None

        try:
            standardized = (series - series.mean()) / series.std(ddof=0)
            _, p_value = stats.kstest(standardized, "norm")
            entry["KS p"] = float(p_value)
        except Exception:
            entry["KS p"] = None

        valid_tests = [entry[key] for key in ["Shapiro p", "D'Agostino p", "KS p"] if entry[key] is not None]
        if valid_tests:
            above_threshold = sum(p > 0.05 for p in valid_tests)
            ratio = above_threshold / len(valid_tests)
            if ratio >= 0.7:
                entry["Consensus"] = "Likely Normal"
            elif ratio >= 0.3:
                entry["Consensus"] = "Mixed"
            else:
                entry["Consensus"] = "Not Normal"
        else:
            entry["Consensus"] = "Inconclusive"

        rows.append(entry)

    result_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        result_df,
        title="Normality Test Results",
        description="p-values for multiple normality tests (higher is closer to normal).",
        round_decimals=4,
    )

    charts: List[AnalysisChart] = []
    for col in result_df.sort_values("Sample Size", ascending=False)["Column"].tolist():
        fig, ax = plt.subplots(figsize=(6, 4))
        stats.probplot(df[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {col}")
        charts.append(
            AnalysisChart(
                title=f"Q-Q Plot: {col}",
                image=fig_to_base64(fig),
                description="Quantile-quantile plot against theoretical normal distribution.",
            )
        )

    insights: List = []
    if (result_df["Consensus"] == "Not Normal").any():
        cols = result_df[result_df["Consensus"] == "Not Normal"]["Column"].tolist()
        insights.append(insight("warning", "Non-normal columns detected: " + ", ".join(cols[:10])))
    if (result_df["Consensus"] == "Likely Normal").any():
        cols = result_df[result_df["Consensus"] == "Likely Normal"]["Column"].tolist()
        insights.append(insight("success", "Columns likely normal: " + ", ".join(cols[:10])))

    return AnalysisResult(
        analysis_id="normality_test",
        title="Normality Test",
        summary="Statistical assessment of normality across numeric columns.",
        tables=[table],
        charts=charts,
        insights=insights,
    )


def distribution_plots(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return _generic_distribution_plots(df, "hist")


def histogram_plots(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return _generic_distribution_plots(df, "hist")


def box_plots(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return _generic_distribution_plots(df, "box")


def violin_plots(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return _generic_distribution_plots(df, "violin")


def kde_plots(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return _generic_distribution_plots(df, "kde")


def _generic_distribution_plots(df: pd.DataFrame, chart_type: str) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if not numeric_cols:
        analysis_id = f"{chart_type}_plots" if chart_type != "hist" else "distribution_plots"
        title = f"{chart_type.title()} Plots" if chart_type != "hist" else "Distribution Plots"
        return AnalysisResult(
            analysis_id=analysis_id,
            title=title,
            summary="No numeric columns available for distribution visualization.",
            status="warning",
            insights=[insight("warning", "Select numeric columns to visualize distributions.")],
        )

    charts: List[AnalysisChart] = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = df[col].dropna()
        if chart_type == "hist":
            sns.histplot(data, kde=True, ax=ax, color="#2B6CB0")
            ax.set_title(f"Histogram: {col}")
        elif chart_type == "box":
            sns.boxplot(x=data, ax=ax, color="#38A169")
            ax.set_title(f"Box Plot: {col}")
        elif chart_type == "violin":
            sns.violinplot(x=data, ax=ax, color="#805AD5")
            ax.set_title(f"Violin Plot: {col}")
        elif chart_type == "kde":
            sns.kdeplot(data, ax=ax, fill=True, color="#D69E2E")
            ax.set_title(f"KDE Plot: {col}")
        else:
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        charts.append(
            AnalysisChart(
                title=ax.get_title(),
                image=fig_to_base64(fig),
                description="Distribution visualization for numeric column.",
            )
        )

    summary_map = {
        "hist": "Histograms with density overlay for numeric variables.",
        "box": "Box plots highlighting medians and potential outliers.",
        "violin": "Violin plots showing distribution density.",
        "kde": "Kernel density estimates for smooth distribution visualization.",
    }
    title_map = {
        "hist": "Distribution Plots",
        "box": "Box Plots",
        "violin": "Violin Plots",
        "kde": "KDE Plots",
    }

    analysis_id = f"{chart_type}_plots" if chart_type != "hist" else "distribution_plots"

    return AnalysisResult(
        analysis_id=analysis_id,
        title=title_map.get(chart_type, "Distribution Plots"),
        summary=summary_map.get(chart_type, "Distribution visualizations for numeric columns."),
        charts=charts,
        insights=[insight("info", "Visualize distributions to spot skewness and outliers.")],
    )


__all__ = [
    "numeric_frequency_analysis",
    "summary_statistics",
    "skewness_analysis",
    "skewness_statistics",
    "skewness_visualization",
    "normality_test",
    "distribution_plots",
    "histogram_plots",
    "box_plots",
    "violin_plots",
    "kde_plots",
]

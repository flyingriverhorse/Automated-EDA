from __future__ import annotations

"""Categorical-focused runtime analyses."""

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    categorical_columns,
    dataframe_to_table,
    fig_to_base64,
    insight,
    metric,
)


def categorical_frequency_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if not categorical_cols:
        return AnalysisResult(
            analysis_id="categorical_frequency_analysis",
            title="Categorical Frequency Analysis",
            summary="No categorical columns found for analysis.",
            status="warning",
            insights=[insight("warning", "Select categorical columns to analyze frequencies.")],
        )

    rows = []
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        for category, count in counts.items():
            rows.append({"Column": col, "Category": str(category), "Count": int(count)})

    freq_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        freq_df,
        title="Category Frequencies",
        description="Frequency counts for every category across categorical columns.",
    )

    return AnalysisResult(
        analysis_id="categorical_frequency_analysis",
        title="Categorical Frequency Analysis",
        summary="Frequency counts for categorical variables.",
        tables=[table],
        insights=[insight("info", "Review distribution breadth before encoding categories." )],
    )


def categorical_visualization(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return categorical_bar_charts(df, context)


def categorical_bar_charts(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if not categorical_cols:
        return AnalysisResult(
            analysis_id="categorical_bar_charts",
            title="Categorical Bar Charts",
            summary="No categorical columns available for visualization.",
            status="warning",
            insights=[insight("warning", "Select categorical columns to visualize frequencies.")],
        )

    charts: List[AnalysisChart] = []
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        category_labels = counts.index.astype(str)
        fig, ax = plt.subplots(figsize=(7, max(4, len(counts) * 0.4)))
        sns.barplot(
            x=counts.values,
            y=category_labels,
            hue=category_labels,
            palette="Blues_d",
            dodge=False,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(f"Category Counts: {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        charts.append(
            AnalysisChart(
                title=f"Category Counts: {col}",
                image=fig_to_base64(fig),
                description="Bar chart of category counts including all available values.",
            )
        )

    return AnalysisResult(
        analysis_id="categorical_bar_charts",
        title="Categorical Bar Charts",
        summary="Bar charts for categorical columns including every category.",
        charts=charts,
        insights=[insight("info", "Use bar charts to assess dominant and rare categories." )],
    )


def categorical_pie_charts(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if not categorical_cols:
        return AnalysisResult(
            analysis_id="categorical_pie_charts",
            title="Categorical Pie Charts",
            summary="No categorical columns available for visualization.",
            status="warning",
            insights=[insight("warning", "Select categorical columns to visualize proportions.")],
        )

    charts: List[AnalysisChart] = []
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=140,
            normalize=True,
        )
        ax.set_title(f"Category Share: {col}")
        charts.append(
            AnalysisChart(
                title=f"Category Share: {col}",
                image=fig_to_base64(fig),
                description="Pie chart showing proportion of every category.",
            )
        )

    return AnalysisResult(
        analysis_id="categorical_pie_charts",
        title="Categorical Pie Charts",
        summary="Visual representation of categorical proportions for each column.",
        charts=charts,
        insights=[insight("info", "Pie charts highlight proportional makeup of categories." )],
    )


def categorical_cardinality_profile(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if not categorical_cols:
        return AnalysisResult(
            analysis_id="categorical_cardinality_profile",
            title="Categorical Cardinality Profile",
            summary="No categorical columns available to evaluate cardinality.",
            status="warning",
            insights=[
                insight("warning", "Select categorical columns to inspect cardinality saturation."),
            ],
        )

    rows = []
    high_cardinality: List[str] = []
    dominant_columns: List[str] = []
    cumulative_cardinality = 0.0
    total_unique = 0

    for col in categorical_cols:
        series = df[col]
        total_count = int(series.size)
        missing_count = int(series.isna().sum())
        valid_count = total_count - missing_count
        unique_count = int(series.dropna().nunique()) if valid_count else 0
        total_unique += unique_count

        cardinality_pct = (unique_count / valid_count * 100) if valid_count else 0.0
        cumulative_cardinality += cardinality_pct

        most_common_value = "—"
        most_common_pct = 0.0
        if valid_count:
            counts = series.dropna().value_counts()
            if not counts.empty:
                raw_value = counts.index[0]
                most_common_value = str(raw_value)
                if len(most_common_value) > 40:
                    most_common_value = most_common_value[:37] + "…"
                most_common_pct = counts.iloc[0] / valid_count * 100

        if cardinality_pct >= 60:
            high_cardinality.append(col)
        if most_common_pct >= 80 and unique_count > 1:
            dominant_columns.append(col)

        rows.append(
            {
                "Column": col,
                "Unique Categories": unique_count,
                "Cardinality %": round(cardinality_pct, 2),
                "Most Common": most_common_value,
                "Most Common %": round(most_common_pct, 2),
                "Missing %": round((missing_count / total_count * 100) if total_count else 0.0, 2),
            }
        )

    summary_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        summary_df,
        title="Categorical Cardinality Overview",
        description="Unique category counts, saturation ratio, and dominance for every categorical column.",
    )

    average_cardinality = round(cumulative_cardinality / len(categorical_cols), 2) if categorical_cols else 0.0
    metrics = [
        metric("Categorical Columns", len(categorical_cols)),
        metric("Average Cardinality %", average_cardinality, unit="%"),
        metric("Total Unique Categories", total_unique),
    ]

    insights = []
    if high_cardinality:
        sample = ", ".join(high_cardinality[:5])
        more = "…" if len(high_cardinality) > 5 else ""
        insights.append(
            insight(
                "warning",
                f"High-cardinality columns (≥60% unique): {sample}{more}. Consider hashing or target encoding.",
            )
        )
    if dominant_columns:
        sample = ", ".join(dominant_columns[:5])
        more = "…" if len(dominant_columns) > 5 else ""
        insights.append(
            insight(
                "info",
                f"Columns dominated by a single category (≥80% share): {sample}{more}. One-hot encoding may create sparse vectors.",
            )
        )
    if not insights:
        insights.append(
            insight("success", "Cardinality looks balanced across scanned categorical columns."),
        )

    return AnalysisResult(
        analysis_id="categorical_cardinality_profile",
        title="Categorical Cardinality Profile",
        summary="Unique category saturation and dominance signals for categorical variables.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


def rare_category_detection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if not categorical_cols:
        return AnalysisResult(
            analysis_id="rare_category_detection",
            title="Rare Category Detection",
            summary="No categorical columns available to inspect rare categories.",
            status="warning",
            insights=[
                insight("warning", "Select categorical columns to flag infrequent categories."),
            ],
        )

    metadata = context.metadata or {}
    try:
        threshold_pct = float(metadata.get("rare_category_threshold_pct", 5.0))
    except (TypeError, ValueError):
        threshold_pct = 5.0
    try:
        min_count = int(metadata.get("rare_category_min_count", 10))
    except (TypeError, ValueError):
        min_count = 10

    rows = []
    flagged_columns: List[str] = []
    MAX_ROWS = 500

    for col in categorical_cols:
        series = df[col].dropna()
        total_count = int(series.size)
        if total_count == 0:
            continue

        counts = series.value_counts()
        for category, count in counts.items():
            percent = count / total_count * 100
            if percent <= threshold_pct or count <= min_count:
                recommendation_parts = []
                if percent <= threshold_pct:
                    recommendation_parts.append("group with 'Other'")
                if count <= min_count:
                    recommendation_parts.append("switch to target/frequency encoding")
                recommendation = "; ".join(recommendation_parts) if recommendation_parts else "Review encoding"

                rows.append(
                    {
                        "Column": col,
                        "Category": str(category),
                        "Count": int(count),
                        "Percent": round(percent, 2),
                        "Recommendation": recommendation,
                    }
                )
                if col not in flagged_columns:
                    flagged_columns.append(col)

                if len(rows) >= MAX_ROWS:
                    break
        if len(rows) >= MAX_ROWS:
            break

    rows.sort(key=lambda item: (item["Percent"], item["Count"]))

    if not rows:
        return AnalysisResult(
            analysis_id="rare_category_detection",
            title="Rare Category Detection",
            summary="No categories met the rare threshold criteria.",
            insights=[
                insight(
                    "success",
                    "No categories fell below the configured rare thresholds. Standard encoding should be safe.",
                )
            ],
        )

    table = dataframe_to_table(
        pd.DataFrame(rows),
        title="Rare Categories",
        description=(
            f"Categories flagged because they account for ≤ {threshold_pct:.1f}% of a column or ≤ {min_count} rows."
        ),
    )

    insights = [
        insight(
            "warning",
            f"{len(rows)} rare categories identified across {len(flagged_columns)} columns. Consider regrouping before encoding.",
        )
    ]
    if len(rows) >= MAX_ROWS:
        insights.append(
            insight(
                "info",
                "Only the first 500 rare categories are shown. Export results for a full review if needed.",
            )
        )

    metrics = [
        metric("Columns Scanned", len(categorical_cols)),
        metric("Rare Category Threshold", f"≤{threshold_pct:.1f}% or ≤{min_count} rows"),
        metric("Categories Flagged", len(rows)),
    ]

    return AnalysisResult(
        analysis_id="rare_category_detection",
        title="Rare Category Detection",
        summary="Flags infrequent categories that may require regrouping or alternative encoding strategies.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


__all__ = [
    "categorical_frequency_analysis",
    "categorical_visualization",
    "categorical_bar_charts",
    "categorical_pie_charts",
    "categorical_cardinality_profile",
    "rare_category_detection",
]

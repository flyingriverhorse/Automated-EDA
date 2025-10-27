from __future__ import annotations

"""Bivariate and multivariate runtime analyses."""

from itertools import combinations
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisMetric,
    AnalysisResult,
    AnalysisTable,
    categorical_columns,
    dataframe_to_table,
    fig_to_base64,
    insight,
    numeric_columns,
)


def correlation_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    return pearson_correlation(df, context)


def pearson_correlation(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="pearson_correlation",
            title="Pearson Correlation",
            summary="At least two numeric columns are required to compute correlations.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns for correlation analysis.")],
        )

    corr = df[numeric_cols].corr(method="pearson")
    table = dataframe_to_table(
        corr.reset_index().rename(columns={"index": "Column"}),
        title="Pearson Correlation Matrix",
        description="Pairwise correlation coefficients between numeric columns.",
        round_decimals=3,
    )

    fig, ax = plt.subplots(figsize=(min(12, 2 + len(numeric_cols)), min(10, 2 + len(numeric_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    chart = AnalysisChart(
        title="Correlation Heatmap",
        image=fig_to_base64(fig),
        description="Heatmap highlighting stronger positive/negative correlations.",
    )

    insights: List = []
    high_corr = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .abs()
        .stack()
        .reset_index()
        .rename(columns={0: "Correlation"})
    )
    strong_pairs = high_corr[high_corr["Correlation"] > 0.8].sort_values("Correlation", ascending=False)
    if not strong_pairs.empty:
        insights.append(
            insight(
                "warning",
                "Highly correlated pairs (>|0.8|): "
                + ", ".join(
                    f"{row['level_0']} & {row['level_1']} ({row['Correlation']:.2f})"
                    for _, row in strong_pairs.iterrows()
                ),
            )
        )
    else:
        insights.append(insight("info", "No strong linear correlations detected above |0.8|."))

    return AnalysisResult(
        analysis_id="pearson_correlation",
        title="Pearson Correlation",
        summary="Linear correlation analysis between numeric variables.",
        tables=[table],
        charts=[chart],
        insights=insights,
    )


def spearman_correlation(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="spearman_correlation",
            title="Spearman Correlation",
            summary="At least two numeric columns are required to compute correlations.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns for correlation analysis.")],
        )

    corr = df[numeric_cols].corr(method="spearman")
    table = dataframe_to_table(
        corr.reset_index().rename(columns={"index": "Column"}),
        title="Spearman Correlation Matrix",
        description="Rank-based correlation coefficients between numeric columns.",
        round_decimals=3,
    )

    fig, ax = plt.subplots(figsize=(min(12, 2 + len(numeric_cols)), min(10, 2 + len(numeric_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", center=0, ax=ax)
    ax.set_title("Spearman Correlation Heatmap")
    chart = AnalysisChart(
        title="Spearman Heatmap",
        image=fig_to_base64(fig),
        description="Heatmap showing monotonic relationships irrespective of linearity.",
    )

    return AnalysisResult(
        analysis_id="spearman_correlation",
        title="Spearman Correlation",
        summary="Rank-based correlation analysis (Spearman).",
        tables=[table],
        charts=[chart],
        insights=[
            insight(
                "info",
                "Spearman correlation is robust to non-linear but monotonic relationships.",
            )
        ],
    )


def scatter_plot_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    numeric_cols = numeric_columns(df)
    if len(numeric_cols) < 2:
        return AnalysisResult(
            analysis_id="scatter_plot_analysis",
            title="Scatter Plot Analysis",
            summary="At least two numeric columns are required to create scatter plots.",
            status="warning",
            insights=[insight("warning", "Select multiple numeric columns for scatter plot visualization.")],
        )

    charts: List[AnalysisChart] = []
    pair_limit = 15
    for idx, (x_col, y_col) in enumerate(combinations(numeric_cols, 2)):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#DD6B20", alpha=0.7)
        ax.set_title(f"Scatter: {x_col} vs {y_col}")
        charts.append(
            AnalysisChart(
                title=f"Scatter: {x_col} vs {y_col}",
                image=fig_to_base64(fig),
                description="Scatter plot to explore pairwise relationships.",
            )
        )
        if idx + 1 >= pair_limit:
            break

    notes = "Generated scatter plots for all numeric column pairs." if len(charts) < pair_limit else "Generated scatter plots for first 15 numeric pairs."

    return AnalysisResult(
        analysis_id="scatter_plot_analysis",
        title="Scatter Plot Analysis",
        summary="Visual exploration of numeric relationships using scatter plots.",
        charts=charts,
        insights=[insight("info", notes)],
    )


def cross_tabulation_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    if len(categorical_cols) < 2:
        return AnalysisResult(
            analysis_id="cross_tabulation_analysis",
            title="Cross Tabulation Analysis",
            summary="At least two categorical columns are required for cross tabulation.",
            status="warning",
            insights=[insight("warning", "Select multiple categorical columns to generate cross tabulations.")],
        )

    try:
        col_a, col_b = categorical_cols[:2]
        clean_pairs = df[[col_a, col_b]].dropna()

        if clean_pairs.empty:
            return AnalysisResult(
                analysis_id="cross_tabulation_analysis",
                title="Cross Tabulation Analysis",
                summary="Selected columns contain no valid data for cross tabulation.",
                status="warning",
                insights=[insight("warning", f"Columns {col_a} and {col_b} contain insufficient valid data.")],
            )

        col_a_series = clean_pairs[col_a].astype(str)
        col_b_series = clean_pairs[col_b].astype(str)

        try:
            crosstab_counts = pd.crosstab(index=col_a_series, columns=col_b_series, dropna=False)
        except Exception:
            grouped_counts = clean_pairs.groupby([col_a, col_b]).size()
            crosstab_counts = grouped_counts.unstack(fill_value=0)
            crosstab_counts.index = crosstab_counts.index.astype(str)
            crosstab_counts.columns = crosstab_counts.columns.astype(str)

        if crosstab_counts.empty or crosstab_counts.sum().sum() == 0:
            return AnalysisResult(
                analysis_id="cross_tabulation_analysis",
                title="Cross Tabulation Analysis",
                summary="No valid combinations found for cross tabulation.",
                status="warning",
                insights=[insight("warning", f"No valid data combinations between {col_a} and {col_b}.")],
            )

        if crosstab_counts.shape[0] < 1 or crosstab_counts.shape[1] < 1:
            return AnalysisResult(
                analysis_id="cross_tabulation_analysis",
                title="Cross Tabulation Analysis",
                summary="Cross tabulation requires at least one category in each dimension.",
                status="warning",
                insights=[insight("warning", f"Columns {col_a} and {col_b} need at least one category each for cross tabulation.")],
            )

        row_sums = crosstab_counts.sum(axis=1).replace(0, np.nan)
        crosstab_normalized = crosstab_counts.div(row_sums, axis=0).fillna(0) * 100
        crosstab_normalized.index.name = str(col_a)
        crosstab_normalized.columns.name = str(col_b)

        base_table = dataframe_to_table(
            crosstab_normalized.reset_index().rename(columns={col_a: "Category"}),
            title=f"Cross Tabulation: {col_a} vs {col_b}",
            description="Row-normalized cross tabulation expressed as percentages.",
            round_decimals=1,
        )

        tables: List[AnalysisTable] = [base_table]
        charts: List[AnalysisChart] = []

        # Heatmap visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(crosstab_normalized, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
        ax.set_title(f"Cross Tab Heatmap: {col_a} vs {col_b}")
        charts.append(
            AnalysisChart(
                title="Cross Tab Heatmap",
                image=fig_to_base64(fig),
                description="Heatmap of row-normalized cross tabulation percentages.",
            )
        )

        # Stacked bar visualization
        stack_data = crosstab_counts.div(crosstab_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
        bottom = np.zeros(len(stack_data))
        categories = stack_data.index.astype(str)
        for column in stack_data.columns:
            values = stack_data[column].values
            ax_bar.bar(categories, values, bottom=bottom, label=str(column))
            bottom += values
        ax_bar.set_title(f"Stacked Bar: {col_a} vs {col_b}")
        ax_bar.set_ylabel("Percentage (%)")
        ax_bar.set_xlabel(col_a)
        ax_bar.legend(title=col_b, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")
        charts.append(
            AnalysisChart(
                title="Stacked Bar",
                image=fig_to_base64(fig_bar),
                description="Stacked percentage bar chart comparing category combinations.",
            )
        )

        # Chi-square test for association
        chi_square_details: Dict[str, Any] = {}
        insights: List = [insight("info", "Use cross tabs to inspect segment distributions.")]

        if crosstab_counts.shape[0] >= 2 and crosstab_counts.shape[1] >= 2:
            try:
                chi2, p_value, dof, expected = chi2_contingency(crosstab_counts)
                n = crosstab_counts.values.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(crosstab_counts.shape) - 1))) if n > 0 else 0.0
                chi_table = dataframe_to_table(
                    pd.DataFrame(
                        [
                            {
                                "Chi-square": chi2,
                                "p-value": p_value,
                                "Degrees of Freedom": dof,
                                "Cramér's V": cramers_v,
                                "Samples": int(n),
                            }
                        ]
                    ),
                    title="Chi-Square Test Summary",
                    description="Association test between categorical variables.",
                    round_decimals=4,
                )

                tables.append(chi_table)

                strength = "None"
                level = "info"
                if p_value < 0.001:
                    strength, level = "Very Strong", "danger"
                elif p_value < 0.01:
                    strength, level = "Strong", "warning"
                elif p_value < 0.05:
                    strength, level = "Moderate", "warning"
                elif p_value < 0.1:
                    strength, level = "Weak", "info"

                insights.append(
                    insight(
                        level,
                        f"Chi-square test indicates {strength.lower()} association between '{col_a}' and '{col_b}' (χ²={chi2:.3f}, p={p_value:.4f}).",
                    )
                )

                expected_df = pd.DataFrame(expected, index=crosstab_counts.index, columns=crosstab_counts.columns)
                chi_square_details = {
                    "chi_square": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                    "cramers_v": cramers_v,
                    "expected_frequencies": expected_df.reset_index().rename(columns={"index": col_a}).to_dict(orient="records"),
                }
            except Exception:
                chi_square_details = {"error": "Chi-square test could not be computed."}
        else:
            chi_square_details = {"note": "Insufficient category combinations to compute chi-square test."}

        return AnalysisResult(
            analysis_id="cross_tabulation_analysis",
            title="Cross Tabulation Analysis",
            summary="Explores interactions between categorical features with statistical association testing.",
            tables=tables,
            charts=charts,
            insights=insights,
            details={"chi_square": chi_square_details},
        )

    except Exception as exc:
        return AnalysisResult(
            analysis_id="cross_tabulation_analysis",
            title="Cross Tabulation Analysis",
            summary="Failed to generate cross tabulation.",
            status="error",
            insights=[insight("danger", f"Error generating cross tabulation: {str(exc)}")],
        )


def categorical_numeric_relationships(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    categorical_cols = categorical_columns(df)
    numeric_cols = numeric_columns(df)

    if not categorical_cols or not numeric_cols:
        return AnalysisResult(
            analysis_id="categorical_numeric_relationships",
            title="Categorical vs Numeric Relationships",
            summary="Requires at least one categorical and one numeric column.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select both categorical and numeric columns to explore grouped distributions.",
                )
            ],
        )

    metadata = context.metadata or {}

    def _lookup_override(container: Dict[str, Any], *keys: str) -> Dict[str, Any]:
        node: Any = container
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return {}
        return node if isinstance(node, dict) else {}

    categorical_numeric_override: Dict[str, Any] = {}
    candidates = [
        _lookup_override(metadata, "client_overrides", "analysis", "categorical_numeric"),
        _lookup_override(metadata, "client_overrides", "categorical_numeric"),
        _lookup_override(metadata, "analysis", "categorical_numeric"),
        metadata.get("categorical_numeric") if isinstance(metadata.get("categorical_numeric"), dict) else {},
    ]
    for candidate in candidates:
        if candidate:
            categorical_numeric_override = candidate
            break

    requested_categorical = set()
    requested_numeric = set()
    requested_pairs: List[Dict[str, Any]] = []
    category_cap_override: int | None = None

    if categorical_numeric_override:
        cat_values = categorical_numeric_override.get("categorical_columns")
        num_values = categorical_numeric_override.get("numeric_columns")
        if isinstance(cat_values, list):
            requested_categorical = {str(value) for value in cat_values if value is not None}
        if isinstance(num_values, list):
            requested_numeric = {str(value) for value in num_values if value is not None}
        pairs_value = categorical_numeric_override.get("pairs")
        if isinstance(pairs_value, list):
            requested_pairs = [pair for pair in pairs_value if isinstance(pair, dict)]
        cap_value = categorical_numeric_override.get("category_cap")
        if isinstance(cap_value, int) and cap_value > 1:
            category_cap_override = cap_value

    if requested_categorical:
        categorical_cols = [col for col in categorical_cols if col in requested_categorical]
    if requested_numeric:
        numeric_cols = [col for col in numeric_cols if col in requested_numeric]

    pair_stats: List[Dict[str, Any]] = []
    group_statistics_frames: List[pd.DataFrame] = []
    analyzed_pairs: List[Dict[str, Any]] = []
    category_truncations: List[Dict[str, Any]] = []
    charts: List[AnalysisChart] = []
    per_pair_tables: List[AnalysisTable] = []
    per_pair_group_details: Dict[str, List[Dict[str, Any]]] = {}

    if requested_pairs:
        pair_queue = []
        for pair in requested_pairs:
            cat_col = str(pair.get("categorical") or pair.get("categorical_column") or "")
            num_col = str(pair.get("numeric") or pair.get("numeric_column") or "")
            if cat_col and num_col:
                pair_queue.append((cat_col, num_col))
    else:
        pair_queue = [(cat, num) for cat in categorical_cols for num in numeric_cols]

    seen_pairs = set()

    for cat_col, num_col in pair_queue:
        if (cat_col, num_col) in seen_pairs:
            continue
        seen_pairs.add((cat_col, num_col))

        if cat_col not in df.columns or num_col not in df.columns:
            continue

        try:
            pair_df = df[[cat_col, num_col]].dropna()
        except KeyError:
            continue

        if pair_df.empty or pair_df[num_col].nunique() <= 1:
            continue

        pair_df = pair_df.copy()
        pair_df[cat_col] = pair_df[cat_col].astype(str)

        unique_categories = pair_df[cat_col].nunique()
        if unique_categories < 2:
            continue

        category_cap = category_cap_override or 30
        truncated_categories: List[str] = []
        if unique_categories > category_cap:
            value_counts = pair_df[cat_col].value_counts()
            top_categories = value_counts.nlargest(category_cap).index
            truncated_categories = [str(item) for item in value_counts.index if item not in top_categories]
            pair_df = pair_df[pair_df[cat_col].isin(top_categories)].copy()
            if pair_df.empty or pair_df[num_col].nunique() <= 1:
                continue

        grouped = (
            pair_df.groupby(cat_col)[num_col]
            .agg(["mean", "median", "std", "count", "min", "max"])
            .reset_index()
        )

        grouped.rename(
            columns={
                cat_col: "Category",
                "mean": "Mean",
                "median": "Median",
                "std": "Std",
                "count": "Count",
                "min": "Min",
                "max": "Max",
            },
            inplace=True,
        )

        grouped.insert(0, "Categorical Column", cat_col)
        grouped.insert(2, "Numeric Column", num_col)
        group_statistics_frames.append(grouped.copy())

        mean_range = grouped["Mean"].max() - grouped["Mean"].min()
        pair_info: Dict[str, Any] = {
            "categorical_column": cat_col,
            "numeric_column": num_col,
            "categories_analyzed": int(grouped.shape[0]),
            "mean_range": float(mean_range),
            "min_mean": float(grouped["Mean"].min()),
            "max_mean": float(grouped["Mean"].max()),
            "samples": int(pair_df.shape[0]),
        }

        analyzed_pairs.append(
            {
                "categorical_column": cat_col,
                "numeric_column": num_col,
                "categories_analyzed": int(grouped.shape[0]),
                "samples": int(pair_df.shape[0]),
                "trimmed_categories": truncated_categories,
            }
        )

        if truncated_categories:
            category_truncations.append(
                {
                    "categorical_column": cat_col,
                    "numeric_column": num_col,
                    "dropped_categories": truncated_categories,
                    "retained_category_limit": category_cap_override or 30,
                }
            )

        group_values = [vals[num_col].values for _, vals in pair_df.groupby(cat_col) if len(vals) > 1]
        if len(group_values) >= 2:
            try:
                anova_stat, anova_p = f_oneway(*group_values)
                pair_info["anova_statistic"] = float(anova_stat)
                pair_info["anova_p_value"] = float(anova_p)
            except Exception:
                pair_info["anova_p_value"] = None

        pair_stats.append(pair_info)

        per_pair_tables.append(
            dataframe_to_table(
                grouped,
                title=f"Group Statistics • {num_col} by {cat_col}",
                description="Per-category summary statistics for the selected categorical and numeric columns.",
                round_decimals=3,
                max_rows=category_cap_override or 30,
            )
        )
        per_pair_group_details[f"{cat_col}::{num_col}"] = grouped.to_dict(orient="records")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=pair_df, x=cat_col, y=num_col, hue=cat_col, ax=axes[0], dodge=False, palette="Set2")
        axes[0].set_title(f"Boxplot: {num_col} by {cat_col}")
        axes[0].set_xlabel(cat_col)
        axes[0].set_ylabel(num_col)
        plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
        legend = axes[0].get_legend()
        if legend:
            legend.remove()

        try:
            sns.violinplot(
                data=pair_df,
                x=cat_col,
                y=num_col,
                hue=cat_col,
                ax=axes[1],
                dodge=False,
                palette="muted",
                cut=0,
            )
            axes[1].set_title(f"Violin: {num_col} by {cat_col}")
            axes[1].set_xlabel(cat_col)
            axes[1].set_ylabel(num_col)
            plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
            legend = axes[1].get_legend()
            if legend:
                legend.remove()
        except Exception:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "Violin plot unavailable", ha="center", va="center")

        charts.append(
            AnalysisChart(
                title=f"{num_col} by {cat_col}",
                image=fig_to_base64(fig),
                description="Box and violin plots comparing numeric distributions across categories.",
            )
        )

    if not pair_stats:
        return AnalysisResult(
            analysis_id="categorical_numeric_relationships",
            title="Categorical vs Numeric Relationships",
            summary="No suitable categorical/numeric pairs available for analysis.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Need at least one categorical column with manageable cardinality and a numeric column with variation.",
                )
            ],
        )

    pair_df = pd.DataFrame(pair_stats)
    pair_table = dataframe_to_table(
        pair_df,
        title="Pair Summary",
        description="Per-pair overview including category coverage and mean span.",
        round_decimals=3,
    )

    group_stats_table: AnalysisTable | None = None
    if group_statistics_frames:
        combined_group_stats = pd.concat(group_statistics_frames, ignore_index=True)
        group_stats_table = dataframe_to_table(
            combined_group_stats,
            title="Group Statistics",
            description="Category-level statistics for every analyzed categorical and numeric pair.",
            round_decimals=3,
        )

    metrics = [
        AnalysisMetric(label="Pairs Analyzed", value=len(pair_stats)),
        AnalysisMetric(label="Charts Generated", value=len(charts)),
    ]

    if pair_stats:
        metrics.append(
            AnalysisMetric(
                label="Total Categories Evaluated",
                value=int(sum(item.get("categories_analyzed", 0) for item in pair_stats)),
            )
        )

    insights: List = []
    top_spreads = pair_df.sort_values("mean_range", ascending=False).head(3)
    for _, row in top_spreads.iterrows():
        insights.append(
            insight(
                "info",
                f"{row['numeric_column']} spans {row['mean_range']:.2f} across categories of {row['categorical_column']} (means {row['min_mean']:.2f}–{row['max_mean']:.2f}).",
            )
        )

    if "anova_p_value" in pair_df.columns:
        significant = pair_df.dropna(subset=["anova_p_value"])
        significant = significant[significant["anova_p_value"] < 0.05]
        if not significant.empty:
            best = significant.sort_values("anova_p_value").iloc[0]
            insights.append(
                insight(
                    "warning",
                    f"ANOVA indicates significant mean differences for {best['numeric_column']} by {best['categorical_column']} (p={best['anova_p_value']:.4f}).",
                )
            )
        else:
            insights.append(
                insight("info", "No ANOVA p-values below 0.05 were detected across analyzed pairs."),
            )
    else:
        insights.append(
            insight("info", "Insufficient samples for ANOVA significance tests."),
        )

    tables: List[AnalysisTable] = [pair_table]
    if group_stats_table is not None:
        tables.append(group_stats_table)
    tables.extend(per_pair_tables)

    return AnalysisResult(
        analysis_id="categorical_numeric_relationships",
        title="Categorical vs Numeric Relationships",
        summary="Visualizes numeric distributions across categorical segments with group statistics and ANOVA screening.",
        metrics=metrics,
        tables=tables,
        charts=charts,
        insights=insights,
        details={
            "pair_statistics": pair_stats,
            "analyzed_pairs": analyzed_pairs,
            "category_truncations": category_truncations,
            "group_statistics_by_pair": per_pair_group_details,
        },
    )


__all__ = [
    "correlation_analysis",
    "pearson_correlation",
    "spearman_correlation",
    "scatter_plot_analysis",
    "cross_tabulation_analysis",
    "categorical_numeric_relationships",
]

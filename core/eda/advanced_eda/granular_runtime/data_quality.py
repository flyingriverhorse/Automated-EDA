from __future__ import annotations

"""Data quality focused runtime analyses."""

from typing import Dict, List

import pandas as pd

from .common import (
    AnalysisContext,
    AnalysisResult,
    AnalysisTable,
    categorical_columns,
    dataframe_to_table,
    datetime_columns,
    insight,
    metric,
    numeric_columns,
)


def dataset_shape_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    """Overview of dataset dimensions and column metadata."""

    num_rows, num_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum()

    metrics = [
        metric("Rows", num_rows),
        metric("Columns", num_cols),
        metric("Memory", f"{memory_usage / (1024 ** 2):.2f}", unit="MB"),
    ]

    table = dataframe_to_table(
        pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
            }
        ),
        title="Column Overview",
        description="Summary of column data types and missing values.",
    )

    insights: List = []
    if num_rows == 0:
        insights.append(insight("warning", "Dataset is empty."))
    if num_cols == 0:
        insights.append(insight("warning", "Dataset has no columns."))
    if not insights:
        insights.append(insight("info", "Dataset size and structure captured successfully."))

    return AnalysisResult(
        analysis_id="dataset_shape_analysis",
        title="Dataset Shape Analysis",
        summary="Overview of dataset dimensions and column metadata.",
        metrics=metrics,
        tables=[table],
        insights=insights,
    )


def data_range_validation(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    """Checks numerical and datetime columns for range and logical consistency issues."""

    numeric_cols = numeric_columns(df)
    datetime_cols = datetime_columns(df)

    if not numeric_cols and not datetime_cols:
        return AnalysisResult(
            analysis_id="data_range_validation",
            title="Data Range Validation",
            summary="No numeric or datetime-like columns available for range validation.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Add numeric or datetime columns (or adjust column selection) to validate ranges.",
                )
            ],
        )

    tables: List[AnalysisTable] = []
    insights: List = []
    issues_table_rows: List[Dict[str, object]] = []
    issues_details: List[Dict[str, object]] = []
    insight_cache: List[str] = []

    price_keywords = (
        "price",
        "cost",
        "amount",
        "revenue",
        "sales",
        "payment",
        "income",
        "charge",
        "fare",
        "fee",
        "salary",
        "wage",
    )
    quantity_keywords = ("quantity", "qty", "count", "units", "stock", "inventory", "orders")
    percentage_keywords = ("percent", "percentage", "pct", "rate", "ratio")

    def register_issue(
        column: str,
        issue_text: str,
        *,
        severity: str = "warning",
        mask: pd.Series | None = None,
        series_reference: pd.Series | None = None,
        affected: int | None = None,
        example: object | None = None,
    ) -> None:
        nonlocal issues_table_rows, issues_details, insights

        if mask is not None:
            mask_series = mask
            try:
                affected_count = int(mask_series.sum())
            except Exception:
                mask_series = pd.Series(mask)
                affected_count = int(mask_series.sum())
            if affected_count == 0:
                return
            affected = affected_count
            if series_reference is not None and example is None:
                try:
                    example = series_reference[mask_series].iloc[0]
                except Exception:  # pragma: no cover - defensive sample extraction
                    example = None

        affected = affected or 0

        issues_details.append(
            {
                "column": column,
                "issue": issue_text,
                "severity": severity,
                "affected_rows": affected,
                "example_value": example,
            }
        )

        issues_table_rows.append(
            {
                "Column": column,
                "Issue": issue_text,
                "Severity": severity.capitalize(),
                "Affected Rows": affected,
                "Example Value": "" if example is None else str(example),
            }
        )

        cache_key = f"{column}:{issue_text}"
        if cache_key not in insight_cache and len(insight_cache) < 6:
            postfix = f" ({affected} rows)" if affected else ""
            insights.append(insight(severity, f"{column}: {issue_text}{postfix}"))
            insight_cache.append(cache_key)

    if numeric_cols:
        numeric_frame = df[numeric_cols]
        stats_df = pd.DataFrame(
            {
                "Column": numeric_cols,
                "Min": numeric_frame.min().values,
                "Max": numeric_frame.max().values,
                "Mean": numeric_frame.mean().values,
                "Std": numeric_frame.std().values,
            }
        )
        tables.append(
            dataframe_to_table(
                stats_df,
                title="Numeric Column Ranges",
                description="Minimum, maximum, mean, and standard deviation for numeric columns.",
            )
        )

        current_year = pd.Timestamp.utcnow().year

        for col in numeric_cols:
            numeric_series = pd.to_numeric(numeric_frame[col], errors="coerce")
            valid_numeric = numeric_series.dropna()
            if valid_numeric.empty:
                continue

            col_lower = col.lower()

            if "age" in col_lower:
                negative_ages = valid_numeric < 0
                if negative_ages.any():
                    register_issue(
                        col,
                        "contains negative ages (expected age ≥ 0)",
                        severity="danger",
                        mask=negative_ages,
                        series_reference=valid_numeric,
                    )

                implausible_ages = valid_numeric > 120
                if implausible_ages.any():
                    register_issue(
                        col,
                        "contains ages above 120",
                        severity="warning",
                        mask=implausible_ages,
                        series_reference=valid_numeric,
                    )

            if any(keyword in col_lower for keyword in price_keywords):
                negative_prices = valid_numeric < 0
                if negative_prices.any():
                    register_issue(
                        col,
                        "contains negative monetary values",
                        severity="danger",
                        mask=negative_prices,
                        series_reference=valid_numeric,
                    )

            if any(keyword in col_lower for keyword in quantity_keywords):
                negative_quantity = valid_numeric < 0
                if negative_quantity.any():
                    register_issue(
                        col,
                        "contains negative counts/quantities",
                        severity="warning",
                        mask=negative_quantity,
                        series_reference=valid_numeric,
                    )

            if any(keyword in col_lower for keyword in percentage_keywords):
                below_zero = valid_numeric < 0
                if below_zero.any():
                    register_issue(
                        col,
                        "contains percentages below 0",
                        severity="warning",
                        mask=below_zero,
                        series_reference=valid_numeric,
                    )

                above_hundred = valid_numeric > 100
                if above_hundred.any():
                    register_issue(
                        col,
                        "contains percentages above 100",
                        severity="warning",
                        mask=above_hundred,
                        series_reference=valid_numeric,
                    )

            if "year" in col_lower:
                early_years = valid_numeric < 1900
                if early_years.any():
                    register_issue(
                        col,
                        "contains years earlier than 1900",
                        severity="warning",
                        mask=early_years,
                        series_reference=valid_numeric,
                    )

                far_future_years = valid_numeric > current_year + 1
                if far_future_years.any():
                    register_issue(
                        col,
                        "contains years far in the future",
                        severity="warning",
                        mask=far_future_years,
                        series_reference=valid_numeric,
                    )

    if datetime_cols:
        datetime_rows: List[Dict[str, object]] = []
        now = pd.Timestamp.utcnow().tz_localize(None).floor("s")
        far_future_threshold = now + pd.DateOffset(years=5)
        far_past_threshold = pd.Timestamp("1900-01-01")

        for col in datetime_cols:
            raw_series = df[col]
            parsed_series = pd.to_datetime(raw_series, errors="coerce")

            try:
                naive_series = parsed_series.dt.tz_localize(None)
            except (TypeError, AttributeError, ValueError):
                naive_series = parsed_series

            valid_datetimes = naive_series.dropna()
            total_count = len(raw_series)
            invalid_count = total_count - len(valid_datetimes)
            coverage_pct = round((len(valid_datetimes) / total_count) * 100, 2) if total_count else 0.0

            if not valid_datetimes.empty:
                min_ts = valid_datetimes.min()
                max_ts = valid_datetimes.max()

                datetime_rows.append(
                    {
                        "Column": col,
                        "Earliest": min_ts.isoformat(),
                        "Latest": max_ts.isoformat(),
                        "Invalid Values": invalid_count,
                        "Coverage (%)": coverage_pct,
                    }
                )

                too_old_mask = valid_datetimes < far_past_threshold
                if too_old_mask.any():
                    register_issue(
                        col,
                        f"contains dates earlier than {far_past_threshold.date()}",
                        severity="warning",
                        mask=too_old_mask,
                        series_reference=valid_datetimes,
                    )

                too_future_mask = valid_datetimes > far_future_threshold
                if too_future_mask.any():
                    register_issue(
                        col,
                        "contains timestamps more than 5 years in the future",
                        severity="warning",
                        mask=too_future_mask,
                        series_reference=valid_datetimes,
                    )

            else:
                datetime_rows.append(
                    {
                        "Column": col,
                        "Earliest": "N/A",
                        "Latest": "N/A",
                        "Invalid Values": invalid_count,
                        "Coverage (%)": coverage_pct,
                    }
                )

                invalid_examples = raw_series[parsed_series.isna()].dropna().head(1)
                example_value = invalid_examples.iloc[0] if not invalid_examples.empty else None
                register_issue(
                    col,
                    "all values failed datetime parsing",
                    severity="danger",
                    affected=invalid_count,
                    example=example_value,
                )

        if datetime_rows:
            tables.append(
                dataframe_to_table(
                    pd.DataFrame(datetime_rows),
                    title="Datetime Column Coverage",
                    description="Earliest/latest timestamp and parsing coverage for datetime-like columns.",
                )
            )

    if issues_table_rows:
        tables.append(
            dataframe_to_table(
                pd.DataFrame(issues_table_rows),
                title="Logical Consistency Checks",
                description="Potential range or logical consistency issues detected across columns.",
            )
        )

    if not insights:
        insights.append(
            insight("success", "Numeric and datetime ranges appear logically consistent."),
        )

    summary_parts = []
    if numeric_cols:
        summary_parts.append(f"{len(numeric_cols)} numeric columns")
    if datetime_cols:
        summary_parts.append(f"{len(datetime_cols)} datetime columns")

    summary_prefix = "Reviewed " + " and ".join(summary_parts) if summary_parts else "Reviewed column ranges"
    if issues_table_rows:
        summary = f"{summary_prefix}; flagged {len(issues_table_rows)} potential issue(s)."
    else:
        summary = f"{summary_prefix}; no logical issues detected."

    status = "success"
    if issues_details:
        status = "warning"
        if any(issue["severity"] == "danger" for issue in issues_details):
            status = "warning"

    return AnalysisResult(
        analysis_id="data_range_validation",
        title="Data Range Validation",
        summary=summary,
        status=status,
        tables=tables,
        insights=insights,
        details={"logical_issues": issues_details},
    )


def data_types_validation(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    """Distribution of data types and detection of potential conversion opportunities."""

    try:
        dtype_counts = df.dtypes.value_counts().rename_axis("Data Type").reset_index(name="Count")
        dtype_counts.sort_values("Data Type", inplace=True)

        column_details = (
            pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str).values,
                }
            )
            .sort_values(["Data Type", "Column"])
            .reset_index(drop=True)
        )

        type_summary_table = dataframe_to_table(
            dtype_counts,
            title="Data Types Distribution",
            description="Number of columns per pandas data type.",
        )

        column_detail_table = dataframe_to_table(
            column_details,
            title="Column Data Types",
            description="Each column with its detected pandas data type.",
        )

        categorical_cols = categorical_columns(df)
        dt_cols = datetime_columns(df)
        num_cols = numeric_columns(df)

        insights = [
            insight("info", f"Detected {len(num_cols)} numeric columns."),
            insight("info", f"Detected {len(categorical_cols)} categorical columns."),
            insight("info", f"Detected {len(dt_cols)} datetime-like columns."),
        ]

        if num_cols:
            preview = ", ".join(num_cols[:5]) + ("..." if len(num_cols) > 5 else "")
            insights.append(insight("success", f"Numeric columns: {preview}"))

        if categorical_cols:
            preview = ", ".join(categorical_cols[:5]) + ("..." if len(categorical_cols) > 5 else "")
            insights.append(insight("info", f"Categorical columns: {preview}"))

        return AnalysisResult(
            analysis_id="data_types_validation",
            title="Data Types Validation",
            summary="Distribution of data types and detection of potential conversion opportunities.",
            tables=[type_summary_table, column_detail_table],
            insights=insights,
        )

    except Exception as exc:  # pragma: no cover - defensive handling
        return AnalysisResult(
            analysis_id="data_types_validation",
            title="Data Types Validation",
            summary="Failed to analyze data types.",
            status="error",
            insights=[insight("danger", f"Error analyzing data types: {str(exc)}")],
        )


def missing_value_analysis(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    """Highlights columns with missing data to guide imputation strategy."""

    missing_counts = df.isna().sum()
    if missing_counts.sum() == 0:
        return AnalysisResult(
            analysis_id="missing_value_analysis",
            title="Missing Value Analysis",
            summary="No missing values detected in the selected columns.",
            status="success",
            insights=[insight("success", "All selected columns are fully populated.")],
        )

    percent = (missing_counts / len(df)) * 100 if len(df) > 0 else 0
    missing_df = (
        pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": missing_counts.values,
                "Percent Missing": percent.values,
            }
        )
        .sort_values("Percent Missing", ascending=False)
        .reset_index(drop=True)
    )

    table = dataframe_to_table(
        missing_df,
        title="Missing Values by Column",
        description="Count and percentage of missing values for each column.",
    )

    insights: List = []
    high_missing = missing_df[missing_df["Percent Missing"] > 30]
    if not high_missing.empty:
        insights.append(
            insight(
                "warning",
                "Columns with more than 30% missing values: " + ", ".join(high_missing["Column"].head(5)),
            )
        )
    insights.append(
        insight(
            "info",
            f"Total missing cells: {int(missing_counts.sum())} ({percent.mean():.2f}% on average per column)",
        )
    )

    return AnalysisResult(
        analysis_id="missing_value_analysis",
        title="Missing Value Analysis",
        summary="Highlights columns with missing data to guide imputation strategy.",
        tables=[table],
        insights=insights,
    )


def duplicate_detection(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    """Detect duplicate rows and highlight potential quality issues."""

    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    total_rows = len(df)

    metrics = [
        metric("Duplicate Rows", duplicate_count),
        metric(
            "Duplicate Percentage",
            f"{(duplicate_count / total_rows * 100) if total_rows else 0:.2f}",
            unit="%",
        ),
    ]

    tables: List[AnalysisTable] = []
    if duplicate_count > 0:
        duplicate_rows = df[duplicate_mask].copy()
        tables.append(
            dataframe_to_table(
                duplicate_rows,
                title="Duplicate Rows",
                description="All duplicate rows detected in the dataset.",
            )
        )

    insights: List = []
    if duplicate_count == 0:
        insights.append(insight("success", "No duplicate rows detected."))
    else:
        insights.append(
            insight(
                "warning",
                f"Detected {duplicate_count} duplicate rows ({(duplicate_count / total_rows * 100) if total_rows else 0:.2f}%).",
            )
        )

    return AnalysisResult(
        analysis_id="duplicate_detection",
        title="Duplicate Detection",
        summary="Identifies duplicate rows for data quality review.",
        metrics=metrics,
        tables=tables,
        insights=insights,
    )

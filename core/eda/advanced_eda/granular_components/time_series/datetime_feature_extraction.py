"""Datetime Feature Extraction Component.

Summarises candidate calendar features for datetime columns.
"""

from typing import Any, Dict, Optional


class DatetimeFeatureExtractionAnalysis:
    """Evaluate datetime columns for feature engineering opportunities."""

    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "datetime_feature_extraction",
            "display_name": "Datetime Feature Extraction",
            "description": "Extract calendar-based feature ideas (month, weekday, hour, etc.).",
            "category": "time-series",
            "complexity": "basic",
            "tags": ["datetime", "feature-engineering", "seasonality", "calendar"],
            "estimated_runtime": "1-2 seconds",
            "icon": "⏱",
        }

    def validate_data_compatibility(self, data_preview: Optional[Dict[str, Any]] = None) -> bool:
        if not data_preview:
            return True

        datetime_cols = data_preview.get("datetime_columns") or []
        potential_datetime_cols = data_preview.get("potential_datetime_columns") or []
        object_cols = data_preview.get("object_columns") or data_preview.get("categorical_columns") or []

        # Allow execution when we can already see datetime columns, or have strong candidates
        if datetime_cols or potential_datetime_cols:
            return True

        # Be permissive if there are object columns that might be convertible; the runtime code
        # will surface actionable guidance when no datetime values are found.
        return bool(object_cols)

    def generate_code(self, data_preview: Optional[Dict[str, Any]] = None) -> str:
        return '''
# ==== DATETIME FEATURE EXTRACTION ====

import pandas as pd

print("=" * 70)
print("⏱ DATETIME FEATURE EXTRACTION")
print("=" * 70)

if isinstance(df.index, pd.DatetimeIndex):
    print("ℹ️  Dataset index is already a DatetimeIndex")

datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
if not datetime_cols:
    # Try coercing object columns
    for col in df.select_dtypes(include=['object']).columns:
        sample = pd.to_datetime(df[col], errors='coerce')
        if sample.notna().mean() > 0.5:
            datetime_cols.append(col)

print(f"Detected {len(datetime_cols)} datetime-like columns: {datetime_cols}\n")

if not datetime_cols:
    print("❌ No datetime columns detected. Convert columns with pd.to_datetime first.")
else:
    for col in datetime_cols:
        series = pd.to_datetime(df[col], errors='coerce')
        valid = series.dropna()
        coverage = len(valid) / len(series) * 100 if len(series) else 0
        print(f"📅 Column: {col}")
        print(f"   • Coverage: {coverage:.1f}%")
        if len(valid) == 0:
            print("   • No valid timestamps after coercion. Check original format.")
            continue

        print(f"   • Span: {valid.min()} → {valid.max()}")
        print(f"   • Unique years: {valid.dt.year.nunique()}")
        print(f"   • Unique months: {valid.dt.month.nunique()}")
        print(f"   • Unique weekdays: {valid.dt.weekday.nunique()}")
        print(f"   • Unique hours: {valid.dt.hour.nunique()}")

        print("   Suggested features:")
        print("     - df['%s_year'] = df['%s'].dt.year" % (col, col))
        print("     - df['%s_month'] = df['%s'].dt.month" % (col, col))
        print("     - df['%s_weekday'] = df['%s'].dt.weekday" % (col, col))
        if valid.dt.hour.nunique() > 1:
            print("     - df['%s_hour'] = df['%s'].dt.hour" % (col, col))
        print("     - df['%s_is_weekend'] = df['%s'].dt.weekday >= 5" % (col, col))
        print()

print("✅ Datetime feature extraction complete")
print("=" * 70)
'''

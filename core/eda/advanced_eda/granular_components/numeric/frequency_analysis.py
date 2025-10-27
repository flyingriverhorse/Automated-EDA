"""Numeric Frequency Analysis Component.

Provides frequency distributions for numeric variables with adaptive binning.
"""
from typing import Any, Dict


class NumericFrequencyAnalysis:
    """Analyze frequency distributions and dominant ranges for numeric variables."""

    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "numeric_frequency_analysis",
            "display_name": "Numeric Frequency Analysis",
            "description": "Frequency counts of numeric columns with adaptive grouping for dense ranges.",
            "category": "univariate",
            "complexity": "basic",
            "tags": ["frequency", "numeric", "distribution", "counts"],
            "estimated_runtime": "1-2 seconds",
            "icon": "📈",
        }

    def validate_data_compatibility(self, data_preview: Dict[str, Any] | None = None) -> bool:
        """Determine whether numeric columns are available for the analysis."""

        if not data_preview:
            return True

        numeric_columns = data_preview.get("numeric_columns") or data_preview.get("number_columns")
        if isinstance(numeric_columns, list):
            return len(numeric_columns) > 0

        column_types = data_preview.get("column_types")
        if isinstance(column_types, dict):
            return any(dtype in {"int", "float", "number", "numeric"} for dtype in column_types.values())

        return True

    def generate_code(self, data_preview: Dict[str, Any] | None = None) -> str:
        """Provide an illustrative code snippet for notebook execution."""

        return """
# ===== NUMERIC FREQUENCY ANALYSIS =====

import numpy as np
import pandas as pd

print("=" * 60)
print("📈 NUMERIC FREQUENCY ANALYSIS")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n🔢 Found {len(numeric_cols)} numeric columns")

if len(numeric_cols) == 0:
    print("❌ No numeric columns available for analysis")
else:
    for col in numeric_cols:
        print(f"\n{'=' * 50}")
        print(f"📊 COLUMN: {col}")
        print('=' * 50)

        series = df[col].dropna()
        missing = df[col].isna().sum()
        total = len(df[col])
        unique_values = series.nunique()
        print(f"Total values: {total:,}")
        print(f"Missing: {missing:,} ({missing / total * 100:.1f}%)")
        print(f"Unique (non-null): {unique_values:,}")

        if unique_values <= 25:
            counts = series.value_counts()
            mode = "Exact value counts"
        else:
            bins = min(10, max(5, int(np.sqrt(len(series)))))
            categories = pd.cut(series, bins=bins, duplicates='drop')
            counts = categories.value_counts().sort_index()
            mode = f"{bins} adaptive bins"

        print(f"\n{mode}:")
        for idx, (value, count) in enumerate(counts.head(20).items(), start=1):
            if isinstance(value, pd.Interval):
                label = f"{value.left:.2f} – {value.right:.2f}"
            else:
                label = value
            pct = count / len(series) * 100
            print(f"  {idx:02d}. {label:<18} | {count:>8,} ({pct:5.1f}%)")

        if counts.shape[0] > 20:
            print(f"  … plus {counts.shape[0] - 20} additional buckets")

print("\n" + "=" * 60)
print("✅ Numeric frequency analysis complete!")
print("=" * 60)
"""

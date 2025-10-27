"""Categorical Cardinality Profile Component.

Produces a concise overview of unique category counts and saturation ratios
for every categorical column.
"""

from typing import Any, Dict, Optional


class CategoricalCardinalityProfileAnalysis:
    """Profile categorical columns for high or low cardinality."""

    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "categorical_cardinality_profile",
            "display_name": "Categorical Cardinality Profile",
            "description": "Quickly summarise unique category counts, dominance, and sparsity.",
            "category": "categorical",
            "complexity": "basic",
            "tags": ["categorical", "cardinality", "profile", "encoding"],
            "estimated_runtime": "1-2 seconds",
            "icon": "🔢",
        }

    def validate_data_compatibility(self, data_preview: Optional[Dict[str, Any]] = None) -> bool:
        if not data_preview:
            return True
        categorical_cols = data_preview.get("categorical_columns") or data_preview.get("object_columns")
        return bool(categorical_cols)

    def generate_code(self, data_preview: Optional[Dict[str, Any]] = None) -> str:
        return '''
# ==== CATEGORICAL CARDINALITY PROFILE ====

import pandas as pd

print("=" * 70)
print("🔢 CATEGORICAL CARDINALITY PROFILE")
print("=" * 70)

categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns\n")

if not categorical_cols:
    print("❌ No categorical columns available for profiling")
else:
    overview_rows = []
    for col in categorical_cols:
        series = df[col]
        total = len(series)
        missing = series.isna().sum()
        valid = total - missing
        unique = series.dropna().nunique()
        cardinality_pct = (unique / valid * 100) if valid else 0.0
        top_value = series.dropna().value_counts().head(1)
        if not top_value.empty:
            top_label = str(top_value.index[0])[:40]
            top_share = top_value.iloc[0] / valid * 100
        else:
            top_label = "—"
            top_share = 0.0

        overview_rows.append({
            "Column": col,
            "Unique": unique,
            "Cardinality_%": round(cardinality_pct, 2),
            "Top Value": top_label,
            "Top Share %": round(top_share, 2),
            "Missing %": round(missing / total * 100 if total else 0.0, 2),
        })

        if cardinality_pct >= 60:
            print(f"⚠️  {col}: High cardinality ({cardinality_pct:.1f}% unique). Consider hashing or target encoding.")
        if top_share >= 80 and unique > 1:
            print(f"ℹ️  {col}: Dominated by '{top_label}' ({top_share:.1f}%). One-hot encoding may create sparse features.")

    profile_df = pd.DataFrame(overview_rows)
    print("\nCardinality overview table:")
    print(profile_df.to_string(index=False))

print("\n✅ Cardinality profile complete")
print("=" * 70)
'''

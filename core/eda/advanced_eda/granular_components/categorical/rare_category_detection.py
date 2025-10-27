"""Rare Category Detection Component.

Highlights infrequent categories that may need regrouping prior to encoding.
"""

from typing import Any, Dict, Optional


class RareCategoryDetectionAnalysis:
    """Identify low-frequency categories across categorical columns."""

    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "rare_category_detection",
            "display_name": "Rare Category Detection",
            "description": "Flag categories that fall below a configurable frequency threshold.",
            "category": "categorical",
            "complexity": "basic",
            "tags": ["categorical", "rare", "encoding", "feature-engineering"],
            "estimated_runtime": "1-2 seconds",
            "icon": "🧩",
        }

    def validate_data_compatibility(self, data_preview: Optional[Dict[str, Any]] = None) -> bool:
        if not data_preview:
            return True
        categorical_cols = data_preview.get("categorical_columns") or data_preview.get("object_columns")
        return bool(categorical_cols)

    def generate_code(self, data_preview: Optional[Dict[str, Any]] = None) -> str:
        return '''
# ==== RARE CATEGORY DETECTION ====

import pandas as pd

print("=" * 70)
print("🧩 RARE CATEGORY DETECTION")
print("=" * 70)

categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"Scanning {len(categorical_cols)} categorical columns\n")

threshold_pct = 5.0  # percentage threshold
min_count = 10       # absolute count threshold

rare_rows = []
for col in categorical_cols:
    series = df[col].dropna()
    total = len(series)
    if total == 0:
        continue

    counts = series.value_counts()
    for category, count in counts.items():
        percent = count / total * 100
        if percent <= threshold_pct or count <= min_count:
            rare_rows.append({
                "Column": col,
                "Category": str(category),
                "Count": int(count),
                "Percent": round(percent, 2),
            })

if not rare_rows:
    print("✅ No categories fell below the rare thresholds")
else:
    rare_df = pd.DataFrame(rare_rows).sort_values(["Percent", "Count"])
    print(rare_df.to_string(index=False))
    print("\n⚠️  Consider grouping these categories under an 'Other' bucket or using target/frequency encoding.")

print("\nThresholds:")
print(f"  - Frequency <= {threshold_pct}%")
print(f"  - Count <= {min_count} rows")
print("✔ Rare category scan complete")
print("=" * 70)
'''

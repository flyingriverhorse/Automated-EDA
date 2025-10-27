"""Text Length Distribution Analysis Component."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class TextLengthDistributionAnalysis:
    """Focused analysis component for understanding text length patterns."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "text_length_distribution",
            "display_name": "Text Length Distribution",
            "description": "Character and word-length distribution statistics for text columns.",
            "category": "Text Analysis",
            "complexity": "intermediate",
            "required_data_types": ["text", "string", "categorical"],
            "estimated_runtime": "5-10 seconds",
            "icon": "align-left",
            "tags": ["text", "length", "distribution", "nlp"],
        }

    def _iter_preview_values(self, data_preview: Dict[str, Any]) -> Iterable[str]:
        if not data_preview:
            return []

        rows = data_preview.get("data") or data_preview.get("sample_data") or []
        if not rows:
            return []

        for row in rows:
            # sample_data returns dictionaries while data may be lists
            values = row.values() if isinstance(row, dict) else row
            for value in values:
                if isinstance(value, str):
                    yield value

    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        if not data_preview:
            return False

        if any(True for _ in self._iter_preview_values(data_preview)):
            return True

        text_like_columns = (
            data_preview.get("text_columns")
            or data_preview.get("categorical_columns")
            or data_preview.get("object_columns")
            or []
        )
        return len(text_like_columns) > 0

    def generate_code(self, data_preview: Dict[str, Any] | None = None) -> str:
        return '''import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

print("=== TEXT LENGTH DISTRIBUTION ANALYSIS ===\n")

# Identify text-like columns (object dtype or string-like)
text_cols = [
    col for col in df.columns
    if df[col].dtype == "object" or "string" in str(df[col].dtype).lower()
]

if not text_cols:
    print("❌ No text-like columns detected. Provide text data to analyse length distributions.")
else:
    print(f"📚 Analysing {len(text_cols)} text columns: {text_cols}\n")

    length_summary = []
    charts = []

    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.str.len() > 0]

        if series.empty:
            print(f"⚠️ Column '{col}' has no non-empty text values after cleaning.\n")
            continue

        char_lengths = series.str.len()
        word_counts = series.str.split().map(len)

        length_summary.append({
            "Column": col,
            "Rows Analysed": len(series),
            "Avg Char Length": round(char_lengths.mean(), 2),
            "Median Char Length": round(char_lengths.median(), 2),
            "P95 Char Length": round(char_lengths.quantile(0.95), 2),
            "Avg Word Count": round(word_counts.mean(), 2),
            "Median Word Count": round(word_counts.median(), 2),
            "P95 Word Count": round(word_counts.quantile(0.95), 2),
            "Short Text Share (%)": round((char_lengths <= 20).mean() * 100, 1),
            "Long Text Share (%)": round((char_lengths >= 200).mean() * 100, 1),
        })

        # Visualise char/word counts side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(char_lengths, bins=25, ax=axes[0], color="#4C51BF", alpha=0.85)
        axes[0].axvline(char_lengths.mean(), color="black", linestyle="--", linewidth=1)
        axes[0].set_title(f"Character length • {col}")
        axes[0].set_xlabel("Characters")

        sns.histplot(word_counts, bins=25, ax=axes[1], color="#6B46C1", alpha=0.85)
        axes[1].axvline(word_counts.mean(), color="black", linestyle="--", linewidth=1)
        axes[1].set_title(f"Word count • {col}")
        axes[1].set_xlabel("Words")

        plt.tight_layout()
        charts.append(fig)

    if not length_summary:
        print("❌ No analyzable text data detected after cleaning.")
    else:
        summary_df = pd.DataFrame(length_summary)
        print("📊 TEXT LENGTH SUMMARY")
        print(summary_df.to_string(index=False))

        # Recommendations based on length distribution
        print("\n💡 RECOMMENDATIONS")
        for row in length_summary:
            col = row["Column"]
            long_share = row["Long Text Share (%)"]
            short_share = row["Short Text Share (%)"]

            if long_share >= 40:
                print(f"• {col}: {long_share}% of rows exceed 200 characters. Consider truncation or summarisation.")
            if short_share >= 70:
                print(f"• {col}: {short_share}% of rows are very short. Consider merging with additional context.")
            if long_share < 20 and short_share < 60:
                print(f"• {col}: Balanced text lengths — suitable for modelling without heavy preprocessing.")

        print("\n📈 Saved distribution charts in memory (charts list). Display them as needed.")
'''

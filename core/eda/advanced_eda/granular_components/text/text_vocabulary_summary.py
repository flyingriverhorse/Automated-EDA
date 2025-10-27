"""Text Vocabulary Summary Analysis Component."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class TextVocabularySummaryAnalysis:
    """Focused component for lexical diversity metrics."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "text_vocabulary_summary",
            "display_name": "Text Vocabulary Summary",
            "description": "Lexical diversity and vocabulary richness metrics for text columns.",
            "category": "Text Analysis",
            "complexity": "intermediate",
            "required_data_types": ["text", "string"],
            "estimated_runtime": "8-12 seconds",
            "icon": "book-open",
            "tags": ["text", "nlp", "vocabulary", "diversity"],
        }

    def _iter_preview_values(self, data_preview: Dict[str, Any]) -> Iterable[str]:
        if not data_preview:
            return []

        rows = data_preview.get("data") or data_preview.get("sample_data") or []
        if not rows:
            return []

        for row in rows:
            values = row.values() if isinstance(row, dict) else row
            for value in values:
                if isinstance(value, str) and value.strip():
                    yield value

    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        if not data_preview:
            return False

        values = list(self._iter_preview_values(data_preview))
        if len(values) >= 5:
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
from collections import Counter

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[\w']+")

print("=== TEXT VOCABULARY SUMMARY ===\n")

text_cols = [
    col for col in df.columns
    if df[col].dtype == "object" or "string" in str(df[col].dtype).lower()
]

if not text_cols:
    print("❌ No text-like columns detected. Provide text data to compute vocabulary metrics.")
else:
    summary_rows = []

    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.str.len() > 0]

        if series.empty:
            print(f"⚠️ Column '{col}' has no usable text for vocabulary analysis.\n")
            continue

        token_counter = Counter()
        total_tokens = 0

        for value in series:
            tokens = TOKEN_PATTERN.findall(value.lower())
            if tokens:
                token_counter.update(tokens)
                total_tokens += len(tokens)

        if total_tokens == 0:
            print(f"⚠️ Column '{col}' did not yield any tokens after cleaning.\n")
            continue

        unique_tokens = len(token_counter)
        hapax_tokens = sum(1 for count in token_counter.values() if count == 1)
        ttr = unique_tokens / total_tokens
        hapax_ratio = hapax_tokens / unique_tokens if unique_tokens else 0
        avg_word_length = (
            sum(len(token) * count for token, count in token_counter.items()) / total_tokens
        )

        summary_rows.append({
            "Column": col,
            "Rows Analysed": len(series),
            "Total Tokens": total_tokens,
            "Unique Tokens": unique_tokens,
            "Type-Token Ratio": round(ttr, 3),
            "Hapax Count": hapax_tokens,
            "Hapax Ratio": round(hapax_ratio, 3),
            "Avg Word Length": round(avg_word_length, 2),
        })

        print(f"📚 {col.upper()} — LEXICAL METRICS")
        print(f"   • Rows analysed: {len(series)}")
        print(f"   • Total tokens: {total_tokens}")
        print(f"   • Unique tokens: {unique_tokens}")
        print(f"   • Type-token ratio: {ttr:.3f}")
        print(f"   • Hapax legomena: {hapax_tokens} ({hapax_ratio:.1%})")
        print(f"   • Avg word length: {avg_word_length:.2f}")

        if ttr < 0.25:
            print("   ⚠️ Low lexical diversity — consider removing repetitive boilerplate.")
        elif ttr > 0.6 and total_tokens > 200:
            print("   ✅ High lexical diversity detected.")

        if hapax_ratio > 0.5:
            print("   ℹ️ Many tokens appear only once — consider stemming before modelling.")
        print()

    if not summary_rows:
        print("❌ Vocabulary summary could not be generated.")
    else:
        summary_df = pd.DataFrame(summary_rows)
        print("\n📊 VOCABULARY SUMMARY TABLE")
        print(summary_df.to_string(index=False))

        median_ttr = summary_df["Type-Token Ratio"].median()
        avg_hapax_ratio = summary_df["Hapax Ratio"].mean()

        print("\n📌 AGGREGATE INSIGHTS")
        print(f"• Median type-token ratio: {median_ttr:.3f}")
        print(f"• Average hapax ratio: {avg_hapax_ratio:.3f}")
        print("• Columns suitable for topic modelling: ", end="")
        suitable = summary_df[summary_df["Type-Token Ratio"] > 0.35]["Column"].tolist()
        print(suitable if suitable else "None identified")
'''

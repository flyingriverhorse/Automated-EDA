"""Text Token Frequency Analysis Component."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class TextTokenFrequencyAnalysis:
    """Focused component for token and n-gram frequency exploration."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "text_token_frequency",
            "display_name": "Text Token Frequency",
            "description": "Top tokens and short phrases detected across text columns.",
            "category": "Text Analysis",
            "complexity": "intermediate",
            "required_data_types": ["text", "string"],
            "estimated_runtime": "10-15 seconds",
            "icon": "font",
            "tags": ["text", "nlp", "tokens", "bigrams"],
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
        if values:
            unique_values = {value.strip().lower() for value in values if value.strip()}
            if len(unique_values) > 1:
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

import pandas as pd

TOKEN_PATTERN = re.compile(r"[\w']+")

print("=== TEXT TOKEN FREQUENCY ANALYSIS ===\n")

text_cols = [
    col for col in df.columns
    if df[col].dtype == "object" or "string" in str(df[col].dtype).lower()
]

if not text_cols:
    print("❌ No text-like columns detected. Provide text data to analyse token frequencies.")
else:
    print(f"🧾 Analysing {len(text_cols)} text columns: {text_cols}\n")

    token_tables = []
    global_counter = Counter()

    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.str.len() > 0]

        if series.empty:
            print(f"⚠️ Column '{col}' has no non-empty text values after cleaning.\n")
            continue

        token_counter = Counter()
        bigram_counter = Counter()

        for value in series:
            tokens = TOKEN_PATTERN.findall(value.lower())
            if not tokens:
                continue
            token_counter.update(tokens)
            global_counter.update(tokens)
            if len(tokens) > 1:
                bigram_counter.update(zip(tokens, tokens[1:]))

        total_tokens = sum(token_counter.values())
        if total_tokens == 0:
            print(f"⚠️ Column '{col}' yielded zero tokens after cleaning.\n")
            continue

        top_tokens = token_counter.most_common(15)
        top_bigrams = [(" ".join(bigram), count) for bigram, count in bigram_counter.most_common(10)]

        token_table = pd.DataFrame([
            {
                "Column": col,
                "Type": "Token",
                "Text": token,
                "Count": count,
                "Share (%)": round((count / total_tokens) * 100, 2),
            }
            for token, count in top_tokens
        ])

        phrase_table = pd.DataFrame([
            {
                "Column": col,
                "Type": "Phrase",
                "Text": phrase,
                "Count": count,
                "Share (%)": round((count / max(total_tokens - 1, 1)) * 100, 2),
            }
            for phrase, count in top_bigrams
        ])

        combined = pd.concat([token_table, phrase_table], ignore_index=True)
        token_tables.append(combined)

        print(f"📌 {col.upper()} — TOP TOKENS")
        print(token_table[['Text', 'Count', 'Share (%)']].to_string(index=False))
        print()
        if not phrase_table.empty:
            print("📎 TOP PHRASES (Bigrams)")
            print(phrase_table[['Text', 'Count', 'Share (%)']].to_string(index=False))
            print()

        dominant_token, dominant_count = top_tokens[0]
        dominance = (dominant_count / total_tokens) * 100
        if dominance > 35:
            print(f"⚠️ Dominant token '{dominant_token}' covers {dominance:.1f}% of all tokens — consider normalisation.")
        elif dominance < 15:
            print(f"✅ Vocabulary looks diverse (top token share {dominance:.1f}%).")
        print("-" * 60)

    if not token_tables:
        print("❌ Token frequency analysis could not be generated.")
    else:
        global_top = pd.DataFrame([
            {"Text": token, "Count": count}
            for token, count in global_counter.most_common(20)
        ])
        print("\n🌐 DATASET-WIDE TOP TOKENS")
        print(global_top.to_string(index=False))

        full_table = pd.concat(token_tables, ignore_index=True)
        print("\n📊 COMBINED TOKEN & PHRASE TABLE")
        print(full_table.to_string(index=False))

        print("\n💡 TIP: Use stop-word filtering or stemming before modelling to reduce token dominance.")
'''

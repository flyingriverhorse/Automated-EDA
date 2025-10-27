"""Text Feature Engineering Profile Component."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class TextFeatureEngineeringProfileAnalysis:
    """Component providing feature-engineering readiness diagnostics for text."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "text_feature_engineering_profile",
            "display_name": "Text Feature Engineering Profile",
            "description": (
                "Highlights stopword density, numeric content, casing and sparsity signals to prepare "
                "text columns for vectorisation and downstream modelling."
            ),
            "category": "Text Analysis",
            "complexity": "intermediate",
            "required_data_types": ["text", "string"],
            "estimated_runtime": "6-10 seconds",
            "icon": "settings",
            "tags": ["text", "feature engineering", "nlp", "preprocessing"],
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
        values = list(self._iter_preview_values(data_preview or {}))
        if len(values) >= 5:
            return True

        text_like_columns: List[str] = (
            (data_preview or {}).get("text_columns")
            or (data_preview or {}).get("categorical_columns")
            or (data_preview or {}).get("object_columns")
            or []
        )
        return len(text_like_columns) > 0

    def generate_code(self, data_preview: Dict[str, Any] | None = None) -> str:
        return '''import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    import langid
except Exception:  # optional dependency
    langid = None

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def detect_language_profile(sample):
    if langid is None or not sample:
        return "unknown", 0.0, {}, 0.0

    language_counts = Counter()
    confidence_totals = {}

    for text in sample:
        try:
            language, confidence = langid.classify(text)
        except Exception:
            continue
        language_counts[language] += 1
        confidence_totals.setdefault(language, []).append(confidence)

    if not language_counts:
        return "unknown", 0.0, {}, 0.0

    total = sum(language_counts.values())
    primary_language, primary_count = language_counts.most_common(1)[0]
    primary_share = primary_count / total
    distribution = {lang: round((count / total) * 100, 1) for lang, count in language_counts.items()}
    avg_confidence = float(np.mean(confidence_totals.get(primary_language, [0.0]))) if confidence_totals else 0.0

    return primary_language, primary_share, distribution, avg_confidence

print("=== TEXT FEATURE ENGINEERING PROFILE ===\n")

text_cols = [
    col for col in df.columns
    if df[col].dtype == "object" or "string" in str(df[col].dtype).lower()
]

if not text_cols:
    print("❌ No text-like columns detected. Provide text data to review feature-engineering readiness.")
else:
    summary_rows = []
    embedding_counts = Counter()

    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.str.len() > 0]

        total_rows = df[col].shape[0]
        non_empty_rows = series.shape[0]
        empty_share_pct = (1 - (non_empty_rows / total_rows)) * 100 if total_rows else 0.0

        if series.empty:
            print(f"⚠️ Column '{col}' contains no usable text for feature profiling.\n")
            continue

        tokens_per_row = []
        stopword_count = 0
        numeric_token_count = 0
        uppercase_chars = 0
        alpha_chars = 0
        digit_chars = 0
        punctuation_chars = 0
        non_ascii_chars = 0
        token_length_sum = 0
        vocabulary = Counter()

        concatenated = " ".join(series.tolist())
        total_chars = len(concatenated)

        for text in series:
            tokens = [tok for tok in text.lower().split() if tok]
            tokens_per_row.append(len(tokens))
            stopword_count += sum(1 for tok in tokens if tok in STOP_WORDS)
            numeric_token_count += sum(1 for tok in tokens if tok.isdigit())
            token_length_sum += sum(len(tok) for tok in tokens)
            vocabulary.update(tokens)

            uppercase_chars += sum(1 for ch in text if ch.isupper())
            alpha_chars += sum(1 for ch in text if ch.isalpha())
            digit_chars += sum(1 for ch in text if ch.isdigit())
            punctuation_chars += sum(1 for ch in text if ch in string.punctuation)
            non_ascii_chars += sum(1 for ch in text if ord(ch) > 127)

        total_tokens = sum(tokens_per_row)
        if total_tokens == 0:
            print(f"⚠️ Column '{col}' yielded zero tokens after tokenisation.\n")
            continue

        avg_tokens_per_doc = total_tokens / len(tokens_per_row)
        median_tokens_per_doc = float(np.median(tokens_per_row))
        max_tokens_per_doc = float(max(tokens_per_row))
        avg_token_length = token_length_sum / total_tokens if total_tokens else 0.0

        stopword_share_pct = (stopword_count / total_tokens) * 100
        numeric_token_share_pct = (numeric_token_count / total_tokens) * 100
        uppercase_share_pct = (uppercase_chars / alpha_chars) * 100 if alpha_chars else 0.0
        digit_char_share_pct = (digit_chars / total_chars) * 100 if total_chars else 0.0
        punctuation_share_pct = (punctuation_chars / total_chars) * 100 if total_chars else 0.0
        non_ascii_share_pct = (non_ascii_chars / total_chars) * 100 if total_chars else 0.0

        unique_tokens = len(vocabulary)

        primary_language, primary_share, language_distribution, language_confidence = detect_language_profile(series.tolist()[:400])
        multilanguage_content = bool(language_distribution) and (primary_share * 100 < 70 or len(language_distribution) >= 2)

        recommendations = []
        if stopword_share_pct >= 55:
            recommendations.append("Remove stopwords or apply TF-IDF weighting")
        if numeric_token_share_pct >= 12:
            recommendations.append("Extract numeric patterns or engineer digit ratios")
        if punctuation_share_pct >= 8:
            recommendations.append("Strip punctuation or add character n-grams")
        if non_ascii_share_pct >= 5:
            recommendations.append("Normalise Unicode and check language-specific tokenisation")
        if median_tokens_per_doc <= 5:
            recommendations.append("Very short documents — prefer character n-grams or sentence embeddings")
        if unique_tokens >= 5000 and total_tokens >= 10000:
            recommendations.append("High vocabulary size — use hashing or cap vocabulary")
        if uppercase_share_pct >= 50:
            recommendations.append("Preserve casing or add uppercase ratio feature")
        if not recommendations:
            recommendations.append("Standard word-level TF-IDF with lemmatisation is sufficient")

        embedding_recommendation = "TF-IDF (word-level)"
        if multilanguage_content or non_ascii_share_pct >= 5:
            embedding_recommendation = "Multilingual sentence embeddings (e.g., LaBSE, XLM-R)"
        elif median_tokens_per_doc >= 25 or max_tokens_per_doc >= 60 or avg_tokens_per_doc >= 25:
            embedding_recommendation = "Sentence embeddings (e.g., all-MiniLM)"
        elif median_tokens_per_doc <= 8:
            embedding_recommendation = "Character-level TF-IDF / n-grams"

        recommendations.append(f"Preferred embedding: {embedding_recommendation}")
        embedding_counts[embedding_recommendation] += 1

        if multilanguage_content:
            top_langs = ", ".join(
                f"{lang}: {pct:.1f}%" for lang, pct in sorted(language_distribution.items(), key=lambda item: item[1], reverse=True)[:3]
            )
            recommendations.append(f"Mixed-language content detected ({top_langs}) — use multilingual embeddings")

        summary_rows.append({
            "Column": col,
            "Rows Analysed": len(series),
            "Empty Share (%)": round(empty_share_pct, 1),
            "Avg Tokens/Doc": round(avg_tokens_per_doc, 2),
            "Median Tokens/Doc": round(median_tokens_per_doc, 2),
            "Max Tokens/Doc": round(max_tokens_per_doc, 2),
            "Avg Token Length": round(avg_token_length, 2),
            "Stopword Share (%)": round(stopword_share_pct, 1),
            "Numeric Token Share (%)": round(numeric_token_share_pct, 1),
            "Uppercase Char Share (%)": round(uppercase_share_pct, 1),
            "Digit Char Share (%)": round(digit_char_share_pct, 1),
            "Punctuation Char Share (%)": round(punctuation_share_pct, 1),
            "Non-ASCII Char Share (%)": round(non_ascii_share_pct, 1),
            "Vocabulary Size": unique_tokens,
            "Embedding Recommendation": embedding_recommendation,
            "Recommended Focus": recommendations[0],
            "Recommendations": "; ".join(recommendations[:3]),
        })

        print(f"🧰 {col.upper()} — FEATURE ENGINEERING SIGNALS")
        print(f"   • Rows analysed: {len(series)} (empty share {empty_share_pct:.1f}%)")
        print(f"   • Tokens/document — avg: {avg_tokens_per_doc:.2f}, median: {median_tokens_per_doc:.2f}, max: {max_tokens_per_doc:.2f}")
    language_note = "unknown" if primary_language == "unknown" else f"{primary_language} ({primary_share * 100:.1f}% | confidence {language_confidence * 100:.1f}%)"
    print(f"   • Language profile: {language_note}")
        print(f"   • Stopword share: {stopword_share_pct:.1f}% | Numeric tokens: {numeric_token_share_pct:.1f}%")
        print(f"   • Uppercase share: {uppercase_share_pct:.1f}% | Punctuation share: {punctuation_share_pct:.1f}%")
        print(f"   • Non-ASCII share: {non_ascii_share_pct:.1f}% | Vocabulary size: {unique_tokens}")
        print(f"   • Embedding recommendation: {embedding_recommendation}")
        print(f"   • Recommended focus: {recommendations[0]}")
        print()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n📊 FEATURE ENGINEERING SUMMARY TABLE")
        print(summary_df.to_string(index=False))

        print("\n📌 AGGREGATE GUIDANCE")
        print(f"• Median tokens per doc: {summary_df['Median Tokens/Doc'].median():.2f}")
        print(f"• Average stopword share: {summary_df['Stopword Share (%)'].mean():.1f}%")
        print(f"• Average empty share: {summary_df['Empty Share (%)'].mean():.1f}%")
        print(f"• Sentence embedding candidates: {embedding_counts.get('Sentence embeddings (e.g., all-MiniLM)', 0) + embedding_counts.get('Multilingual sentence embeddings (e.g., LaBSE, XLM-R)', 0)}")
        sparse_cols = summary_df[summary_df['Empty Share (%)'] > 40]['Column'].tolist()
        if sparse_cols:
            print(f"• Sparse columns (>40% empty): {', '.join(sparse_cols)}")
        else:
            print("• Sparse columns (>40% empty): None")
    else:
        print("❌ Feature engineering diagnostics could not be generated.")
'''

"""Advanced NLP Profile Analysis Component."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class TextNLPProfileAnalysis:
    """Component describing sentiment and entity enrichment for text columns."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "text_nlp_profile",
            "display_name": "Advanced NLP Profile",
            "description": "Sentiment distribution and named entity highlights for text columns.",
            "category": "Text Analysis",
            "complexity": "advanced",
            "required_data_types": ["text", "string"],
            "estimated_runtime": "12-18 seconds",
            "icon": "message-circle",
            "tags": ["text", "nlp", "sentiment", "entities"],
            "dependencies": ["spacy", "vaderSentiment"],
            "configurable_parameters": [
                {
                    "key": "nlp_sample_limit",
                    "type": "integer",
                    "default": None,
                    "description": "Number of rows to sample per column (leave blank for all rows).",
                }
            ],
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
        return '''import math
from collections import Counter, defaultdict

import numpy as np

try:
    import spacy
except Exception:  # Optional dependency
    spacy = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # Optional dependency
    SentimentIntensityAnalyzer = None

def load_spacy_model(model_name: str = "en_core_web_sm"):
    if spacy is None:
        return None
    try:
        nlp_model = spacy.load(model_name)
    except Exception:
        nlp_model = spacy.blank("en")
        if "sentencizer" not in nlp_model.pipe_names:
            nlp_model.add_pipe("sentencizer")
    return nlp_model


def extract_entities(text_values, nlp_model):
    entity_counts = defaultdict(Counter)
    if nlp_model is not None and "ner" in getattr(nlp_model, "pipe_names", []):
        disable_pipes = [pipe for pipe in nlp_model.pipe_names if pipe != "ner"]
        for doc in nlp_model.pipe(text_values, disable=disable_pipes):
            for ent in getattr(doc, "ents", []):
                text_value = ent.text.strip()
                if text_value:
                    entity_counts[ent.label_][text_value] += 1
    else:
        # Fallback: simple proper-noun detection
        import re
        pattern = re.compile(r"\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\b")
        for value in text_values:
            for match in pattern.findall(value):
                entity_counts["PROPN"][match] += 1
    return entity_counts


def get_sentiment_scores(text_values):
    analyzer = None
    if SentimentIntensityAnalyzer is not None:
        try:
            analyzer = SentimentIntensityAnalyzer()
        except Exception:
            analyzer = None

    positive_words = {"amazing", "great", "love", "excellent", "fantastic", "happy", "wonderful"}
    negative_words = {"bad", "terrible", "hate", "awful", "issue", "bug", "problem", "disappointed"}

    sentiment_scores = []
    sentiment_classes = Counter()

    for value in text_values:
        stripped = value.strip()
        if not stripped:
            continue

        if analyzer is not None:
            score = float(analyzer.polarity_scores(stripped)["compound"])
        else:
            tokens = [token.lower() for token in stripped.split()]
            score = 0
            for token in tokens:
                if token in positive_words:
                    score += 1
                elif token in negative_words:
                    score -= 1
            score = score / max(len(tokens), 1)
            score = max(-1.0, min(1.0, score))

        sentiment_scores.append(score)
        if score >= 0.05:
            sentiment_classes["positive"] += 1
        elif score <= -0.05:
            sentiment_classes["negative"] += 1
        else:
            sentiment_classes["neutral"] += 1

    return sentiment_scores, sentiment_classes


print("=== ADVANCED NLP PROFILE ===\n")

text_cols = [
    col for col in df.columns
    if df[col].dtype == "object" or "string" in str(df[col].dtype).lower()
]

if not text_cols:
    print("❌ No text-like columns detected. Provide text data to compute NLP metrics.")
else:
    nlp_model = load_spacy_model()

    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.str.len() > 0]

        if series.empty:
            print(f"⚠️ Column '{col}' has no usable text for NLP profiling.\n")
            continue

    text_values = series.tolist()

        entity_counts = extract_entities(text_values, nlp_model)
        sentiment_scores, sentiment_classes = get_sentiment_scores(text_values)

        total = max(sum(sentiment_classes.values()), 1)
        positive_pct = (sentiment_classes.get("positive", 0) / total) * 100
        negative_pct = (sentiment_classes.get("negative", 0) / total) * 100
        neutral_pct = (sentiment_classes.get("neutral", 0) / total) * 100

        avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0

        print(f"💬 {col.upper()} — SENTIMENT SNAPSHOT")
        print(f"   • Texts analysed: {len(text_values)}")
        print(f"   • Avg sentiment score: {avg_sentiment:.3f} (-1=negative, +1=positive)")
        print(f"   • Positive: {positive_pct:.1f}% | Neutral: {neutral_pct:.1f}% | Negative: {negative_pct:.1f}%")

        top_entities = []
        for label, counter in entity_counts.items():
            for entity, count in counter.most_common(3):
                label_suffix = f" ({label})" if label != "PROPN" else ""
                top_entities.append(f"{entity}{label_suffix} ×{count}")

        if top_entities:
            print("   • Top entities: " + ", ".join(top_entities[:5]))
        else:
            print("   • Top entities: None detected")

        if positive_pct >= 60:
            print("   ✅ Sentiment is strongly positive.")
        if negative_pct >= 35:
            print("   ⚠️ Elevated negative sentiment detected; investigate root causes.")
        if not top_entities:
            print("   ℹ️ Consider enriching text with domain-specific terminology for better entity detection.")
        print()

    print("\n📌 NOTE: Install spaCy and download 'en_core_web_sm' for high-quality entity extraction.")
'''

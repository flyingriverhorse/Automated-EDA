from __future__ import annotations

"""Runtime handlers for text-focused granular analyses."""

from collections import Counter, defaultdict
from importlib import util as importlib_util
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import langid

try:  # pragma: no cover - optional configuration import
    from config import get_settings as _get_global_settings
except Exception:  # pragma: no cover - defensive fallback
    _get_global_settings = None
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

if importlib_util.find_spec("spacy") is not None:  # pragma: no cover - import guard
    import spacy  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency
    spacy = None

if importlib_util.find_spec("vaderSentiment") is not None:  # pragma: no cover - import guard
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]

from .common import (
    AnalysisChart,
    AnalysisContext,
    AnalysisResult,
    dataframe_to_table,
    fig_to_base64,
    insight,
    metric,
    text_columns,
)

TOKEN_PATTERN = re.compile(r"[\w']+")
PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_POSITIVE_LEXICON = {
    "awesome",
    "amazing",
    "balanced",
    "best",
    "better",
    "brilliant",
    "delight",
    "delighted",
    "excellent",
    "fantastic",
    "favorable",
    "good",
    "great",
    "happy",
    "improved",
    "joy",
    "love",
    "loved",
    "positive",
    "successful",
    "superb",
    "wonderful",
}

_NEGATIVE_LEXICON = {
    "angry",
    "awful",
    "bad",
    "bug",
    "bugs",
    "complaint",
    "disappointing",
    "disappointed",
    "fail",
    "failed",
    "failing",
    "failure",
    "frustrated",
    "hate",
    "hated",
    "issue",
    "issues",
    "negative",
    "poor",
    "problem",
    "problems",
    "sad",
    "terrible",
    "unhappy",
    "worse",
    "worst",
}

_SPACY_MODEL_CACHE: Dict[str, Any] = {}
_SENTIMENT_ANALYZER: Any = None
NLP_SAMPLE_LIMIT: Optional[int] = None # or 500 to limit the row cap
FEATURE_SAMPLE_LIMIT: Optional[int] = 500000 # or 50000 to limit the feature cap
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def _normalize_sample_limit(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default

    if isinstance(value, str):
        candidate = value.strip().lower()
        if not candidate:
            return default
        if candidate in {"none", "null", "all", "unlimited"}:
            return None
        try:
            value = int(candidate)
        except ValueError:
            return default

    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default

    if limit <= 0:
        return None

    return limit


def _apply_text_runtime_config() -> None:  # pragma: no cover - configuration glue
    global NLP_SAMPLE_LIMIT, FEATURE_SAMPLE_LIMIT

    if _get_global_settings is None:
        return

    try:
        settings = _get_global_settings()
    except Exception:
        return

    has_helper = hasattr(settings, "is_field_set")
    global_limit_set = has_helper and settings.is_field_set("EDA_GLOBAL_SAMPLE_LIMIT")
    global_limit = settings.EDA_GLOBAL_SAMPLE_LIMIT if global_limit_set else None

    nlp_limit = settings.GRANULAR_TEXT_NLP_SAMPLE_LIMIT
    if nlp_limit is None and has_helper and not settings.is_field_set("GRANULAR_TEXT_NLP_SAMPLE_LIMIT"):
        nlp_limit = global_limit

    feature_limit = settings.GRANULAR_TEXT_FEATURE_SAMPLE_LIMIT
    if feature_limit is None and has_helper and not settings.is_field_set("GRANULAR_TEXT_FEATURE_SAMPLE_LIMIT"):
        feature_limit = global_limit

    NLP_SAMPLE_LIMIT = nlp_limit
    FEATURE_SAMPLE_LIMIT = feature_limit


_apply_text_runtime_config()


def _load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    if spacy is None:  # pragma: no cover - optional dependency
        return None

    cached = _SPACY_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    try:
        nlp_model = spacy.load(model_name)  # type: ignore[attr-defined]
    except Exception:
        try:
            nlp_model = spacy.blank("en")  # type: ignore[attr-defined]
            if "sentencizer" not in nlp_model.pipe_names:
                nlp_model.add_pipe("sentencizer")
        except Exception:
            nlp_model = None

    _SPACY_MODEL_CACHE[model_name] = nlp_model
    return nlp_model


def _extract_named_entities(sample: Iterable[str], nlp_model: Any) -> Dict[str, Counter[str]]:
    entity_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    if nlp_model is not None and hasattr(nlp_model, "pipe") and "ner" in getattr(nlp_model, "pipe_names", []):
        try:
            disable_pipes = [pipe for pipe in nlp_model.pipe_names if pipe not in {"ner"}]
            for doc in nlp_model.pipe(sample, disable=disable_pipes):  # type: ignore[attr-defined]
                for ent in getattr(doc, "ents", []):
                    text_value = ent.text.strip()
                    if text_value:
                        normalized = " ".join(text_value.split())
                        entity_counts[ent.label_][normalized] += 1
            if entity_counts:
                return entity_counts
        except Exception:  # pragma: no cover - defensive guard
            entity_counts.clear()

    for text_value in sample:
        for match in PROPER_NOUN_PATTERN.findall(text_value):
            normalized = " ".join(match.split())
            entity_counts["PROPN"][normalized] += 1

    return entity_counts


def _get_sentiment_analyzer() -> Any:
    global _SENTIMENT_ANALYZER

    if SentimentIntensityAnalyzer is None:  # pragma: no cover - optional dependency
        return None

    if _SENTIMENT_ANALYZER is None:
        try:
            _SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensive guard
            _SENTIMENT_ANALYZER = None

    return _SENTIMENT_ANALYZER


def _lexicon_sentiment_score(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    score = 0
    for token in tokens:
        if token in _POSITIVE_LEXICON:
            score += 1
        elif token in _NEGATIVE_LEXICON:
            score -= 1

    if score == 0:
        return 0.0

    normalized = score / max(len(tokens), 1)
    return float(max(-1.0, min(1.0, normalized)))


def _compute_sentiment_distribution(sample: Iterable[str]) -> Tuple[List[float], Counter[str]]:
    analyzer = _get_sentiment_analyzer()
    sentiment_scores: List[float] = []
    sentiment_categories: Counter[str] = Counter()

    for text_value in sample:
        stripped = text_value.strip()
        if not stripped:
            continue

        if analyzer is not None:
            try:
                score = float(analyzer.polarity_scores(stripped)["compound"])
            except Exception:  # pragma: no cover - defensive guard
                score = 0.0
        else:
            score = _lexicon_sentiment_score(_tokenize(stripped))

        sentiment_scores.append(score)

        if score >= 0.05:
            sentiment_categories["positive"] += 1
        elif score <= -0.05:
            sentiment_categories["negative"] += 1
        else:
            sentiment_categories["neutral"] += 1

    return sentiment_scores, sentiment_categories


def _clean_text_series(series: pd.Series) -> pd.Series:
    """Return a normalized text series with empty values removed."""

    if series.empty:
        return series.iloc[0:0]

    normalized = series.dropna().astype(str).str.strip()
    return normalized[normalized.str.len() > 0]


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _tokenize_series(series: pd.Series) -> Iterable[List[str]]:
    for value in series:
        tokens = _tokenize(value)
        if tokens:
            yield tokens


def _select_text_columns(df: pd.DataFrame, context: AnalysisContext) -> List[str]:
    candidates = text_columns(df)
    if context.selected_columns:
        filtered = [col for col in context.selected_columns if col in candidates]
        if filtered:
            return filtered
    return candidates


def text_length_distribution(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    columns = _select_text_columns(df, context)
    if not columns:
        return AnalysisResult(
            analysis_id="text_length_distribution",
            title="Text Length Distribution",
            summary="No text-like columns detected for length analysis.",
            status="warning",
            insights=[insight("warning", "Select columns with textual data to review length distributions.")],
        )

    rows: List[Dict[str, Any]] = []
    charts: List[AnalysisChart] = []
    insights_list: List = []

    for idx, column in enumerate(columns):
        cleaned = _clean_text_series(df[column])
        if cleaned.empty:
            insights_list.append(insight("warning", f"Column '{column}' contains no non-empty text values."))
            continue

        char_lengths = cleaned.str.len().astype(float)
        word_counts = cleaned.str.split().map(len).astype(float)

        avg_chars = float(char_lengths.mean()) if not char_lengths.empty else 0.0
        median_chars = float(char_lengths.median()) if not char_lengths.empty else 0.0
        p95_chars = float(char_lengths.quantile(0.95)) if not char_lengths.empty else 0.0
        avg_words = float(word_counts.mean()) if not word_counts.empty else 0.0
        median_words = float(word_counts.median()) if not word_counts.empty else 0.0
        p95_words = float(word_counts.quantile(0.95)) if not word_counts.empty else 0.0
        long_share = float(((char_lengths >= 200).mean() or 0.0) * 100)
        short_share = float(((char_lengths <= 20).mean() or 0.0) * 100)

        rows.append(
            {
                "Column": column,
                "Rows Analysed": int(cleaned.shape[0]),
                "Avg Char Length": round(avg_chars, 2),
                "Median Char Length": round(median_chars, 2),
                "P95 Char Length": round(p95_chars, 2),
                "Avg Word Count": round(avg_words, 2),
                "Median Word Count": round(median_words, 2),
                "P95 Word Count": round(p95_words, 2),
                "Short Text Share (%)": round(short_share, 1),
                "Long Text Share (%)": round(long_share, 1),
            }
        )

        if long_share >= 40:
            insights_list.append(
                insight(
                    "warning",
                    f"{column}: {long_share:.1f}% of rows exceed 200 characters. Consider truncation or summarisation.",
                )
            )
        elif short_share >= 70:
            insights_list.append(
                insight(
                    "info",
                    f"{column}: {short_share:.1f}% of rows are short snippets (≤20 characters).",
                )
            )

        if idx < 3:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(char_lengths, bins=20, color="#4C51BF", alpha=0.85)
            axes[0].set_title(f"Char length • {column}")
            axes[0].set_xlabel("Characters")
            axes[0].set_ylabel("Rows")

            axes[1].hist(word_counts, bins=20, color="#6B46C1", alpha=0.85)
            axes[1].set_title(f"Word count • {column}")
            axes[1].set_xlabel("Words")
            axes[1].set_ylabel("Rows")

            fig.tight_layout()
            charts.append(
                AnalysisChart(
                    title=f"Text length distribution • {column}",
                    image=fig_to_base64(fig),
                    description="Character and word count histograms for the selected text column.",
                )
            )
            plt.close(fig)

    if not rows:
        return AnalysisResult(
            analysis_id="text_length_distribution",
            title="Text Length Distribution",
            summary="Unable to compute length statistics for the selected columns.",
            status="warning",
            insights=insights_list or [insight("warning", "No text rows available after cleaning.")],
        )

    summary_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        summary_df,
        title="Text Length Summary",
        description="Character and word-length distribution statistics for text columns.",
        round_decimals=2,
    )

    metrics = [
        metric("Text Columns Analysed", len(summary_df)),
        metric("Rows Considered", int(summary_df["Rows Analysed"].sum())),
        metric("Median Avg Words", round(float(summary_df["Avg Word Count"].median()), 2)),
    ]

    return AnalysisResult(
        analysis_id="text_length_distribution",
        title="Text Length Distribution",
        summary="Character and word-length distribution statistics for text columns.",
        tables=[table],
        charts=charts,
        metrics=metrics,
        insights=insights_list,
    )


def _aggregate_token_stats(cleaned: pd.Series, limit_rows: int = 50000) -> Tuple[Counter, Counter, int]:
    if cleaned.shape[0] > limit_rows:
        cleaned = cleaned.sample(limit_rows, random_state=42)

    token_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    total_tokens = 0

    for tokens in _tokenize_series(cleaned):
        token_counter.update(tokens)
        total_tokens += len(tokens)
        if len(tokens) >= 2:
            bigram_counter.update(zip(tokens, tokens[1:]))

    return token_counter, bigram_counter, total_tokens


def text_token_frequency(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    columns = _select_text_columns(df, context)
    if not columns:
        return AnalysisResult(
            analysis_id="text_token_frequency",
            title="Text Token Frequency",
            summary="No text-like columns detected for token frequency analysis.",
            status="warning",
            insights=[insight("warning", "Select columns with textual data to review token frequencies.")],
        )

    rows: List[Dict[str, Any]] = []
    charts: List[AnalysisChart] = []
    insights_list: List = []
    global_token_counter: Counter = Counter()

    for idx, column in enumerate(columns):
        cleaned = _clean_text_series(df[column])
        if cleaned.empty:
            insights_list.append(
                insight("warning", f"Column '{column}' contains no usable text for tokenisation."),
            )
            continue

        token_counter, bigram_counter, total_tokens = _aggregate_token_stats(cleaned)
        if total_tokens == 0:
            insights_list.append(
                insight("warning", f"Column '{column}' did not yield any tokens after cleaning."),
            )
            continue

        global_token_counter.update(token_counter)

        top_tokens = token_counter.most_common(12)
        top_bigrams = bigram_counter.most_common(8)

        for token, count in top_tokens:
            rows.append(
                {
                    "Column": column,
                    "Type": "Token",
                    "Text": token,
                    "Count": int(count),
                    "Share (%)": round((count / total_tokens) * 100, 2),
                }
            )

        for bigram, count in top_bigrams:
            phrase = " ".join(bigram)
            rows.append(
                {
                    "Column": column,
                    "Type": "Phrase",
                    "Text": phrase,
                    "Count": int(count),
                    "Share (%)": round((count / max(total_tokens - 1, 1)) * 100, 2),
                }
            )

        dominant_share = (top_tokens[0][1] / total_tokens) * 100 if top_tokens else 0.0
        if dominant_share >= 35:
            insights_list.append(
                insight(
                    "warning",
                    f"{column}: Top token '{top_tokens[0][0]}' covers {dominant_share:.1f}% of all tokens. Consider normalising or stemming.",
                )
            )
        else:
            insights_list.append(
                insight(
                    "info",
                    f"{column}: Vocabulary is relatively diverse (top token share {dominant_share:.1f}%).",
                )
            )

        if idx < 2 and top_tokens:
            labels, counts = zip(*top_tokens[:10])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(labels, counts, color="#2B6CB0")
            ax.set_title(f"Top tokens • {column}")
            ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            fig.tight_layout()
            charts.append(
                AnalysisChart(
                    title=f"Top tokens • {column}",
                    image=fig_to_base64(fig),
                    description="Most frequent tokens identified in the text column.",
                )
            )
            plt.close(fig)

    if not rows:
        return AnalysisResult(
            analysis_id="text_token_frequency",
            title="Text Token Frequency",
            summary="Token frequency analysis could not be generated for the selected columns.",
            status="warning",
            insights=insights_list or [insight("warning", "Token extraction returned no results.")],
        )

    results_df = pd.DataFrame(rows)
    results_df.sort_values(by=["Column", "Type", "Count"], ascending=[True, True, False], inplace=True)

    table = dataframe_to_table(
        results_df,
        title="Token & Phrase Frequencies",
        description="Top tokens and short phrases detected in each text column.",
        round_decimals=2,
    )

    metrics = [
        metric("Columns Analysed", len({row["Column"] for row in rows})),
        metric("Tokens Catalogued", int(sum(count for _, count in global_token_counter.items()))),
        metric("Distinct Tokens", int(len(global_token_counter))),
    ]

    if global_token_counter:
        labels, counts = zip(*global_token_counter.most_common(10))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, counts, color="#3182CE")
        ax.set_title("Overall top tokens")
        ax.set_ylabel("Count")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        charts.append(
            AnalysisChart(
                title="Dataset-wide top tokens",
                image=fig_to_base64(fig),
                description="Most frequent tokens across all selected text columns.",
            )
        )
        plt.close(fig)

    return AnalysisResult(
        analysis_id="text_token_frequency",
        title="Text Token Frequency",
        summary="Top tokens and short phrases detected across text columns.",
        tables=[table],
        charts=charts,
        metrics=metrics,
        insights=insights_list,
    )


def _compute_language_distribution(cleaned: pd.Series, max_samples: int = 400) -> Tuple[str, float, Dict[str, float], float]:
    if cleaned.empty:
        return "unknown", 0.0, {}, 0.0

    sample = cleaned.sample(min(len(cleaned), max_samples), random_state=42) if len(cleaned) > max_samples else cleaned

    language_counts: Counter[str] = Counter()
    language_confidences: defaultdict[str, List[float]] = defaultdict(list)

    for value in sample:
        try:
            ranked = langid.rank(value)
        except Exception:
            continue

        if not ranked:
            continue

        primary_language, top_score = ranked[0]
        denominator = 1.0
        for _, score in ranked[1:]:
            denominator += math.exp(score - top_score)

        if denominator <= 0.0 or not math.isfinite(denominator):
            continue

        primary_probability = 1.0 / denominator

        language_counts[primary_language] += 1
        language_confidences[primary_language].append(primary_probability)

    if not language_counts:
        return "unknown", 0.0, {}, 0.0

    total_predictions = sum(language_counts.values())
    primary_language, primary_count = language_counts.most_common(1)[0]
    primary_share = primary_count / total_predictions
    avg_primary_confidence = float(np.mean(language_confidences[primary_language])) if language_confidences[primary_language] else 0.0

    distribution_percentages = {
        language: round((count / total_predictions) * 100.0, 1)
        for language, count in language_counts.items()
    }

    return primary_language, primary_share, distribution_percentages, avg_primary_confidence


def text_vocabulary_summary(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    columns = _select_text_columns(df, context)
    if not columns:
        return AnalysisResult(
            analysis_id="text_vocabulary_summary",
            title="Text Vocabulary Summary",
            summary="No text-like columns detected for vocabulary analysis.",
            status="warning",
            insights=[insight("warning", "Select columns with textual data to compute vocabulary metrics.")],
        )

    rows: List[Dict[str, Any]] = []
    insights_list: List = []
    ttr_values: List[float] = []
    primary_languages: List[str] = []
    primary_language_confidences: List[float] = []
    language_distributions: Dict[str, Dict[str, float]] = {}

    for column in columns:
        cleaned = _clean_text_series(df[column])
        if cleaned.empty:
            insights_list.append(
                insight("warning", f"Column '{column}' contains no usable text for vocabulary analysis."),
            )
            continue

        token_counter, _, total_tokens = _aggregate_token_stats(cleaned)
        if total_tokens == 0:
            insights_list.append(
                insight("warning", f"Column '{column}' yielded zero tokens after normalisation."),
            )
            continue

        unique_tokens = len(token_counter)
        hapax_tokens = sum(1 for count in token_counter.values() if count == 1)
        avg_word_length = (
            sum(len(token) * count for token, count in token_counter.items()) / total_tokens
            if total_tokens
            else 0.0
        )

        ttr = unique_tokens / total_tokens if total_tokens else 0.0
        hapax_ratio = hapax_tokens / unique_tokens if unique_tokens else 0.0

        ttr_values.append(ttr)

        primary_language, primary_share, distribution_percentages, avg_primary_confidence = _compute_language_distribution(cleaned)
        primary_languages.append(primary_language)
        primary_language_confidences.append(avg_primary_confidence)
        if distribution_percentages:
            language_distributions[column] = distribution_percentages

        rows.append(
            {
                "Column": column,
                "Rows Analysed": int(cleaned.shape[0]),
                "Total Tokens": int(total_tokens),
                "Unique Tokens": int(unique_tokens),
                "Type-Token Ratio": round(ttr, 3),
                "Hapax Count": int(hapax_tokens),
                "Hapax Ratio": round(hapax_ratio, 3),
                "Avg Word Length": round(avg_word_length, 2),
                "Primary Language": primary_language,
                "Primary Language Share (%)": round(primary_share * 100.0, 1),
                "Primary Language Confidence (%)": round(avg_primary_confidence * 100.0, 1),
            }
        )

        if ttr < 0.25:
            insights_list.append(
                insight(
                    "warning",
                    f"{column}: Low type-token ratio ({ttr:.2f}) suggests repetitive language.",
                )
            )
        elif ttr > 0.6 and total_tokens > 200:
            insights_list.append(
                insight(
                    "info",
                    f"{column}: High lexical diversity detected (TTR {ttr:.2f}).",
                )
            )

        if hapax_ratio > 0.5 and unique_tokens > 50:
            insights_list.append(
                insight(
                    "info",
                    f"{column}: {hapax_ratio:.0%} of unique tokens occur only once—consider stemming or grouping.",
                )
            )

        if primary_language != "unknown" and primary_share < 0.6:
            mix_description = ", ".join(
                f"{lang}: {share:.1f}%" for lang, share in sorted(distribution_percentages.items(), key=lambda item: item[1], reverse=True)
            )
            insights_list.append(
                insight(
                    "warning",
                    f"{column}: Mixed-language content detected (top languages: {mix_description}).",
                )
            )
        elif primary_language != "unknown" and avg_primary_confidence < 0.7:
            insights_list.append(
                insight(
                    "info",
                    f"{column}: Language detection confidence is moderate ({avg_primary_confidence * 100:.1f}%).",
                )
            )

    if not rows:
        return AnalysisResult(
            analysis_id="text_vocabulary_summary",
            title="Text Vocabulary Summary",
            summary="Vocabulary summary could not be generated for the selected columns.",
            status="warning",
            insights=insights_list or [insight("warning", "Vocabulary metrics require non-empty text columns.")],
        )

    summary_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        summary_df,
        title="Vocabulary Metrics",
        description="Token diversity and lexical richness statistics per text column.",
        round_decimals=3,
    )

    metrics = [
        metric("Columns Analysed", len(summary_df)),
        metric("Median Type-Token Ratio", round(float(np.median(ttr_values)), 3)),
        metric(
            "Average Hapax Ratio",
            round(float(summary_df["Hapax Ratio"].mean()), 3),
        ),
    ]

    if primary_languages:
        dominant_language, dominant_count = Counter(primary_languages).most_common(1)[0]
        metrics.append(metric("Most Common Language", dominant_language))
        metrics.append(
            metric(
                "Avg Language Confidence (%)",
                round(float(np.mean(primary_language_confidences)) * 100.0, 1),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_positions = np.arange(len(summary_df))
    ax.bar(x_positions, summary_df["Type-Token Ratio"], color="#805AD5")
    ax.set_ylim(0, min(1.0, max(summary_df["Type-Token Ratio"].max() * 1.1, 0.1)))
    ax.set_ylabel("Type-Token Ratio")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["Column"], rotation=35, ha="right")
    ax.set_title("Lexical diversity by column")
    fig.tight_layout()

    charts = [
        AnalysisChart(
            title="Lexical diversity",
            image=fig_to_base64(fig),
            description="Type-token ratios highlight lexical richness across columns.",
        )
    ]
    plt.close(fig)

    return AnalysisResult(
        analysis_id="text_vocabulary_summary",
        title="Text Vocabulary Summary",
        summary="Lexical diversity and vocabulary richness metrics for text columns.",
        tables=[table],
        charts=charts,
        metrics=metrics,
        insights=insights_list,
        details={"language_distributions": language_distributions} if language_distributions else {},
    )


def text_feature_engineering_profile(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    columns = _select_text_columns(df, context)
    if not columns:
        return AnalysisResult(
            analysis_id="text_feature_engineering_profile",
            title="Text Feature Engineering Profile",
            summary="No text-like columns detected for feature engineering diagnostics.",
            status="warning",
            insights=[
                insight(
                    "warning",
                    "Select columns with textual data to review feature-engineering readiness.",
                )
            ],
        )

    metadata = context.metadata or {}
    sample_limit = _normalize_sample_limit(metadata.get("feature_sample_limit"), FEATURE_SAMPLE_LIMIT)

    rows: List[Dict[str, Any]] = []
    insights_list: List = []
    recommendation_actions: Dict[str, List[str]] = {}
    recommendation_metadata: Dict[str, Dict[str, Any]] = {}
    embedding_recommendations: Dict[str, str] = {}
    language_profiles: Dict[str, Dict[str, Any]] = {}
    median_tokens_values: List[float] = []
    stopword_shares: List[float] = []
    empty_shares: List[float] = []
    embedding_counts: Counter[str] = Counter()

    for column in columns:
        original_series = df[column]
        total_rows = int(original_series.shape[0])
        cleaned = _clean_text_series(original_series)
        non_empty_rows = int(cleaned.shape[0])
        empty_share_pct = (1.0 - (non_empty_rows / total_rows)) * 100.0 if total_rows else 0.0

        if cleaned.empty:
            insights_list.append(
                insight("warning", f"Column '{column}' contains no usable text for feature profiling."),
            )
            continue

        sample = (
            cleaned.sample(sample_limit, random_state=42) if sample_limit is not None and len(cleaned) > sample_limit else cleaned
        )

        primary_language, primary_share, language_distribution, language_confidence = _compute_language_distribution(sample)

        token_lists = list(_tokenize_series(sample))
        token_counts = [len(tokens) for tokens in token_lists]
        total_tokens = sum(token_counts)

        if total_tokens == 0:
            insights_list.append(
                insight("warning", f"Column '{column}' yielded zero tokens after tokenisation."),
            )
            continue

        unique_tokens = len({token for tokens in token_lists for token in tokens})
        stopword_tokens = sum(sum(1 for token in tokens if token in STOP_WORDS) for tokens in token_lists)
        numeric_tokens = sum(sum(1 for token in tokens if token.isdigit()) for tokens in token_lists)
        token_length_sum = sum(len(token) for tokens in token_lists for token in tokens)

        avg_tokens_per_doc = total_tokens / len(token_counts) if token_counts else 0.0
        median_tokens_per_doc = float(np.median(token_counts)) if token_counts else 0.0
        max_tokens_per_doc = float(max(token_counts)) if token_counts else 0.0
        avg_token_length = token_length_sum / total_tokens if total_tokens else 0.0

        concatenated = " ".join(sample.tolist())
        total_chars = len(concatenated)
        alpha_chars = sum(1 for char in concatenated if char.isalpha())
        uppercase_chars = sum(1 for char in concatenated if char.isupper())
        digit_chars = sum(1 for char in concatenated if char.isdigit())
        punctuation_chars = sum(1 for char in concatenated if char in string.punctuation)
        non_ascii_chars = sum(1 for char in concatenated if ord(char) > 127)

        stopword_share_pct = (stopword_tokens / total_tokens) * 100.0 if total_tokens else 0.0
        numeric_token_share_pct = (numeric_tokens / total_tokens) * 100.0 if total_tokens else 0.0
        uppercase_share_pct = (uppercase_chars / alpha_chars) * 100.0 if alpha_chars else 0.0
        digit_char_share_pct = (digit_chars / total_chars) * 100.0 if total_chars else 0.0
        punctuation_share_pct = (punctuation_chars / total_chars) * 100.0 if total_chars else 0.0
        non_ascii_share_pct = (non_ascii_chars / total_chars) * 100.0 if total_chars else 0.0

        recommendations: List[str] = []
        if stopword_share_pct >= 55.0:
            recommendations.append("Remove stopwords or apply TF-IDF weighting")
        if numeric_token_share_pct >= 12.0:
            recommendations.append("Separate numeric tokens or engineer digit ratios")
        if punctuation_share_pct >= 8.0:
            recommendations.append("Strip punctuation or include character n-grams")
        if non_ascii_share_pct >= 5.0:
            recommendations.append("Normalise Unicode and check language-specific tokenisation")
        if median_tokens_per_doc <= 5.0:
            recommendations.append("Short texts: favour character n-grams or sentence embeddings")
        if unique_tokens >= 5000 and total_tokens >= 10000:
            recommendations.append("Use hashing or vocabulary capping to control dimensionality")
        if uppercase_share_pct >= 50.0:
            recommendations.append("Case sensitivity matters: retain casing or engineer uppercase ratios")

        multilanguage_content = bool(language_distribution) and (primary_share * 100.0 < 70.0 or len(language_distribution) >= 2)
        embedding_recommendation = "TF-IDF (word-level)"
        if multilanguage_content or non_ascii_share_pct >= 5.0:
            embedding_recommendation = "Multilingual sentence embeddings (e.g., LaBSE, XLM-R)"
        elif median_tokens_per_doc >= 25.0 or max_tokens_per_doc >= 60.0 or avg_tokens_per_doc >= 25.0:
            embedding_recommendation = "Sentence embeddings (e.g., all-MiniLM)"
        elif median_tokens_per_doc <= 8.0:
            embedding_recommendation = "Character-level TF-IDF / n-grams"

        if not recommendations:
            recommendations.append("Standard word-level TF-IDF with lemmatisation is sufficient")

        recommendations.append(f"Preferred embedding: {embedding_recommendation}")

        recommendation_actions[column] = recommendations
        recommendation_metadata[column] = {
            "actions": recommendations,
            "embedding": embedding_recommendation,
            "primary_language": primary_language,
            "primary_language_share": round(primary_share * 100.0, 1),
            "language_distribution": language_distribution,
            "language_confidence": round(language_confidence * 100.0, 1) if language_confidence else 0.0,
        }
        recommendation_text = "; ".join(recommendations[:3])
        embedding_recommendations[column] = embedding_recommendation
        language_profiles[column] = {
            "primary_language": primary_language,
            "primary_language_share": round(primary_share * 100.0, 1),
            "language_distribution": language_distribution,
            "average_confidence": round(language_confidence * 100.0, 1) if language_confidence else 0.0,
        }
        embedding_counts[embedding_recommendation] += 1

        rows.append(
            {
                "Column": column,
                "Rows Analysed": int(sample.shape[0]),
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
                "Vocabulary Size": int(unique_tokens),
                "Recommended Focus": recommendation_text,
                "Embedding Recommendation": embedding_recommendation,
            }
        )

        median_tokens_values.append(median_tokens_per_doc)
        stopword_shares.append(stopword_share_pct)
        empty_shares.append(empty_share_pct)

        if empty_share_pct >= 40.0:
            insights_list.append(
                insight(
                    "warning",
                    f"{column}: {empty_share_pct:.1f}% of rows are empty after cleaning. Review preprocessing.",
                )
            )
        if stopword_share_pct >= 55.0:
            insights_list.append(
                insight("info", f"{column}: High stopword density ({stopword_share_pct:.1f}%)."),
            )
        if numeric_token_share_pct >= 12.0:
            insights_list.append(
                insight("info", f"{column}: Numeric-heavy text detected ({numeric_token_share_pct:.1f}% tokens)."),
            )
        if non_ascii_share_pct >= 5.0:
            insights_list.append(
                insight("info", f"{column}: Contains multi-language or accented characters ({non_ascii_share_pct:.1f}%)."),
            )
        if median_tokens_per_doc <= 5.0:
            insights_list.append(
                insight("info", f"{column}: Very short documents (median {median_tokens_per_doc:.1f} tokens)."),
            )
        if multilanguage_content:
            top_langs = ", ".join(
                f"{lang}: {pct:.1f}%" for lang, pct in sorted(language_distribution.items(), key=lambda item: item[1], reverse=True)[:3]
            )
            insights_list.append(
                insight(
                    "info",
                    f"{column}: Mixed-language content detected ({top_langs}). Consider multilingual embeddings.",
                )
            )

    if not rows:
        return AnalysisResult(
            analysis_id="text_feature_engineering_profile",
            title="Text Feature Engineering Profile",
            summary="Unable to compute feature-engineering diagnostics for the selected columns.",
            status="warning",
            insights=insights_list or [insight("warning", "Provide text columns with tokenisable content.")],
        )

    summary_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        summary_df,
        title="Feature Engineering Signals",
        description="Token, character, and sparsity signals to guide text feature engineering decisions.",
        round_decimals=2,
    )

    metrics = [
        metric("Columns Analysed", len(summary_df)),
        metric("Median Tokens/Doc", round(float(np.median(median_tokens_values)), 2) if median_tokens_values else 0.0),
        metric("Avg Stopword Share (%)", round(float(np.mean(stopword_shares)), 1) if stopword_shares else 0.0),
        metric("Avg Empty Share (%)", round(float(np.mean(empty_shares)), 1) if empty_shares else 0.0),
        metric(
            "Sentence Embedding Candidates",
            embedding_counts.get("Sentence embeddings (e.g., all-MiniLM)", 0)
            + embedding_counts.get("Multilingual sentence embeddings (e.g., LaBSE, XLM-R)", 0),
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(
        summary_df["Median Tokens/Doc"],
        summary_df["Stopword Share (%)"],
        s=80,
        color="#DD6B20",
        alpha=0.85,
    )
    for _, row in summary_df.iterrows():
        ax.text(row["Median Tokens/Doc"], row["Stopword Share (%)"], row["Column"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Median tokens per document")
    ax.set_ylabel("Stopword share (%)")
    ax.set_title("Stopwords vs. document length")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    charts = [
        AnalysisChart(
            title="Stopword share vs length",
            image=fig_to_base64(fig),
            description="Helps decide when to remove stopwords or switch to character-based features.",
        )
    ]
    plt.close(fig)

    details: Dict[str, Any] = {
        "recommendations": recommendation_actions,
        "recommendation_metadata": recommendation_metadata,
        "embedding_recommendations": embedding_recommendations,
    }
    if language_profiles:
        details["language_profiles"] = language_profiles

    return AnalysisResult(
        analysis_id="text_feature_engineering_profile",
        title="Text Feature Engineering Profile",
        summary="Diagnostics that highlight preprocessing steps before building text features.",
        tables=[table],
        charts=charts,
        metrics=metrics,
        insights=insights_list,
        details=details,
    )


def text_nlp_profile(df: pd.DataFrame, context: AnalysisContext) -> AnalysisResult:
    columns = _select_text_columns(df, context)
    if not columns:
        return AnalysisResult(
            analysis_id="text_nlp_profile",
            title="Advanced NLP Profile",
            summary="No text-like columns detected for NLP analysis.",
            status="warning",
            insights=[insight("warning", "Select columns with textual data to compute NLP metrics.")],
        )

    metadata = context.metadata or {}
    requested_model = metadata.get("nlp_model", "en_core_web_sm")
    nlp_model = _load_spacy_model(requested_model)
    backend_label = "spaCy" if nlp_model is not None and "ner" in getattr(nlp_model, "pipe_names", []) else "pattern"
    sample_limit = _normalize_sample_limit(metadata.get("nlp_sample_limit"), NLP_SAMPLE_LIMIT)

    rows: List[Dict[str, Any]] = []
    insights_list: List = []
    entity_details: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    sentiment_details: Dict[str, Dict[str, Any]] = {}

    for column in columns:
        cleaned = _clean_text_series(df[column])
        if cleaned.empty:
            insights_list.append(
                insight("warning", f"Column '{column}' contains no usable text for NLP profiling."),
            )
            continue

        if sample_limit is not None and len(cleaned) > sample_limit:
            sample = cleaned.sample(sample_limit, random_state=42)
        else:
            sample = cleaned

        entity_counts = _extract_named_entities(sample, nlp_model)
        sentiment_scores, sentiment_categories = _compute_sentiment_distribution(sample)

        total_sentiment = max(sum(sentiment_categories.values()), 1)
        positive_pct = (sentiment_categories.get("positive", 0) / total_sentiment) * 100.0
        negative_pct = (sentiment_categories.get("negative", 0) / total_sentiment) * 100.0
        neutral_pct = (sentiment_categories.get("neutral", 0) / total_sentiment) * 100.0
        avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0

        display_entities: List[str] = []
        entity_record: Dict[str, List[Dict[str, Any]]] = {}
        for label, counter in entity_counts.items():
            if not counter:
                continue
            entity_record[label] = [
                {"text": ent, "count": int(count)} for ent, count in counter.most_common(5)
            ]
            for ent, count in counter.most_common(3):
                display_entities.append(f"{ent} ({label})" if label != "PROPN" else ent)

        entity_details[column] = entity_record
        sentiment_details[column] = {
            "average_score": avg_sentiment,
            "distribution": {
                "positive": round(positive_pct, 1),
                "negative": round(negative_pct, 1),
                "neutral": round(neutral_pct, 1),
            },
        }

        rows.append(
            {
                "Column": column,
                "Texts Analysed": int(sample.shape[0]),
                "Positive (%)": round(positive_pct, 1),
                "Negative (%)": round(negative_pct, 1),
                "Neutral (%)": round(neutral_pct, 1),
                "Avg Sentiment Score": round(avg_sentiment, 3),
                "Top Entities": ", ".join(display_entities[:5]) if display_entities else "None",
            }
        )

        if positive_pct >= 60:
            insights_list.append(
                insight("info", f"{column}: Sentiment is strongly positive ({positive_pct:.1f}% positive)."),
            )
        if negative_pct >= 35:
            insights_list.append(
                insight("warning", f"{column}: Elevated negative sentiment detected ({negative_pct:.1f}% negative)."),
            )
        if not display_entities:
            insights_list.append(
                insight("info", f"{column}: No prominent named entities were identified."),
            )

    if not rows:
        return AnalysisResult(
            analysis_id="text_nlp_profile",
            title="Advanced NLP Profile",
            summary="NLP profile could not be generated for the selected columns.",
            status="warning",
            insights=insights_list or [insight("warning", "NLP profiling requires non-empty text columns.")],
        )

    summary_df = pd.DataFrame(rows)
    table = dataframe_to_table(
        summary_df,
        title="NLP Sentiment & Entity Summary",
        description="Sentiment distribution and top detected entities per text column.",
        round_decimals=3,
    )

    metrics = [
        metric("Columns Analysed", len(summary_df)),
        metric("Median Sentiment Score", round(float(summary_df["Avg Sentiment Score"].median()), 3)),
        metric("Avg Positive Share (%)", round(float(summary_df["Positive (%)"].mean()), 1)),
        metric("Entity Extractor", backend_label),
    ]

    x_positions = np.arange(len(summary_df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x_positions, summary_df["Avg Sentiment Score"], color="#38A169")
    ax.axhline(0.0, color="#2D3748", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Avg Sentiment Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["Column"], rotation=35, ha="right")
    ax.set_title("Average sentiment by column")
    fig.tight_layout()

    charts = [
        AnalysisChart(
            title="Sentiment by column",
            image=fig_to_base64(fig),
            description="Average sentiment scores per text column (positive=optimistic, negative=frustrated).",
        )
    ]
    plt.close(fig)

    details: Dict[str, Any] = {"nlp_backend": backend_label}
    if entity_details:
        details["entities"] = entity_details
    if sentiment_details:
        details["sentiment"] = sentiment_details

    return AnalysisResult(
        analysis_id="text_nlp_profile",
        title="Advanced NLP Profile",
        summary="Sentiment distribution and named entity highlights for text columns.",
        tables=[table],
        charts=charts,
        metrics=metrics,
        insights=insights_list,
        details=details,
    )


__all__ = [
    "text_length_distribution",
    "text_token_frequency",
    "text_vocabulary_summary",
    "text_feature_engineering_profile",
    "text_nlp_profile",
]

"""
Text analysis utilities for EDA and data quality assessment.

This module provides comprehensive text analysis capabilities for exploratory data analysis,
including text categorization, pattern detection, and quality assessment.
"""
import re
import string
import pandas as pd
from typing import Dict, Any, List


def categorize_text_column(
    unique_ratio: float, 
    avg_length: float, 
    avg_words: float, 
    punctuation_pct: float, 
    unique_count: int, 
    total_count: int
) -> str:
    """
    Categorize text columns into different types based on characteristics.
    Improved logic for better NLP candidate detection.
    
    Args:
        unique_ratio: Ratio of unique values to total values
        avg_length: Average character length of text
        avg_words: Average word count per text
        punctuation_pct: Percentage of texts containing punctuation
        unique_count: Number of unique values
        total_count: Total number of values
        
    Returns:
        Text category classification
    """
    # NLP Candidates (free_text) - More flexible detection
    # Case 1: High uniqueness with reasonable text length (reviews, comments, descriptions)
    if unique_ratio > 0.7 and avg_length > 20 and avg_words > 3:
        return "free_text"

    # Case 2: Medium-high uniqueness with longer text (user content, feedback)
    elif unique_ratio > 0.5 and avg_length > 50 and avg_words > 5:
        return "free_text"

    # Case 3: High word count with decent uniqueness (articles, reviews)
    elif avg_words > 10 and unique_ratio > 0.4:
        return "free_text"

    # Case 4: Very long text regardless of uniqueness (descriptions, content)
    elif avg_length > 200 and avg_words > 15:
        return "free_text"

    # Case 5: Text that looks like reviews/comments - moderate length with multiple words and high punctuation
    # This handles cases where reviews might have lower uniqueness due to repetitive content
    elif (
        avg_length > 40
        and avg_words > 5
        and punctuation_pct > 80
        and unique_ratio > 0.15
    ):
        return "free_text"

    # Case 6: Medium length descriptive text with reasonable word count
    elif avg_length > 30 and avg_words > 4 and unique_ratio > 0.2:
        return "free_text"

    # Identifiers - Very high uniqueness with short/structured text
    elif unique_ratio > 0.95 and avg_length < 50:
        return "identifier"

    # Date/time patterns (common identifier pattern) - check for typical date/timestamp format
    elif (
        unique_ratio > 0.8
        and avg_length < 30
        and punctuation_pct > 30
        and avg_words <= 3
    ):
        return "identifier"

    # Categorical - Low uniqueness with short text and few unique values
    elif unique_ratio <= 0.3 and avg_length < 50 and unique_count < 20:
        return "categorical"

    # Also categorical - Very short text with low uniqueness
    elif unique_ratio < 0.3 and avg_length < 20:
        return "categorical"

    # Semi-structured - Mixed patterns (versions, codes, structured data)
    # Exclude potential review text by checking word count
    elif 0.3 <= unique_ratio <= 0.8 and avg_length < 30 and avg_words < 4:
        return "semi_structured"

    # Descriptive text - Medium uniqueness with longer content
    elif 0.4 <= unique_ratio <= 0.8 and avg_length > 30 and avg_words > 5:
        return "descriptive_text"

    # Codes/Labels - Short text with medium uniqueness
    elif avg_length < 20 and unique_count < 200 and avg_words < 3:
        return "codes_labels"

    # Default for mixed content
    else:
        return "mixed_text"


def analyze_text_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Comprehensive text analysis for object/string columns.
    Returns detailed metrics and categorization for EDA purposes.
    
    Args:
        series: Pandas Series containing text data
        column_name: Name of the column being analyzed
        
    Returns:
        Dictionary with comprehensive text analysis metrics
    """
    # Get non-null text values
    text_values = series.dropna().astype(str)

    if len(text_values) == 0:
        return {
            "data_category": "empty_text",
            "text_category": "empty",
            "avg_text_length": 0,
            "min_text_length": 0,
            "max_text_length": 0,
            "text_length_std": 0,
            "total_characters": 0,
            "total_words": 0,
            "avg_words_per_text": 0,
            "contains_punctuation_pct": 0,
            "contains_numbers_pct": 0,
            "all_caps_pct": 0,
            "whitespace_heavy_pct": 0,
        }

    # Calculate text length statistics
    text_lengths = text_values.str.len()
    avg_length = float(text_lengths.mean())
    min_length = int(text_lengths.min())
    max_length = int(text_lengths.max())
    std_length = float(text_lengths.std()) if len(text_lengths) > 1 else 0

    # Word count analysis
    word_counts = text_values.apply(lambda x: len(str(x).split()))
    total_words = int(word_counts.sum())
    avg_words = float(word_counts.mean()) if len(word_counts) > 0 else 0

    # Character analysis
    total_chars = int(text_values.str.len().sum())

    # Pattern analysis
    has_punctuation = text_values.apply(lambda x: bool(re.search(r"[^\w\s]", str(x))))
    has_numbers = text_values.apply(lambda x: bool(re.search(r"\d", str(x))))
    is_all_caps = text_values.apply(lambda x: str(x).isupper() and str(x).isalpha())
    def whitespace_density(text: str) -> float:
        normalized = re.sub(r"[_\-]+", " ", str(text))
        normalized = re.sub(r"\s+", " ", normalized.strip())
        if not normalized:
            return 0.0
        token_count = len(normalized.split())
        return token_count / max(len(normalized), 1)

    is_whitespace_heavy = text_values.apply(lambda x: whitespace_density(x) < 0.1)

    punctuation_pct = float(has_punctuation.mean() * 100)
    numbers_pct = float(has_numbers.mean() * 100)
    all_caps_pct = float(is_all_caps.mean() * 100)
    whitespace_pct = float(is_whitespace_heavy.mean() * 100)

    # Categorize text type
    unique_ratio = series.nunique() / len(series)
    text_category = categorize_text_column(
        unique_ratio,
        avg_length,
        avg_words,
        punctuation_pct,
        series.nunique(),
        len(series),
    )

    # Text quality flags
    quality_flags = []
    if avg_length > 500:
        quality_flags.append("very_long_text")
    if avg_length < 3:
        quality_flags.append("very_short_text")
    if whitespace_pct > 20:
        quality_flags.append("whitespace_heavy")
    if all_caps_pct > 50:
        quality_flags.append("caps_heavy")
    if unique_ratio < 0.01:
        quality_flags.append("highly_repetitive")

    # Additional text metrics
    special_char_pct = float(
        text_values.apply(
            lambda x: sum(1 for c in str(x) if c in string.punctuation)
            / max(len(str(x)), 1)
            * 100
        ).mean()
    )

    # Check for common text patterns
    email_pattern_pct = float(
        text_values.apply(
            lambda x: bool(
                re.search(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", str(x)
                )
            )
        ).mean()
        * 100
    )

    url_pattern_pct = float(
        text_values.apply(
            lambda x: bool(
                re.search(
                    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                    str(x),
                )
            )
        ).mean()
        * 100
    )

    phone_pattern_pct = float(
        text_values.apply(
            lambda x: bool(
                re.search(
                    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                    str(x),
                )
            )
        ).mean()
        * 100
    )

    return {
        "data_category": "text",
        "text_category": text_category,
        "avg_text_length": round(avg_length, 2),
        "min_text_length": min_length,
        "max_text_length": max_length,
        "text_length_std": round(std_length, 2),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_words_per_text": round(avg_words, 2),
        "contains_punctuation_pct": round(punctuation_pct, 2),
        "contains_numbers_pct": round(numbers_pct, 2),
        "all_caps_pct": round(all_caps_pct, 2),
        "whitespace_heavy_pct": round(whitespace_pct, 2),
        "special_char_pct": round(special_char_pct, 2),
        "email_pattern_pct": round(email_pattern_pct, 2),
        "url_pattern_pct": round(url_pattern_pct, 2),
        "phone_pattern_pct": round(phone_pattern_pct, 2),
        "quality_flags": quality_flags,
        "text_length_distribution": {
            "q25": float(text_lengths.quantile(0.25)),
            "q50": float(text_lengths.quantile(0.50)),
            "q75": float(text_lengths.quantile(0.75)),
            "q90": float(text_lengths.quantile(0.90)),
            "q95": float(text_lengths.quantile(0.95)),
        },
    }


def detect_text_patterns(series: pd.Series) -> Dict[str, Any]:
    """
    Detect common patterns in text data for EDA insights.
    
    Args:
        series: Pandas Series containing text data
        
    Returns:
        Dictionary with pattern detection results
    """
    text_values = series.dropna().astype(str)
    
    if len(text_values) == 0:
        return {"patterns": []}
    
    patterns = []
    
    pattern_checks = [
        (
            "email",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            {"case": False},
        ),
        (
            "url",
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            {},
        ),
        (
            "phone",
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            {},
        ),
        (
            "ssn",
            r"\b\d{3}-\d{2}-\d{4}\b",
            {},
        ),
        (
            "credit_card",
            r"\b(?:\d[ -]?){12,18}\d\b",
            {},
        ),
        (
            "ip_address",
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
            {},
        ),
        (
            "date",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            {},
        ),
    ]

    for pattern_type, regex_pattern, options in pattern_checks:
        case_sensitive = options.get("case", True)
        count = text_values.str.contains(
            regex_pattern,
            regex=True,
            case=case_sensitive,
        ).sum()
        if count > 0:
            patterns.append({
                "type": pattern_type,
                "count": int(count),
                "percentage": float(count / len(text_values) * 100),
            })
    
    return {"patterns": patterns}


def get_text_insights(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Get comprehensive text insights for EDA visualization and recommendations.
    
    Args:
        series: Pandas Series containing text data
        column_name: Name of the column being analyzed
        
    Returns:
        Dictionary with EDA-focused text insights
    """
    # Get basic analysis
    analysis = analyze_text_column(series, column_name)
    
    # Get pattern detection
    patterns = detect_text_patterns(series)
    
    # Generate EDA recommendations
    recommendations = []
    
    text_category = analysis.get("text_category", "unknown")
    avg_length = analysis.get("avg_text_length", 0)
    
    if text_category == "free_text":
        recommendations.extend([
            "Consider text preprocessing: tokenization, lowercasing, stop word removal",
            "Suitable for NLP analysis: sentiment analysis, topic modeling, text classification",
            "Consider feature extraction: TF-IDF, word embeddings, n-grams"
        ])
    elif text_category == "categorical":
        recommendations.extend([
            "Consider one-hot encoding or label encoding for ML models",
            "Analyze category distribution and frequency",
            "Check for typos or inconsistent formatting"
        ])
    elif text_category == "identifier":
        recommendations.extend([
            "Consider dropping from ML features (unless needed for joins)",
            "Check for uniqueness and potential data quality issues",
            "May be useful for data linking or debugging"
        ])
    
    if avg_length > 200:
        recommendations.append("Consider text summarization or truncation for analysis")
    
    if analysis.get("quality_flags"):
        recommendations.append(f"Address text quality issues: {', '.join(analysis.get('quality_flags', []))}")
    
    return {
        **analysis,
        **patterns,
        "eda_recommendations": recommendations,
        "nlp_suitability": text_category in ["free_text", "descriptive_text"],
        "requires_preprocessing": text_category in ["free_text", "descriptive_text", "mixed_text"]
    }
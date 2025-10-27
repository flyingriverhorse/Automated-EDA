"""
Domain-specific analysis engine for automated EDA workflows
Provides pre-built analysis pipelines based on data characteristics
"""
import asyncio
import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import json
import pickle
from pathlib import Path

# Statistical and ML imports
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Internal imports
from core.utils.logging_utils import log_data_action
from .cache_manager import CacheManager
from .resource_monitor import ResourceMonitor

GEOSPATIAL_LAT_HINTS: Tuple[str, ...] = ("latitude", "lat", "lat_deg", "latitud")
GEOSPATIAL_LON_HINTS: Tuple[str, ...] = ("longitude", "lon", "lng", "long", "lon_deg")
GEOSPATIAL_TEXT_HINTS: Tuple[str, ...] = (
    "address",
    "city",
    "state",
    "country",
    "zip",
    "zipcode",
    "postal",
    "postcode",
    "region",
    "county",
    "province",
    "territory",
    "geometry",
    "geom",
    "geography",
    "geocode",
    "geocoded",
    "coordinate",
    "coordinates",
    "spatial",
    "geospatial",
    "geopoint",
    "geohash",
    "shape",
    "wkt",
)

ENVIRONMENTAL_HINTS: Tuple[str, ...] = (
    "temperature",
    "temp",
    "humidity",
    "precip",
    "precipitation",
    "rain",
    "rainfall",
    "snow",
    "wind",
    "speed",
    "air_quality",
    "aqi",
    "pm2_5",
    "pm10",
    "emission",
    "emissions",
    "pollution",
    "co2",
    "carbon",
    "ghg",
    "water_quality",
    "water",
    "soil",
    "moisture",
    "pressure",
    "climate",
    "weather",
    "sensor",
    "sensors",
    "environment",
    "environmental",
    "solar",
    "radiation",
    "uv",
    "ozone",
    "visibility",
    "dewpoint",
    "dew_point",
    "turbidity",
    "ph",
)

ENVIRONMENTAL_VALUE_HINTS: Tuple[str, ...] = (
    "°c",
    "°f",
    " ppm",
    " ppb",
    "µg/m3",
    "ug/m3",
    "mg/l",
    "mg\l",
    "aqi",
    "co2",
    "carbon",
    "pm2.5",
    "pm2_5",
    "pm10",
    "humidity",
    "%rh",
    "relative humidity",
    "wind speed",
    "m/s",
    "km/h",
    "knots",
    "solar",
    "uv",
    "ozone",
    "rain",
    "rainfall",
    "precip",
    "hpa",
    "baro",
)


def _column_contains_hint(column: str, hints: Tuple[str, ...]) -> bool:
    if not column:
        return False

    for hint in hints:
        pattern = rf"(?:^|[ _\-]){re.escape(hint)}(?:$|[ _\-])"
        if column == hint or re.search(pattern, column):
            return True
    return False


def _text_contains_hint(text: str, hints: Tuple[str, ...]) -> bool:
    if not text:
        return False
    lower_text = text.lower()
    return any(hint in lower_text for hint in hints)

class DomainAnalyzer:
    """
    Automated domain-specific EDA analysis engine
    """
    
    DEFAULT_ML_MODEL_PATH = (
        Path(__file__).resolve().parents[3]
        / "model_trainings_inuse"
        / "model_ml"
        / "domain_analyzer_xgb.pkl"
    )

    # Analysis workflows by domain
    DOMAIN_WORKFLOWS = {
        "tabular_classification": [
            "basic_statistics",
            "class_distribution",
            "feature_distributions",
            "correlation_analysis",
            "feature_importance",
            "class_separation",
            "outlier_detection"
        ],
        "tabular_regression": [
            "basic_statistics",
            "target_distribution",
            "feature_distributions",
            "correlation_analysis",
            "feature_importance",
            "residual_patterns",
            "outlier_detection"
        ],
        "time_series": [
            "temporal_statistics",
            "trend_analysis",
            "seasonality_detection",
            "autocorrelation",
            "stationarity_tests",
            "decomposition",
            "anomaly_detection"
        ],
        "text_analysis": [
            "text_statistics",
            "vocabulary_analysis",
            "entity_extraction",
            "sentiment_distribution",
            "topic_modeling",
            "text_clustering"
        ],
        "image_analysis": [
            "image_statistics",
            "color_distribution",
            "dimension_analysis",
            "sample_visualization"
        ],
        "marketing_analytics": [
            "campaign_metrics_analysis",
            "conversion_funnel_analysis", 
            "engagement_analysis",
            "channel_performance_analysis",
            "audience_segmentation_analysis",
            "roi_analysis",
            "attribution_analysis",
            "cohort_analysis"
        ],
        "geospatial": [
            "basic_statistics",
            "feature_distributions",
            "correlation_analysis",
            "outlier_detection"
        ],
        "environmental": [
            "basic_statistics",
            "feature_distributions",
            "correlation_analysis",
            "outlier_detection",
            "temporal_statistics",
            "trend_analysis",
            "seasonality_detection"
        ]
    }
    
    def __init__(
        self,
        cache_manager = None,
        resource_monitor = None,
        *,
        enable_ml_classifier: bool = True,
        ml_model_path: Optional[Union[str, Path]] = None,
        ml_min_confidence: float = 0.7
    ):
        self.cache = cache_manager or CacheManager()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.analysis_registry = self._build_analysis_registry()
        self.enable_ml_classifier = enable_ml_classifier
        self.ml_min_confidence = ml_min_confidence
        self.ml_classifier = None
        self.ml_label_encoder = None
        self.ml_feature_names: List[str] = []
        self._ml_artifact_path = None

        if self.enable_ml_classifier:
            self._ml_artifact_path = Path(ml_model_path) if ml_model_path else self.DEFAULT_ML_MODEL_PATH
            self._load_ml_classifier()
        
    def _build_analysis_registry(self) -> Dict[str, callable]:
        """Build registry of analysis functions"""
        return {
            # Basic statistics
            "basic_statistics": self._analyze_basic_statistics,
            "feature_distributions": self._analyze_distributions,
            "correlation_analysis": self._analyze_correlations,
            
            # Classification specific
            "class_distribution": self._analyze_class_distribution,
            "class_separation": self._analyze_class_separation,
            
            # Regression specific
            "target_distribution": self._analyze_target_distribution,
            "residual_patterns": self._analyze_residuals,
            
            # Feature importance
            "feature_importance": self._analyze_feature_importance,
            
            # Outlier detection
            "outlier_detection": self._detect_outliers,
            
            # Time series specific
            "temporal_statistics": self._analyze_temporal_stats,
            "trend_analysis": self._analyze_trends,
            "seasonality_detection": self._detect_seasonality,
            "autocorrelation": self._analyze_autocorrelation,
            "stationarity_tests": self._test_stationarity,
            "decomposition": self._decompose_timeseries,
            
            # Text specific
            "text_statistics": self._analyze_text_stats,
            "vocabulary_analysis": self._analyze_vocabulary,
            "entity_extraction": self._extract_entities,
            "sentiment_distribution": self._analyze_sentiment,
            "topic_modeling": self._model_topics,
            
            # Marketing specific
            "campaign_metrics_analysis": self._analyze_campaign_metrics,
            "conversion_funnel_analysis": self._analyze_conversion_funnel,
            "engagement_analysis": self._analyze_engagement,
            "channel_performance_analysis": self._analyze_channel_performance,
            "audience_segmentation_analysis": self._analyze_audience_segmentation,
            "roi_analysis": self._analyze_roi,
            "attribution_analysis": self._analyze_attribution,
            "cohort_analysis": self._analyze_cohort
        }

    def _load_ml_classifier(self) -> None:
        """Load optional ML-based domain classifier artifact if available."""

        if not self._ml_artifact_path or not self._ml_artifact_path.exists():
            return

        try:
            artifact = self._load_serialized_artifact(self._ml_artifact_path)

            model = None
            label_encoder = None
            feature_names = None

            if isinstance(artifact, dict):
                model = artifact.get("model")
                label_encoder = artifact.get("label_encoder")
                feature_names = artifact.get("feature_names")
            elif isinstance(artifact, (list, tuple)) and len(artifact) >= 3:
                model, label_encoder, feature_names = artifact[:3]

            if model is None or label_encoder is None or not feature_names:
                raise ValueError("Incomplete domain analyzer ML artifact")

            if not hasattr(model, "predict"):
                raise ValueError("Domain analyzer ML model lacks predict method")

            self.ml_classifier = model
            self.ml_label_encoder = label_encoder
            self.ml_feature_names = list(feature_names)

        except Exception as exc:
            log_data_action(
                "DOMAIN_ANALYZER_ML_LOAD_FAILED",
                details=f"path:{self._ml_artifact_path};error:{exc}"
            )
            self.ml_classifier = None
            self.ml_label_encoder = None
            self.ml_feature_names = []

    @staticmethod
    def _load_serialized_artifact(artifact_path: Path):
        """Attempt to load a serialized artifact via joblib, falling back to pickle."""

        try:
            return joblib.load(artifact_path)
        except Exception:
            with open(artifact_path, "rb") as handle:
                return pickle.load(handle)

    def _extract_ml_schema_signature(self, df: pd.DataFrame, max_sample: int = 5000) -> Dict[str, Any]:
        """Replicate schema feature extraction used by the ML classifier."""

        if df.empty:
            return {}

        sampled = df.sample(min(len(df), max_sample), random_state=42)

        features: Dict[str, Any] = {}
        n_rows, n_cols = sampled.shape
        features["n_rows"] = n_rows
        features["n_cols"] = n_cols

        dtype_counts = sampled.dtypes.astype(str).value_counts().to_dict()
        for dtype, count in dtype_counts.items():
            features[f"dtype_{dtype}"] = count / max(n_cols, 1)

        colnames = " ".join(sampled.columns.str.lower())
        for token in [
            "id",
            "date",
            "time",
            "amount",
            "price",
            "score",
            "code",
            "age",
            "lat",
            "long",
            "desc",
            "text",
        ]:
            features[f"col_has_{token}"] = int(token in colnames)

        numeric_cols = sampled.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            features["num_mean_mean"] = float(numeric_cols.mean().mean())
            features["num_mean_std"] = float(numeric_cols.std(ddof=0).mean())
            features["num_skew_mean"] = float(numeric_cols.skew().mean())
            features["num_missing_rate"] = float(numeric_cols.isna().mean().mean())

        categorical_cols = sampled.select_dtypes(include=["object", "category"])
        if not categorical_cols.empty:
            cardinalities = [categorical_cols[col].nunique(dropna=False) for col in categorical_cols.columns]
            features["cat_mean_cardinality"] = float(np.mean(cardinalities))
            features["cat_max_cardinality"] = float(np.max(cardinalities))
            features["cat_missing_rate"] = float(categorical_cols.isna().mean().mean())

        return features

    def _ml_predict_domain(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Return ML-based domain prediction if classifier is available."""

        if (
            not self.enable_ml_classifier
            or self.ml_classifier is None
            or self.ml_label_encoder is None
            or not self.ml_feature_names
            or df.empty
        ):
            return None

        try:
            signature = self._extract_ml_schema_signature(df)
            if not signature:
                return None

            feature_frame = pd.DataFrame([signature], columns=self.ml_feature_names).fillna(0)

            if hasattr(self.ml_classifier, "predict_proba"):
                proba = self.ml_classifier.predict_proba(feature_frame)[0]
                class_index = int(np.argmax(proba))
                confidence = float(proba[class_index])
            else:
                prediction = self.ml_classifier.predict(feature_frame)[0]
                class_index = int(prediction)
                confidence = 1.0
                proba = None

            label = self.ml_label_encoder.inverse_transform([class_index])[0]

            probability_map: Dict[str, float] = {}
            if proba is not None:
                indices = np.arange(len(proba))
                labels = self.ml_label_encoder.inverse_transform(indices)
                probability_map = {label_name: float(score) for label_name, score in zip(labels, proba)}

            return {
                "domain": label,
                "confidence": confidence,
                "probabilities": probability_map,
            }

        except Exception as exc:
            log_data_action("DOMAIN_ANALYZER_ML_PREDICT_FAILED", details=str(exc))
            return None
    
    def analyze_dataset_domain(
        self, 
        columns: List[str], 
        sample_data: List[List[Any]], 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze dataset to determine its domain and characteristics.
        
        Args:
            columns: List of column names
            sample_data: Sample rows of data
            metadata: Additional metadata about the dataset
            
        Returns:
            Dictionary containing domain analysis results
        """
        try:
            # Convert sample data to DataFrame for analysis
            df = pd.DataFrame(sample_data, columns=columns) if sample_data else pd.DataFrame(columns=columns)
            
            # Analyze column names for domain indicators
            column_patterns = self._analyze_column_patterns(columns)
            
            # Analyze data types and patterns
            data_patterns = self._analyze_data_patterns(df) if not df.empty else {}
            
            # Determine primary domain based on patterns
            domain_scores = {
                "healthcare": 0,
                "finance": 0, 
                "retail": 0,
                "real_estate": 0,
                "tech": 0,
                "marketing": 0,
                "time_series": 0,
                "nlp": 0,
                "computer_vision": 0,
                "fraud": 0,
                "geospatial": 0,
                "environmental": 0,
                "general": 1  # baseline score
            }
            
            # Score based on column patterns
            fraud_indicators = 0  # Track number of fraud indicators
            geospatial_indicators = 0
            
            for pattern, score in column_patterns.items():
                if "price" in pattern or "cost" in pattern or "amount" in pattern:
                    domain_scores["finance"] += score
                    domain_scores["retail"] += score * 0.7
                    domain_scores["real_estate"] += score * 0.5
                elif "patient" in pattern or "medical" in pattern or "health" in pattern:
                    domain_scores["healthcare"] += score * 2
                elif "user" in pattern or "click" in pattern or "session" in pattern:
                    domain_scores["tech"] += score
                    domain_scores["marketing"] += score * 0.8
                elif "tech" in pattern or "api" in pattern:
                    domain_scores["tech"] += score * 1.5
                elif "customer" in pattern or "sale" in pattern:
                    domain_scores["retail"] += score
                    domain_scores["marketing"] += score * 0.6
                elif "time" in pattern or "date" in pattern:
                    domain_scores["time_series"] += score * 0.5
                elif "text" in pattern or "comment" in pattern or "review" in pattern:
                    domain_scores["nlp"] += score
                elif "location" in pattern:
                    domain_scores["real_estate"] += score
                    domain_scores["marketing"] += score * 0.4
                    domain_scores["geospatial"] += score * 1.5
                    geospatial_indicators += score
                elif "geospatial" in pattern:
                    domain_scores["geospatial"] += score * 2.5
                    geospatial_indicators += score
                elif "computer_vision" in pattern or "image" in pattern:
                    domain_scores["computer_vision"] += score * 2.0
                elif "marketing" in pattern:
                    # Direct marketing pattern match from pattern_keywords
                    domain_scores["marketing"] += score * 3  # Very strong marketing indicator
                elif "fraud" in pattern:
                    # Direct fraud pattern match from pattern_keywords
                    domain_scores["fraud"] += score * 3  # Very strong fraud indicator
                    fraud_indicators += score
                elif "environmental" in pattern:
                    domain_scores["environmental"] += score * 2.0
                    domain_scores["time_series"] += score * 0.3
                    domain_scores["geospatial"] += score * 0.2
            
            # Boost fraud score if multiple fraud indicators are present
            if fraud_indicators >= 2:
                domain_scores["fraud"] += 3.0  # Significant boost for multiple indicators
            if geospatial_indicators >= 2:
                domain_scores["geospatial"] += 3.0
            
            # Score based on data patterns
            if data_patterns:
                if data_patterns.get("has_time_series", False):
                    domain_scores["time_series"] += 2
                if data_patterns.get("has_text_columns", False):
                    domain_scores["nlp"] += 1
                if data_patterns.get("has_numeric_target", False):
                    domain_scores["finance"] += 1
                    domain_scores["real_estate"] += 1
                if data_patterns.get("has_geospatial_coordinates", False):
                    domain_scores["geospatial"] += 2.5
                if data_patterns.get("has_geospatial_text", False):
                    domain_scores["geospatial"] += 1
                if data_patterns.get("has_image_data", False):
                    domain_scores["computer_vision"] += 2.5
                    domain_scores["tech"] += 0.5
                if data_patterns.get("has_environmental_signals", False):
                    domain_scores["environmental"] += 2.5
                    if data_patterns.get("has_time_series", False):
                        domain_scores["time_series"] += 0.5
                # Fraud-specific data pattern scoring
                if data_patterns.get("has_binary_target", False):
                    # Binary classification often indicates fraud detection
                    domain_scores["fraud"] += 1.5
                if data_patterns.get("has_categorical_features", False) and data_patterns.get("has_numeric_features", False):
                    # Mixed feature types common in fraud detection
                    domain_scores["fraud"] += 0.5
            
            ml_prediction = self._ml_predict_domain(df)
            if ml_prediction:
                ml_domain = ml_prediction["domain"]
                ml_confidence = ml_prediction["confidence"]
                base_sum = max(sum(domain_scores.values()), 1.0)
                domain_scores[ml_domain] = max(
                    domain_scores.get(ml_domain, 0.0),
                    ml_confidence * base_sum
                )

            # Find primary and secondary domains
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            primary_domain = sorted_domains[0][0]
            
            # Get secondary domains with meaningful scores (at least 10% of primary score and > 0.5 absolute)
            primary_score = sorted_domains[0][1]
            secondary_domains = []
            for domain, score in sorted_domains[1:]:
                if score >= max(primary_score * 0.1, 0.5) and len(secondary_domains) < 2:
                    secondary_domains.append({
                        "domain": domain,
                        "confidence": min(score / max(sum(domain_scores.values()), 1), 0.95)
                    })
            
            primary_confidence = min(primary_score / max(sum(domain_scores.values()), 1), 0.95)
            
            # Generate recommendations based on domain
            recommendations = self._generate_domain_recommendations(primary_domain, domain_scores, data_patterns)
            
            result = {
                "primary_domain": primary_domain,
                "primary_confidence": primary_confidence,
                "confidence": primary_confidence,  # Keep for backward compatibility
                "secondary_domains": secondary_domains,
                "domain_scores": domain_scores,
                "patterns": {
                    "column_patterns": column_patterns,
                    "data_patterns": data_patterns
                },
                "recommendations": recommendations,
                "metadata": metadata or {}
            }

            if ml_prediction:
                result["ml_prediction"] = ml_prediction

                if ml_prediction["confidence"] >= self.ml_min_confidence:
                    if ml_prediction["domain"] != result["primary_domain"]:
                        # Insert previous primary domain into secondary list for transparency
                        result["secondary_domains"] = [
                            {
                                "domain": result["primary_domain"],
                                "confidence": result["primary_confidence"]
                            }
                        ] + [
                            entry for entry in result["secondary_domains"]
                            if entry["domain"] != ml_prediction["domain"]
                        ]
                        result["primary_domain"] = ml_prediction["domain"]
                        result["primary_confidence"] = ml_prediction["confidence"]
                        result["confidence"] = ml_prediction["confidence"]
                    else:
                        result["primary_confidence"] = max(
                            result["primary_confidence"], ml_prediction["confidence"]
                        )
                        result["confidence"] = result["primary_confidence"]

            return result
            
        except Exception as e:
            log_data_action("DOMAIN_ANALYSIS_ERROR", details=f"error:{str(e)}")
            return {
                "primary_domain": "general",
                "confidence": 0.5,
                "domain_scores": {"general": 1},
                "patterns": {},
                "recommendations": ["Start with basic exploratory data analysis", "Check for missing values", "Examine data distributions"],
                "error": str(e)
            }
    
    def _generate_domain_recommendations(self, primary_domain: str, domain_scores: Dict[str, float], data_patterns: Dict) -> List[str]:
        """Generate domain-specific recommendations."""
        recommendations = []
        
        if primary_domain == "finance":
            recommendations.extend([
                "Analyze price trends and volatility",
                "Check for outliers in financial metrics",
                "Examine correlations between financial indicators",
                "Consider time-based analysis for trends"
            ])
        elif primary_domain == "healthcare":
            recommendations.extend([
                "Examine patient demographics and outcomes",
                "Check for missing values in medical records",
                "Analyze treatment effectiveness patterns",
                "Consider privacy implications in visualizations"
            ])
        elif primary_domain == "retail":
            recommendations.extend([
                "Analyze customer purchase patterns",
                "Examine seasonal trends in sales",
                "Check product performance metrics",
                "Identify customer segmentation opportunities"
            ])
        elif primary_domain == "real_estate":
            recommendations.extend([
                "Analyze price distributions by location",
                "Examine property features vs. price correlations",
                "Check for geographical clustering patterns",
                "Consider market trend analysis"
            ])
        elif primary_domain == "time_series":
            recommendations.extend([
                "Plot time-based trends and seasonality",
                "Check for autocorrelation patterns",
                "Examine moving averages and trends",
                "Consider forecasting analysis"
            ])
        elif primary_domain == "nlp":
            recommendations.extend([
                "Analyze text length distributions",
                "Examine sentiment patterns if applicable",
                "Check for common words/phrases",
                "Consider text preprocessing needs"
            ])
        elif primary_domain == "marketing":
            recommendations.extend([
                "Analyze campaign performance metrics",
                "Examine customer engagement patterns",
                "Check conversion funnel analysis",
                "Consider A/B testing results"
            ])
        elif primary_domain == "tech":
            recommendations.extend([
                "Review product usage funnels and drop-off points",
                "Monitor latency, error rates, and API performance over time",
                "Segment events by platform or device to uncover stability issues",
                "Correlate feature adoption with retention or engagement metrics"
            ])
        elif primary_domain == "computer_vision":
            recommendations.extend([
                "Visualize sample images alongside labels to verify data quality",
                "Check class balance and annotation coverage across image categories",
                "Inspect image dimensions, channels, and augmentation readiness",
                "Evaluate label consistency (e.g., bounding boxes or masks) before training"
            ])
        elif primary_domain == "fraud":
            recommendations.extend([
                "Inspect class imbalance and consider resampling or class weighting",
                "Review high-risk rule triggers, flags, and investigation outcomes",
                "Analyze temporal or geographic spikes in suspicious activity",
                "Validate feature leakage and ensure robust cross-validation setup"
            ])
        elif primary_domain == "geospatial":
            recommendations.extend([
                "Visualize latitude and longitude points on interactive maps",
                "Validate coordinate reference system consistency",
                "Run geospatial proximity or clustering analyses",
                "Overlay contextual boundaries (e.g., regions, districts) to compare metrics"
            ])
        elif primary_domain == "environmental":
            recommendations.extend([
                "Plot environmental metrics over time to assess trends and anomalies",
                "Monitor threshold breaches for regulatory or safety compliance",
                "Correlate environmental readings with external factors (weather, location, season)",
                "Validate sensor calibration and handle missing telemetry gracefully"
            ])
        else:
            recommendations.extend([
                "Start with basic exploratory data analysis",
                "Check data quality and missing values",
                "Examine variable distributions",
                "Identify correlations between variables"
            ])
        
        # Add general recommendations based on data patterns
        if data_patterns:
            if data_patterns.get("has_missing_values", False):
                recommendations.insert(0, "Address missing values before analysis")
            if data_patterns.get("has_outliers", False):
                recommendations.append("Investigate and handle outliers appropriately")
            if data_patterns.get("high_cardinality_cats", False):
                recommendations.append("Consider grouping high-cardinality categorical variables")
            if data_patterns.get("has_geospatial_coordinates", False):
                geo_rec = "Leverage geospatial analyses (distance, clustering, map visuals) for coordinate columns"
                if geo_rec not in recommendations:
                    recommendations.append(geo_rec)
            elif data_patterns.get("has_geospatial_text", False):
                geo_text_rec = "Geocode address fields to unlock geospatial insights"
                if geo_text_rec not in recommendations:
                    recommendations.append(geo_text_rec)
            if data_patterns.get("has_image_data", False):
                vision_rec = "Audit sample images and labels to confirm annotation quality before modeling"
                if vision_rec not in recommendations:
                    recommendations.append(vision_rec)
            if data_patterns.get("has_environmental_signals", False):
                env_rec = "Track environmental KPI thresholds and set alerts for extreme conditions"
                if env_rec not in recommendations:
                    recommendations.append(env_rec)
        
        return recommendations[:6]  # Return top 6 recommendations

    def _analyze_column_patterns(self, columns: List[str]) -> Dict[str, float]:
        """Analyze column names for domain-specific patterns."""
        patterns = {}
        
        for col in columns:
            col_lower = col.lower()
            
            # Common patterns with scores
            pattern_keywords = {
                "price": ["price", "cost", "amount", "fee", "charge", "value"],
                "time": ["date", "time", "timestamp", "created", "updated"],
                "user": ["user", "customer", "client", "account"],
                "tech": ["event", "click", "session", "device", "browser", "api", "endpoint", "latency", "request", "response", "app", "platform"],
                "medical": ["patient", "diagnosis", "treatment", "medical", "health"],
                "location": ["address", "city", "state", "country", "zip", "location"],
                "id": ["id", "key", "identifier", "uuid"],
                "category": ["type", "category", "class", "group", "segment"],
                "text": ["text", "description", "comment", "review", "message"],
                "marketing": ["campaign", "impression", "click", "conversion", "ctr", "cpc", 
                             "roas", "roi", "spend", "budget", "channel", "source", "medium",
                             "utm", "ad", "audience", "engagement", "funnel", "attribution",
                             "cohort", "retention", "acquisition", "bounce", "session"],
                "fraud": ["fraud", "suspicious", "anomaly", "risk", "flag", "alert", 
                         "transaction", "payment", "transfer", "card", "dispute",
                         "claim", "chargeback", "refund", "verify", "legitimate",
                         "genuine", "authentic", "valid", "invalid", "label"],
                "environmental": [
                    "temperature",
                    "temp",
                    "humidity",
                    "precip",
                    "rain",
                    "rainfall",
                    "wind",
                    "air_quality",
                    "aqi",
                    "pm2_5",
                    "pm10",
                    "emission",
                    "pollution",
                    "co2",
                    "carbon",
                    "ghg",
                    "water",
                    "soil",
                    "moisture",
                    "pressure",
                    "climate",
                    "weather",
                    "sensor",
                    "environment",
                    "solar",
                    "radiation",
                    "uv",
                    "ozone",
                    "dewpoint",
                    "dew_point",
                    "turbidity",
                    "ph",
                ],
                "computer_vision": [
                    "image",
                    "img",
                    "picture",
                    "pixel",
                    "frame",
                    "bounding_box",
                    "bbox",
                    "mask",
                    "segmentation",
                    "annotation",
                    "image_path",
                    "image_url",
                    "filepath",
                    "file_path",
                    "sprite",
                    "tilename",
                    "tile",
                    "width",
                    "height",
                    "channel",
                ]
            }
            
            for pattern, keywords in pattern_keywords.items():
                for keyword in keywords:
                    if keyword in col_lower:
                        patterns[pattern] = patterns.get(pattern, 0) + 1
                        break

            if _column_contains_hint(col_lower, GEOSPATIAL_LAT_HINTS):
                patterns["geospatial"] = patterns.get("geospatial", 0) + 1
            if _column_contains_hint(col_lower, GEOSPATIAL_LON_HINTS):
                patterns["geospatial"] = patterns.get("geospatial", 0) + 1
            if _column_contains_hint(col_lower, GEOSPATIAL_TEXT_HINTS):
                patterns["location"] = patterns.get("location", 0) + 0.5
                patterns["geospatial"] = patterns.get("geospatial", 0) + 0.5
        
        return patterns
    
    def _analyze_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns to identify domain characteristics."""
        if df.empty:
            return {}
        
        patterns = {
            "has_time_series": False,
            "has_text_columns": False,
            "has_numeric_target": False,
            "has_binary_target": False,
            "has_categorical_features": False,
            "has_numeric_features": False,
            "column_count": len(df.columns),
            "row_count": len(df),
            "numeric_columns": 0,
            "categorical_columns": 0,
            "has_geospatial_coordinates": False,
            "has_geospatial_text": False,
            "geospatial_latitude_columns": [],
            "geospatial_longitude_columns": [],
            "geospatial_text_columns": [],
            "has_image_data": False,
            "image_columns": [],
            "has_environmental_signals": False,
            "environmental_columns": [],
        }
        
        for col in df.columns:
            col_lower = str(col).lower()

            if df[col].dtype in ['datetime64[ns]', 'object']:
                # Check if it's a datetime column
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format')
                        pd.to_datetime(df[col].head(10), errors='raise')
                    patterns["has_time_series"] = True
                except:
                    # Check if it's a text column
                    if df[col].dtype == 'object':
                        avg_length = df[col].astype(str).str.len().mean()
                        if avg_length > 20:  # Arbitrary threshold for text
                            patterns["has_text_columns"] = True
                        patterns["categorical_columns"] += 1
                        patterns["has_categorical_features"] = True

                        if _column_contains_hint(col_lower, GEOSPATIAL_TEXT_HINTS) and col not in patterns["geospatial_text_columns"]:
                            patterns["geospatial_text_columns"].append(col)

                        sample_values = df[col].dropna().astype(str).head(5)
                        geometry_prefixes = (
                            "POINT",
                            "LINESTRING",
                            "POLYGON",
                            "MULTIPOINT",
                            "MULTILINESTRING",
                            "MULTIPOLYGON",
                            "GEOMETRYCOLLECTION",
                        )
                        sample_upper = [val.strip().upper() for val in sample_values if isinstance(val, str)]
                        if any(text.startswith(prefix) for text in sample_upper for prefix in geometry_prefixes):
                            if col not in patterns["geospatial_text_columns"]:
                                patterns["geospatial_text_columns"].append(col)
                        elif any(val.strip().startswith("{") and '"type"' in val and '"coordinates"' in val for val in sample_values):
                            if col not in patterns["geospatial_text_columns"]:
                                patterns["geospatial_text_columns"].append(col)

                        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
                        sample_lower = [val.strip().lower() for val in sample_values if isinstance(val, str)]
                        if any(lower.startswith("data:image") for lower in sample_lower) or any(lower.endswith(ext) for lower in sample_lower for ext in image_extensions):
                            patterns["has_image_data"] = True
                            if col not in patterns["image_columns"]:
                                patterns["image_columns"].append(col)
                        elif any("<img" in lower or "image/" in lower for lower in sample_lower):
                            patterns["has_image_data"] = True
                            if col not in patterns["image_columns"]:
                                patterns["image_columns"].append(col)

                        if _column_contains_hint(col_lower, ENVIRONMENTAL_HINTS) and col not in patterns["environmental_columns"]:
                            patterns["environmental_columns"].append(col)
                            patterns["has_environmental_signals"] = True
                        elif any(_text_contains_hint(lower, ENVIRONMENTAL_VALUE_HINTS) for lower in sample_lower):
                            if col not in patterns["environmental_columns"]:
                                patterns["environmental_columns"].append(col)
                            patterns["has_environmental_signals"] = True

                        # Check for binary target (common in fraud detection)
                        unique_values = df[col].nunique()
                        if unique_values == 2 and col.lower() in ['target', 'fraud', 'is_fraud', 'label', 'class', 'y']:
                            patterns["has_binary_target"] = True
                            
            elif df[col].dtype in ['int64', 'float64']:
                patterns["numeric_columns"] += 1
                patterns["has_numeric_features"] = True
                
                # Check if this could be a target variable
                if col.lower() in ['target', 'price', 'value', 'amount', 'score']:
                    patterns["has_numeric_target"] = True
                    
                # Check for binary numeric target (0/1)
                unique_values = df[col].nunique()
                if unique_values == 2 and set(df[col].unique()).issubset({0, 1, 0.0, 1.0}) and col.lower() in ['target', 'fraud', 'is_fraud', 'label', 'class', 'y']:
                    patterns["has_binary_target"] = True

                if _column_contains_hint(col_lower, GEOSPATIAL_LAT_HINTS) and col not in patterns["geospatial_latitude_columns"]:
                    patterns["geospatial_latitude_columns"].append(col)
                if _column_contains_hint(col_lower, GEOSPATIAL_LON_HINTS) and col not in patterns["geospatial_longitude_columns"]:
                    patterns["geospatial_longitude_columns"].append(col)

                if any(keyword in col_lower for keyword in ["image", "pixel", "frame", "bbox", "mask"]):
                    patterns["has_image_data"] = True
                    if col not in patterns["image_columns"]:
                        patterns["image_columns"].append(col)

                if _column_contains_hint(col_lower, ENVIRONMENTAL_HINTS):
                    if col not in patterns["environmental_columns"]:
                        patterns["environmental_columns"].append(col)
                    patterns["has_environmental_signals"] = True

        if patterns["geospatial_latitude_columns"] and patterns["geospatial_longitude_columns"]:
            patterns["has_geospatial_coordinates"] = True
        if patterns["geospatial_text_columns"]:
            patterns["has_geospatial_text"] = True
        if patterns["image_columns"]:
            patterns["has_image_data"] = True
        if patterns["environmental_columns"]:
            patterns["has_environmental_signals"] = True
        
        return patterns
    
    async def analyze(
        self,
        session_id: str,
        data: pd.DataFrame,
        domain_type: str,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute domain-specific analysis workflow
        
        Args:
            session_id: EDA session identifier
            data: DataFrame to analyze
            domain_type: Type of domain analysis
            analysis_config: Optional configuration
            
        Returns:
            Analysis results with visualizations
        """
        # Check cache first
        cache_key = f"domain_{session_id}_{domain_type}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Check resources
        if not await self.resource_monitor.check_available_resources():
            raise ResourceError("Insufficient resources for analysis")
        
        # Get workflow
        workflow = self.DOMAIN_WORKFLOWS.get(domain_type, [])
        if not workflow:
            raise ValueError(f"Unknown domain type: {domain_type}")
        
        results = {
            "domain_type": domain_type,
            "timestamp": datetime.now().isoformat(),
            "analyses": {},
            "visualizations": {},
            "insights": [],
            "recommendations": []
        }
        
        # Execute workflow steps
        for step in workflow:
            if step in self.analysis_registry:
                try:
                    step_result = await self._execute_analysis_step(
                        step, data, analysis_config
                    )
                    results["analyses"][step] = step_result["analysis"]
                    if "visualizations" in step_result:
                        results["visualizations"][step] = step_result["visualizations"]
                    if "insights" in step_result:
                        results["insights"].extend(step_result["insights"])
                except Exception as e:
                    log_data_action(
                        "DOMAIN_ANALYSIS_ERROR",
                        success=False,
                        details=f"Step {step} failed: {str(e)}"
                    )
                    results["analyses"][step] = {"error": str(e)}
        
        # Generate overall insights and recommendations
        results["insights"].extend(self._generate_insights(results["analyses"]))
        results["recommendations"].extend(
            self._generate_recommendations(domain_type, results["analyses"])
        )
        
        # Cache results
        await self.cache.set(cache_key, results, ttl=3600)
        
        return results
    
    async def _execute_analysis_step(
        self,
        step: str,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute single analysis step with resource monitoring"""
        with self.resource_monitor.track_operation(step):
            analysis_func = self.analysis_registry[step]
            
            # Run analysis (potentially CPU-intensive)
            if asyncio.iscoroutinefunction(analysis_func):
                result = await analysis_func(data, config)
            else:
                # Run in executor for blocking operations
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, analysis_func, data, config
                )
            
            return result
    
    # Basic Statistics Analysis
    def _analyze_basic_statistics(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comprehensive basic statistics"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        analysis = {
            "shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
            "numeric_summary": data[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            "categorical_summary": {},
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "data_types": data.dtypes.astype(str).to_dict()
        }
        
        # Categorical statistics
        for col in categorical_cols:
            analysis["categorical_summary"][col] = {
                "unique_values": data[col].nunique(),
                "top_values": data[col].value_counts().head(10).to_dict(),
                "mode": data[col].mode()[0] if not data[col].mode().empty else None
            }
        
        # Create visualizations
        visualizations = []
        
        # Missing values heatmap
        if data.isnull().sum().sum() > 0:
            fig = px.imshow(
                data.isnull().astype(int),
                title="Missing Values Pattern",
                labels=dict(x="Features", y="Samples", color="Missing"),
                color_continuous_scale="RdYlBu_r"
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig.to_json(),
                "title": "Missing Values Heatmap"
            })
        
        insights = []
        if data.duplicated().sum() > 0:
            insights.append(f"Found {data.duplicated().sum()} duplicate rows")
        
        high_missing = [col for col, count in analysis["missing_values"].items() 
                       if count > len(data) * 0.3]
        if high_missing:
            insights.append(f"High missing values in: {', '.join(high_missing)}")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    # Distribution Analysis
    def _analyze_distributions(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature distributions"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:20]  # Limit to 20
        
        analysis = {}
        visualizations = []
        
        for col in numeric_cols:
            # Statistical tests
            col_data = data[col].dropna()
            
            # Normality test
            if len(col_data) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000])
                skewness = col_data.skew()
                kurtosis = col_data.kurtosis()
                
                analysis[col] = {
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "is_normal": shapiro_p > 0.05,
                    "outliers_iqr": self._count_outliers_iqr(col_data)
                }
        
        # Create distribution plots
        if len(numeric_cols) > 0:
            fig = make_subplots(
                rows=(len(numeric_cols) + 2) // 3,
                cols=3,
                subplot_titles=[col for col in numeric_cols]
            )
            
            for idx, col in enumerate(numeric_cols):
                row = idx // 3 + 1
                col_idx = idx % 3 + 1
                
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, showlegend=False),
                    row=row, col=col_idx
                )
            
            fig.update_layout(height=300 * ((len(numeric_cols) + 2) // 3))
            visualizations.append({
                "type": "plotly",
                "figure": fig.to_json(),
                "title": "Feature Distributions"
            })
        
        # Generate insights
        insights = []
        skewed_features = [col for col, stats in analysis.items() 
                          if abs(stats.get("skewness", 0)) > 1]
        if skewed_features:
            insights.append(f"Highly skewed features: {', '.join(skewed_features)}")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    # Correlation Analysis
    def _analyze_correlations(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature correlations"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "analysis": {"message": "Not enough numeric columns for correlation"},
                "visualizations": [],
                "insights": []
            }
        
        # Compute correlations
        corr_matrix = data[numeric_cols].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })
        
        analysis = {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_corr_pairs,
            "average_correlation": corr_matrix.abs().mean().mean()
        }
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        
        visualizations = [{
            "type": "plotly",
            "figure": fig.to_json(),
            "title": "Correlation Heatmap"
        }]
        
        insights = []
        if high_corr_pairs:
            insights.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    # Feature Importance Analysis
    def _analyze_feature_importance(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature importance using Random Forest"""
        target_col = config.get("target_column") if config else None
        
        if not target_col or target_col not in data.columns:
            return {
                "analysis": {"error": "No target column specified"},
                "visualizations": [],
                "insights": []
            }
        
        # Prepare data
        X = data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
        y = data[target_col]
        
        if X.empty:
            return {
                "analysis": {"error": "No numeric features found"},
                "visualizations": [],
                "insights": []
            }
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Determine problem type
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        # Train model and get importance
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        try:
            model.fit(X, y)
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            analysis = {
                "feature_importance": importance.to_dict('records'),
                "top_features": importance.head(10)['feature'].tolist(),
                "problem_type": "classification" if is_classification else "regression"
            }
            
            # Create importance plot
            fig = px.bar(
                importance.head(20),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 20 Feature Importance"
            )
            
            visualizations = [{
                "type": "plotly",
                "figure": fig.to_json(),
                "title": "Feature Importance"
            }]
            
            insights = [
                f"Top 3 important features: {', '.join(importance.head(3)['feature'].tolist())}"
            ]
            
        except Exception as e:
            analysis = {"error": f"Failed to compute importance: {str(e)}"}
            visualizations = []
            insights = []
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    # Outlier Detection
    def _detect_outliers(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        outlier_summary = {}
        total_outliers = set()
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound))
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            z_outliers = z_scores > 3
            
            outlier_indices = data.index[iqr_outliers | z_outliers].tolist()
            total_outliers.update(outlier_indices)
            
            outlier_summary[col] = {
                "iqr_outliers": int(iqr_outliers.sum()),
                "z_score_outliers": int(z_outliers.sum()),
                "outlier_percentage": len(outlier_indices) / len(data) * 100
            }
        
        analysis = {
            "outlier_summary": outlier_summary,
            "total_outlier_rows": len(total_outliers),
            "outlier_percentage": len(total_outliers) / len(data) * 100
        }
        
        # Create outlier visualization
        outlier_counts = pd.DataFrame([
            {"feature": col, "outliers": stats["iqr_outliers"]}
            for col, stats in outlier_summary.items()
        ]).sort_values("outliers", ascending=False).head(20)
        
        fig = px.bar(
            outlier_counts,
            x="feature",
            y="outliers",
            title="Outlier Count by Feature (IQR Method)"
        )
        
        visualizations = [{
            "type": "plotly",
            "figure": fig.to_json(),
            "title": "Outlier Distribution"
        }]
        
        insights = []
        high_outlier_features = [
            col for col, stats in outlier_summary.items()
            if stats["outlier_percentage"] > 5
        ]
        if high_outlier_features:
            insights.append(f"High outlier features (>5%): {', '.join(high_outlier_features[:5])}")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    # Helper methods
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _generate_insights(self, analyses: Dict[str, Any]) -> List[str]:
        """Generate overall insights from analyses"""
        insights = []
        
        # Data quality insights
        if "basic_statistics" in analyses:
            stats = analyses["basic_statistics"]
            if stats.get("duplicate_rows", 0) > 0:
                insights.append("⚠️ Dataset contains duplicate rows")
            
            missing = stats.get("missing_values", {})
            if any(v > 0 for v in missing.values()):
                insights.append("⚠️ Missing values detected - consider imputation strategy")
        
        # Distribution insights
        if "feature_distributions" in analyses:
            dist = analyses["feature_distributions"]
            non_normal = [k for k, v in dist.items() if not v.get("is_normal", True)]
            if non_normal:
                insights.append(f"📊 Non-normal distributions detected in {len(non_normal)} features")
        
        # Correlation insights
        if "correlation_analysis" in analyses:
            corr = analyses["correlation_analysis"]
            if corr.get("high_correlations"):
                insights.append("🔗 High correlations detected - consider feature selection")
        
        return insights
    
    def _generate_recommendations(
        self,
        domain_type: str,
        analyses: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # General recommendations
        if "outlier_detection" in analyses:
            outliers = analyses["outlier_detection"]
            if outliers.get("outlier_percentage", 0) > 5:
                recommendations.append(
                    "Consider outlier treatment: capping, transformation, or removal"
                )
        
        # Domain-specific recommendations
        if domain_type == "tabular_classification":
            if "class_distribution" in analyses:
                # Check for imbalanced classes
                recommendations.append(
                    "For imbalanced classes, consider SMOTE, class weights, or ensemble methods"
                )
        
        elif domain_type == "tabular_regression":
            recommendations.append(
                "Check residual patterns for heteroscedasticity and non-linearity"
            )
        
        elif domain_type == "time_series":
            recommendations.append(
                "Ensure stationarity before modeling - consider differencing or detrending"
            )
        
        # Feature engineering recommendations
        if "feature_distributions" in analyses:
            recommendations.append(
                "Consider log transformation for skewed features"
            )
        
        if "correlation_analysis" in analyses:
            corr = analyses["correlation_analysis"]
            if corr.get("high_correlations"):
                recommendations.append(
                    "Use PCA or feature selection to handle multicollinearity"
                )
        
        return recommendations

    # Classification-specific methods
    def _analyze_class_distribution(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze target class distribution for classification"""
        target_col = config.get("target_column") if config else None
        
        if not target_col or target_col not in data.columns:
            return {"analysis": {"error": "No target column specified"}}
        
        class_counts = data[target_col].value_counts()
        class_proportions = data[target_col].value_counts(normalize=True)
        
        analysis = {
            "class_counts": class_counts.to_dict(),
            "class_proportions": class_proportions.to_dict(),
            "n_classes": len(class_counts),
            "majority_class": class_counts.index[0],
            "minority_class": class_counts.index[-1],
            "imbalance_ratio": class_counts.max() / class_counts.min()
        }
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Class Distribution", "Class Proportions"],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        fig.add_trace(
            go.Bar(x=class_counts.index.astype(str), y=class_counts.values, name="Count"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(labels=class_counts.index.astype(str), values=class_counts.values),
            row=1, col=2
        )
        
        visualizations = [{
            "type": "plotly",
            "figure": fig.to_json(),
            "title": "Target Class Distribution"
        }]
        
        insights = []
        if analysis["imbalance_ratio"] > 10:
            insights.append("⚠️ Severe class imbalance detected")
        elif analysis["imbalance_ratio"] > 3:
            insights.append("⚠️ Moderate class imbalance detected")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    def _analyze_class_separation(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze feature separation between classes"""
        target_col = config.get("target_column") if config else None
        
        if not target_col:
            return {"analysis": {"error": "No target column specified"}}
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        numeric_features = numeric_features.drop(target_col, errors='ignore')
        
        separation_scores = {}
        
        for feature in numeric_features[:20]:  # Limit to 20 features
            # Calculate separation using ANOVA F-statistic
            groups = [data[data[target_col] == cls][feature].dropna() 
                      for cls in data[target_col].unique()]
            
            if all(len(g) > 0 for g in groups):
                f_stat, p_value = stats.f_oneway(*groups)
                separation_scores[feature] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        # Sort by F-statistic
        best_features = sorted(
            separation_scores.items(),
            key=lambda x: x[1]["f_statistic"],
            reverse=True
        )[:10]
        
        analysis = {
            "separation_scores": separation_scores,
            "best_separating_features": [f[0] for f in best_features],
            "n_significant_features": sum(
                1 for v in separation_scores.values() if v["significant"]
            )
        }
        
        return {
            "analysis": analysis,
            "visualizations": [],
            "insights": [
                f"Found {analysis['n_significant_features']} features with significant class separation"
            ]
        }
    
    # Regression-specific methods
    def _analyze_target_distribution(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze target distribution for regression"""
        target_col = config.get("target_column") if config else None
        
        if not target_col or target_col not in data.columns:
            return {"analysis": {"error": "No target column specified"}}
        
        target_data = data[target_col].dropna()
        
        analysis = {
            "mean": target_data.mean(),
            "median": target_data.median(),
            "std": target_data.std(),
            "min": target_data.min(),
            "max": target_data.max(),
            "skewness": target_data.skew(),
            "kurtosis": target_data.kurtosis(),
            "outliers": self._count_outliers_iqr(target_data)
        }
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Distribution", "Box Plot", "Q-Q Plot", "Violin Plot"]
        )
        
        fig.add_trace(
            go.Histogram(x=target_data, nbinsx=50),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=target_data, name="Target"),
            row=1, col=2
        )
        
        # Q-Q plot
        theoretical_quantiles = np.random.normal(0, 1, len(target_data))
        theoretical_quantiles.sort()
        target_sorted = np.sort(target_data)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=target_sorted, mode='markers'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Violin(y=target_data, name="Target"),
            row=2, col=2
        )
        
        visualizations = [{
            "type": "plotly",
            "figure": fig.to_json(),
            "title": "Target Distribution Analysis"
        }]
        
        insights = []
        if abs(analysis["skewness"]) > 1:
            insights.append(f"Target is {'right' if analysis['skewness'] > 0 else 'left'} skewed")
        if analysis["outliers"] > len(target_data) * 0.05:
            insights.append("Target contains significant outliers")
        
        return {
            "analysis": analysis,
            "visualizations": visualizations,
            "insights": insights
        }
    
    def _analyze_residuals(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze residual patterns (placeholder for actual model residuals)"""
        # This would require a fitted model
        # For now, return a placeholder
        return {
            "analysis": {
                "note": "Residual analysis requires a fitted model",
                "recommendation": "Train a baseline model first"
            },
            "visualizations": [],
            "insights": []
        }
    
    # Time series methods (simplified versions)
    def _analyze_temporal_stats(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic temporal statistics"""
        # Implement time series specific statistics
        return {
            "analysis": {"placeholder": "Time series analysis"},
            "visualizations": [],
            "insights": []
        }
    
    def _analyze_trends(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trend analysis for time series"""
        return {
            "analysis": {"placeholder": "Trend analysis"},
            "visualizations": [],
            "insights": []
        }
    
    def _detect_seasonality(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Seasonality detection"""
        return {
            "analysis": {"placeholder": "Seasonality detection"},
            "visualizations": [],
            "insights": []
        }
    
    def _analyze_autocorrelation(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Autocorrelation analysis"""
        return {
            "analysis": {"placeholder": "Autocorrelation"},
            "visualizations": [],
            "insights": []
        }
    
    def _test_stationarity(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stationarity tests"""
        return {
            "analysis": {"placeholder": "Stationarity tests"},
            "visualizations": [],
            "insights": []
        }
    
    def _decompose_timeseries(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Time series decomposition"""
        return {
            "analysis": {"placeholder": "Decomposition"},
            "visualizations": [],
            "insights": []
        }
    
    # Text analysis methods (simplified)
    def _analyze_text_stats(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Text statistics"""
        return {
            "analysis": {"placeholder": "Text statistics"},
            "visualizations": [],
            "insights": []
        }
    
    def _analyze_vocabulary(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Vocabulary analysis"""
        return {
            "analysis": {"placeholder": "Vocabulary analysis"},
            "visualizations": [],
            "insights": []
        }
    
    def _extract_entities(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Entity extraction"""
        return {
            "analysis": {"placeholder": "Entity extraction"},
            "visualizations": [],
            "insights": []
        }
    
    def _analyze_sentiment(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sentiment analysis"""
        return {
            "analysis": {"placeholder": "Sentiment analysis"},
            "visualizations": [],
            "insights": []
        }
    
    def _model_topics(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Topic modeling"""
        return {
            "analysis": {"placeholder": "Topic modeling"},
            "visualizations": [],
            "insights": []
        }
    
    # Marketing analysis methods (placeholders that delegate to granular components)
    def _analyze_campaign_metrics(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Campaign metrics analysis"""
        return {
            "analysis": {"placeholder": "Campaign metrics analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use CampaignMetricsAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_conversion_funnel(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Conversion funnel analysis"""
        return {
            "analysis": {"placeholder": "Conversion funnel analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use ConversionFunnelAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_engagement(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Engagement analysis"""
        return {
            "analysis": {"placeholder": "Engagement analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use EngagementAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_channel_performance(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Channel performance analysis"""
        return {
            "analysis": {"placeholder": "Channel performance analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use ChannelPerformanceAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_audience_segmentation(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Audience segmentation analysis"""
        return {
            "analysis": {"placeholder": "Audience segmentation analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use AudienceSegmentationAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_roi(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ROI analysis"""
        return {
            "analysis": {"placeholder": "ROI analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use ROIAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_attribution(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Attribution analysis"""
        return {
            "analysis": {"placeholder": "Attribution analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use AttributionAnalysis granular component for detailed analysis"]
        }
    
    def _analyze_cohort(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cohort analysis"""
        return {
            "analysis": {"placeholder": "Cohort analysis - use granular components"},
            "visualizations": [],
            "insights": ["Use CohortAnalysis granular component for detailed analysis"]
        }


class ResourceError(Exception):
    """Exception raised when resources are insufficient"""
    pass
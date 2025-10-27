"""
Enhanced Notebook Template Manager for EDA
Provides pre-built analysis templates based on Kaggle examples and domain expertise
Supports both automated domain-specific analysis and LLM-generated notebooks
"""
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import nbformat
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.utils.logging_utils import log_data_action


class DataDomain(Enum):
    """Data domain classifications based on Kaggle examples analysis"""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    REAL_ESTATE = "real_estate"
    TECH = "tech"
    MARKETING = "marketing"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    FRAUD = "fraud"
    GENERAL = "general"


class AnalysisComplexity(Enum):
    """Analys                "insights": [
                    {
                        "type": "info",
                        "icon": "info-circle",
                        "title": "Dataset Overview",
                        "description": f"Dataset contains {len(columns)} columns and {len(sample_data)} sample rows"
                    },
                    {
                        "type": "success", 
                        "icon": "check-circle",
                        "title": "Template Selection",
                        "description": f"Template '{template_info.get('name', template_name)}' is appropriate for this analysis"
                    },
                    {
                        "type": "info",
                        "icon": "gear",
                        "title": "Analysis Complexity",
                        "description": f"Analysis complexity: {self._get_complexity_string(template_info.get('complexity', 'intermediate'))}"
                    },
                    {
                        "type": "warning",
                        "icon": "exclamation-triangle", 
                        "title": "Template Preview",
                        "description": "This is a generated analysis template - results will be more detailed once analysis is executed"
                    }
                ],plexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DatasetProfile:
    """Dataset characteristics for template selection"""
    rows: int
    columns: int
    numeric_cols: int
    categorical_cols: int
    text_cols: int
    datetime_cols: int
    missing_percentage: float
    target_type: Optional[str] = None  # 'binary', 'multiclass', 'regression', 'unsupervised'
    domain: Optional[DataDomain] = None
    has_time_component: bool = False
    is_balanced: Optional[bool] = None


class NotebookTemplateManager:
    """
    Enhanced template manager with domain expertise and LLM integration
    Based on analysis of 141+ Kaggle notebooks from data_science_examples
    """
    
    # Enhanced template categories based on Kaggle analysis
    TEMPLATE_CATEGORIES = {
        "basic_overview": {
            "name": "Basic Data Overview",
            "description": "Simple descriptive statistics and data quality check",
            "domains": [DataDomain.GENERAL],
            "complexity": AnalysisComplexity.BASIC,
            "sections": [
                "data_overview",
                "basic_statistics", 
                "missing_values_check",
                "data_types_summary",
                "simple_visualizations"
            ],
            "kaggle_patterns": [
                "exploratory-data-analysis",
                "basic-data-exploration"
            ]
        },
        "descriptive_analysis": {
            "name": "Descriptive Statistical Analysis",
            "description": "Core statistical measures and distributions",
            "domains": [DataDomain.GENERAL, DataDomain.RETAIL, DataDomain.MARKETING],
            "complexity": AnalysisComplexity.BASIC,
            "sections": [
                "data_overview",
                "descriptive_statistics",
                "distribution_analysis", 
                "basic_correlations",
                "outlier_detection",
                "summary_insights"
            ],
            "kaggle_patterns": [
                "statistical-analysis",
                "data-distribution-analysis"
            ]
        },
        "classification": {
            "name": "Classification Analysis",
            "description": "Binary and multi-class classification problems",
            "domains": [DataDomain.HEALTHCARE, DataDomain.FINANCE, DataDomain.MARKETING],
            "complexity": AnalysisComplexity.INTERMEDIATE,
            "sections": [
                "data_overview",
                "target_analysis", 
                "feature_exploration",
                "correlation_analysis",
                "class_imbalance_check",
                "feature_importance",
                "visualization_suite",
                "preprocessing_recommendations",
                "model_recommendations"
            ],
            "kaggle_patterns": [
                "heart-failure-prediction",
                "diabetes-eda-random-forest",
                "credit-card-fraud-detection",
                "titanic-analysis"
            ]
        },
        "regression": {
            "name": "Regression Analysis", 
            "description": "Continuous target prediction problems",
            "domains": [DataDomain.REAL_ESTATE, DataDomain.FINANCE, DataDomain.RETAIL],
            "complexity": AnalysisComplexity.INTERMEDIATE,
            "sections": [
                "data_overview",
                "target_distribution",
                "feature_relationships", 
                "correlation_analysis",
                "outlier_analysis",
                "normality_testing",
                "feature_engineering_ideas",
                "model_recommendations"
            ],
            "kaggle_patterns": [
                "house-prices-solution",
                "boston-house-price-prediction",
                "comprehensive-data-exploration"
            ]
        },
        "time_series": {
            "name": "Time Series Analysis",
            "description": "Temporal data exploration and forecasting",
            "domains": [DataDomain.FINANCE, DataDomain.RETAIL, DataDomain.TIME_SERIES],
            "complexity": AnalysisComplexity.ADVANCED,
            "sections": [
                "temporal_overview",
                "trend_analysis",
                "seasonality_detection", 
                "stationarity_testing",
                "decomposition",
                "autocorrelation_analysis",
                "forecasting_recommendations"
            ],
            "kaggle_patterns": [
                "bitcoin-time-series-forecasting",
                "store-sales-analysis-time-serie", 
                "walmart-sales-forecasting"
            ]
        },
        "nlp": {
            "name": "Natural Language Processing",
            "description": "Text data analysis and processing", 
            "domains": [DataDomain.NLP, DataDomain.MARKETING, DataDomain.TECH],
            "complexity": AnalysisComplexity.ADVANCED,
            "sections": [
                "text_overview",
                "vocabulary_analysis",
                "length_distribution", 
                "sentiment_analysis",
                "topic_modeling_prep",
                "preprocessing_pipeline",
                "embedding_exploration",
                "model_recommendations"
            ],
            "kaggle_patterns": [
                "nlp-with-disaster-tweets",
                "twitter-sentiment-analysis",
                "text-modelling-in-pytorch"
            ]
        },
        "healthcare": {
            "name": "Healthcare Data Analysis",
            "description": "Medical and health-related data exploration",
            "domains": [DataDomain.HEALTHCARE],
            "complexity": AnalysisComplexity.EXPERT,
            "sections": [
                "clinical_data_overview",
                "patient_demographics",
                "biomarker_analysis",
                "survival_analysis_prep",
                "risk_factor_analysis", 
                "medical_imaging_prep",
                "regulatory_compliance_check",
                "clinical_recommendations"
            ],
            "kaggle_patterns": [
                "diabetes-eda-random-forest",
                "heart-failure-prediction", 
                "breast-cancer",
                "lung-cancer-analysis"
            ]
        },
        "finance": {
            "name": "Financial Data Analysis",
            "description": "Financial modeling and risk analysis",
            "domains": [DataDomain.FINANCE],
            "complexity": AnalysisComplexity.EXPERT,
            "sections": [
                "financial_overview",
                "risk_assessment",
                "fraud_detection_prep",
                "portfolio_analysis",
                "volatility_analysis",
                "correlation_networks",
                "regulatory_metrics", 
                "model_recommendations"
            ],
            "kaggle_patterns": [
                "credit-card-fraud-detection",
                "credit-risk-modeling",
                "bitcoin-price-prediction"
            ]
        },
        "clustering": {
            "name": "Clustering Analysis", 
            "description": "Unsupervised learning and pattern discovery",
            "domains": [DataDomain.MARKETING, DataDomain.RETAIL, DataDomain.GENERAL],
            "complexity": AnalysisComplexity.INTERMEDIATE,
            "sections": [
                "data_overview",
                "feature_scaling_analysis",
                "dimensionality_exploration",
                "cluster_tendency_analysis",
                "optimal_clusters_detection",
                "visualization_suite",
                "cluster_interpretation"
            ],
            "kaggle_patterns": [
                "customer-segmentation-k-means",
                "mall-customer-segmentation"
            ]
        },
        "fraud_detection": {
            "name": "Fraud Detection Analysis",
            "description": "Specialized analysis for fraud detection and anomaly identification",
            "domains": [DataDomain.FRAUD, DataDomain.FINANCE],
            "complexity": AnalysisComplexity.EXPERT,
            "sections": [
                "fraud_overview",
                "class_imbalance_analysis",
                "anomaly_detection_prep",
                "transaction_patterns",
                "risk_factor_analysis",
                "temporal_fraud_patterns",
                "feature_engineering_fraud",
                "model_recommendations_fraud"
            ],
            "kaggle_patterns": [
                "credit-card-fraud-detection",
                "fraud-detection-analysis",
                "anomaly-detection"
            ]
        },
        "computer_vision": {
            "name": "Computer Vision Analysis",
            "description": "Image data exploration and preparation",
            "domains": [DataDomain.COMPUTER_VISION, DataDomain.HEALTHCARE],
            "complexity": AnalysisComplexity.EXPERT,
            "sections": [
                "image_dataset_overview", 
                "image_statistics",
                "class_distribution",
                "sample_visualization",
                "augmentation_preview",
                "preprocessing_pipeline",
                "model_recommendations"
            ],
            "kaggle_patterns": [
                "plant-pathology-2020",
                "melanoma-metadata-analysis"
            ]
        }
    }

    def __init__(self, templates_dir: Optional[Path] = None, examples_dir: Optional[Path] = None):
        """
        Initialize template manager with access to Kaggle examples
        
        Args:
            templates_dir: Directory for generated templates
            examples_dir: Directory containing Kaggle example notebooks
        """
        self.templates_dir = templates_dir or Path("data/eda_templates")
        self.examples_dir = examples_dir or Path("data_science_examples")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.template_cache = {}
        
    def analyze_dataset(self, data: pd.DataFrame) -> DatasetProfile:
        """
        Analyze dataset characteristics to suggest appropriate templates
        
        Args:
            data: Input DataFrame
            
        Returns:
            DatasetProfile with dataset characteristics
        """
        # Basic statistics
        rows, columns = data.shape
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(data.select_dtypes(include=['datetime64']).columns)
        
        # Text column detection (heuristic)
        text_cols = 0
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].astype(str).str.len().mean() > 50:  # Average length > 50 chars
                text_cols += 1
                
        # Missing data percentage
        missing_percentage = (data.isnull().sum().sum() / (rows * columns)) * 100
        
        # Time component detection
        has_time_component = datetime_cols > 0 or any(
            'date' in col.lower() or 'time' in col.lower() 
            for col in data.columns
        )
        
        return DatasetProfile(
            rows=rows,
            columns=columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            text_cols=text_cols,
            datetime_cols=datetime_cols,
            missing_percentage=missing_percentage,
            has_time_component=has_time_component
        )
    
    def suggest_templates(self, profile: DatasetProfile) -> List[Tuple[str, float]]:
        """
        Suggest appropriate templates based on dataset profile
        
        Args:
            profile: Dataset profile
            
        Returns:
            List of (template_key, confidence_score) tuples
        """
        suggestions = []
        
        for key, template in self.TEMPLATE_CATEGORIES.items():
            confidence = 0.0
            
            # Domain-specific scoring
            if profile.domain and profile.domain in template.get("domains", []):
                confidence += 0.4
            
            # Data type scoring
            if key == "nlp" and profile.text_cols > 0:
                confidence += 0.5 * (profile.text_cols / profile.columns)
            elif key == "time_series" and profile.has_time_component:
                confidence += 0.6
            elif key == "computer_vision":
                # Would need image detection logic
                confidence += 0.1
            elif key in ["classification", "regression"]:
                # Prefer for structured data
                if profile.numeric_cols > profile.columns * 0.3:
                    confidence += 0.3
            elif key == "clustering":
                # Good for unsupervised scenarios
                confidence += 0.2
            elif key in ["basic_overview", "descriptive_analysis"]:
                # Always good for basic exploration
                confidence += 0.3
            elif key == "fraud_detection":
                # Look for fraud indicators in column names
                fraud_keywords = ['transaction', 'amount', 'fraud', 'suspicious', 'payment', 'credit', 'card']
                if any(keyword in str(profile.__dict__).lower() for keyword in fraud_keywords):
                    confidence += 0.5
                
            # Size-based adjustments
            if profile.rows < 1000:
                if template["complexity"] == AnalysisComplexity.BASIC:
                    confidence += 0.1
            elif profile.rows > 100000:
                if template["complexity"] in [AnalysisComplexity.ADVANCED, AnalysisComplexity.EXPERT]:
                    confidence += 0.1
                    
            suggestions.append((key, confidence))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_template(
        self,
        template_type: str,
        data_info: Dict[str, Any],
        custom_sections: Optional[List[str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> nbformat.NotebookNode:
        """
        Generate notebook template for specific analysis type
        
        Args:
            template_type: Type of template
            data_info: Information about the dataset
            custom_sections: Optional custom sections to include
            user_preferences: User preferences for template generation
            
        Returns:
            Notebook template
        """
        if template_type not in self.TEMPLATE_CATEGORIES:
            raise ValueError(f"Unknown template type: {template_type}")
        
        # Check cache
        cache_key = f"{template_type}_{hash(str(custom_sections))}"
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
        # Create notebook
        nb = nbformat.v4.new_notebook()
        
        # Add metadata
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "template_type": template_type,
            "created_at": datetime.now().isoformat(),
            "data_info": data_info
        }
        
        # Get template configuration
        template_info = self.TEMPLATE_CATEGORIES[template_type]
        
        # Add title cell
        nb.cells.append(self._create_title_cell(template_info, data_info))
        
        # Add setup cell
        nb.cells.append(self._create_setup_cell(template_type))
        
        # Add data loading cell
        nb.cells.append(self._create_data_loading_cell(data_info))
        
        # Add section cells
        sections = custom_sections or template_info["sections"]
        for section in sections:
            section_cells = self._create_section_cells(section, template_type, data_info)
            nb.cells.extend(section_cells)
        
        # Add footer cell
        nb.cells.append(self._create_footer_cell(template_info))
        
        # Cache the result
        self.template_cache[cache_key] = nb
        
        return nb
    
    def _create_title_cell(self, template_info: Dict[str, Any], data_info: Dict[str, Any]) -> nbformat.NotebookNode:
        """Create notebook title cell"""
        title = template_info["name"]
        description = template_info["description"]
        
        content = f"""# {title}

## Dataset: {data_info.get('name', 'Unknown Dataset')}

**Analysis Type:** {description}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overview

This notebook provides a comprehensive exploratory data analysis (EDA) following industry best practices and patterns derived from successful Kaggle competitions.

### Key Analysis Areas:
"""
        
        for section in template_info["sections"]:
            content += f"- {section.replace('_', ' ').title()}\n"
        
        content += "\n---\n"
        
        return nbformat.v4.new_markdown_cell(content)
    
    def _create_setup_cell(self, template_type: str) -> nbformat.NotebookNode:
        """Create setup cell with imports based on template type"""
        
        # Base imports
        imports = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            "",
            "# Set plotting style",
            "plt.style.use('default')",
            "sns.set_palette('husl')",
            "%matplotlib inline"
        ]
        
        # Template-specific imports
        if template_type == "classification":
            imports.extend([
                "",
                "# Classification specific imports",
                "from sklearn.model_selection import train_test_split",
                "from sklearn.metrics import classification_report, confusion_matrix",
                "from sklearn.preprocessing import LabelEncoder, StandardScaler"
            ])
        elif template_type == "regression":
            imports.extend([
                "",
                "# Regression specific imports", 
                "from sklearn.model_selection import train_test_split",
                "from sklearn.metrics import mean_squared_error, r2_score",
                "from sklearn.preprocessing import StandardScaler",
                "from scipy import stats"
            ])
        elif template_type == "time_series":
            imports.extend([
                "",
                "# Time series specific imports",
                "from datetime import datetime, timedelta",
                "from statsmodels.tsa.seasonal import seasonal_decompose",
                "from statsmodels.tsa.stattools import adfuller",
                "import plotly.graph_objects as go",
                "import plotly.express as px"
            ])
        elif template_type == "nlp":
            imports.extend([
                "",
                "# NLP specific imports",
                "import re",
                "import string",
                "from collections import Counter",
                "from wordcloud import WordCloud",
                "import nltk",
                "from textblob import TextBlob"
            ])
        elif template_type == "healthcare":
            imports.extend([
                "",
                "# Healthcare specific imports",
                "from scipy import stats",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.model_selection import StratifiedKFold"
            ])
        elif template_type == "finance":
            imports.extend([
                "",
                "# Finance specific imports", 
                "from scipy import stats",
                "import plotly.graph_objects as go",
                "from sklearn.ensemble import IsolationForest"
            ])
        elif template_type == "clustering":
            imports.extend([
                "",
                "# Clustering specific imports",
                "from sklearn.cluster import KMeans, DBSCAN",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.decomposition import PCA",
                "from sklearn.metrics import silhouette_score"
            ])
            
        return nbformat.v4.new_code_cell("\n".join(imports))
    
    def _create_data_loading_cell(self, data_info: Dict[str, Any]) -> nbformat.NotebookNode:
        """Create data loading cell"""
        
        source_id = data_info.get('source_id', 'unknown')
        file_format = data_info.get('format', 'csv')
        
        content = f"""# Load the dataset
# Data source: {source_id}

# Note: In production, this would connect to your data ingestion system
# For now, we'll assume the data is available as 'df'

try:
    # Load data based on format
    if '{file_format}' == 'csv':
        df = pd.read_csv('uploads/data/{source_id}*.csv')
    elif '{file_format}' == 'excel':
        df = pd.read_excel('uploads/data/{source_id}*.xlsx')
    elif '{file_format}' == 'json':
        df = pd.read_json('uploads/data/{source_id}*.json')
    else:
        print("Please ensure your data is loaded into 'df' variable")
        
    print(f"Dataset loaded successfully!")
    print(f"Shape: {{df.shape}}")
    print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")
    
except Exception as e:
    print(f"Error loading data: {{e}}")
    print("Please ensure the dataset is available and try again.")
"""
        
        return nbformat.v4.new_code_cell(content)
        
    def _create_section_cells(
        self, 
        section: str, 
        template_type: str, 
        data_info: Dict[str, Any]
    ) -> List[nbformat.NotebookNode]:
        """Create cells for a specific section"""
        
        cells = []
        
        # Add section header
        section_title = section.replace('_', ' ').title()
        cells.append(nbformat.v4.new_markdown_cell(f"## {section_title}"))
        
        # Generate section-specific content
        if section == "data_overview":
            cells.extend(self._create_data_overview_cells())
        elif section == "target_analysis":
            cells.extend(self._create_target_analysis_cells())
        elif section == "correlation_analysis":
            cells.extend(self._create_correlation_cells())
        elif section == "visualization_suite":
            cells.extend(self._create_visualization_cells(template_type))
        # Add more sections as needed...
        else:
            # Generic placeholder
            cells.append(nbformat.v4.new_markdown_cell(f"### {section_title} Analysis\n\nAdd your analysis for {section} here."))
            cells.append(nbformat.v4.new_code_cell("# Add your code here\npass"))
        
        return cells
    
    def _create_data_overview_cells(self) -> List[nbformat.NotebookNode]:
        """Create data overview analysis cells"""
        cells = []
        
        cells.append(nbformat.v4.new_markdown_cell("### Basic Information"))
        
        code = """# Basic dataset information
print("Dataset Shape:", df.shape)
print("\\nColumn Names and Types:")
print(df.dtypes)
print("\\nBasic Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\\nMissing Values:")
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
if not missing_summary.empty:
    print(missing_summary)
    print(f"\\nTotal missing values: {df.isnull().sum().sum()}")
    print(f"Percentage missing: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")
else:
    print("No missing values found!")"""
    
        cells.append(nbformat.v4.new_code_cell(code))
        
        cells.append(nbformat.v4.new_markdown_cell("### First Few Rows"))
        cells.append(nbformat.v4.new_code_cell("df.head()"))
        
        return cells
    
    def _create_target_analysis_cells(self) -> List[nbformat.NotebookNode]:
        """Create target variable analysis cells"""
        cells = []
        
        cells.append(nbformat.v4.new_markdown_cell("### Target Variable Analysis"))
        
        code = """# Assuming target variable is the last column or named 'target'
# Adjust this based on your specific dataset

target_col = df.columns[-1]  # Adjust as needed
print(f"Target column: {target_col}")

if target_col in df.columns:
    print(f"\\nTarget variable distribution:")
    print(df[target_col].value_counts())
    
    # Visualize target distribution
    plt.figure(figsize=(10, 6))
    
    if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
        # Categorical target
        plt.subplot(1, 2, 1)
        df[target_col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {target_col}')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Proportion of {target_col}')
    else:
        # Numerical target
        plt.subplot(1, 2, 1)
        df[target_col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        df.boxplot(column=target_col)
        plt.title(f'Box Plot of {target_col}')
    
    plt.tight_layout()
    plt.show()
else:
    print("Please specify the correct target column name")"""
    
        cells.append(nbformat.v4.new_code_cell(code))
        
        return cells
    
    def _create_correlation_cells(self) -> List[nbformat.NotebookNode]:
        """Create correlation analysis cells"""
        cells = []
        
        cells.append(nbformat.v4.new_markdown_cell("### Correlation Analysis"))
        
        code = """# Correlation analysis for numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated pairs
    threshold = 0.8
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"\\nHighly correlated pairs (|r| > {threshold}):")
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} - {var2}: {corr:.3f}")
    else:
        print(f"\\nNo highly correlated pairs found (|r| > {threshold})")
else:
    print("Not enough numerical columns for correlation analysis")"""
    
        cells.append(nbformat.v4.new_code_cell(code))
        
        return cells
    
    def _create_visualization_cells(self, template_type: str) -> List[nbformat.NotebookNode]:
        """Create visualization cells based on template type"""
        cells = []
        
        cells.append(nbformat.v4.new_markdown_cell("### Advanced Visualizations"))
        
        if template_type == "classification":
            code = """# Feature distributions by target class
target_col = df.columns[-1]  # Adjust as needed
numeric_cols = df.select_dtypes(include=[np.number]).columns

if target_col in df.columns and len(numeric_cols) > 0:
    # Select top 6 numeric features for visualization
    features_to_plot = numeric_cols[:6]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 3, i)
        for target_value in df[target_col].unique():
            subset = df[df[target_col] == target_value]
            plt.hist(subset[feature], alpha=0.6, label=f'{target_col}={target_value}')
        
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature} by {target_col}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("Please ensure target column and numeric features are properly defined")"""
        
        elif template_type == "regression":
            code = """# Feature relationships with target
target_col = df.columns[-1]  # Adjust as needed
numeric_cols = df.select_dtypes(include=[np.number]).columns

if target_col in df.columns and len(numeric_cols) > 1:
    # Select top 6 features for scatter plots
    features_to_plot = [col for col in numeric_cols if col != target_col][:6]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 3, i)
        plt.scatter(df[feature], df[target_col], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f'{target_col} vs {feature}')
        
        # Add trend line
        z = np.polyfit(df[feature].dropna(), df[target_col].loc[df[feature].dropna().index], 1)
        p = np.poly1d(z)
        plt.plot(df[feature], p(df[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()
else:
    print("Please ensure target column and numeric features are properly defined")"""
        
        else:
            code = """# General visualization suite
numeric_cols = df.select_dtypes(include=[np.number]).columns

if len(numeric_cols) > 0:
    # Distribution plots
    n_cols = min(3, len(numeric_cols))
    n_rows = min(2, (len(numeric_cols) + n_cols - 1) // n_cols)
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numeric_cols[:6]):
        plt.subplot(n_rows, n_cols, i + 1)
        df[col].hist(bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns available for visualization")"""
        
        cells.append(nbformat.v4.new_code_cell(code))
        
        return cells
    
    def _create_footer_cell(self, template_info: Dict[str, Any]) -> nbformat.NotebookNode:
        """Create notebook footer cell"""
        
        content = f"""---

## Summary

This {template_info['name'].lower()} analysis has covered the key areas for understanding your dataset:

"""
        
        for section in template_info["sections"]:
            content += f"✅ **{section.replace('_', ' ').title()}**\n"
        
        content += """
## Next Steps

Based on this analysis, consider:

1. **Data Quality**: Address any missing values or outliers identified
2. **Feature Engineering**: Create new features based on insights discovered
3. **Model Selection**: Choose appropriate algorithms based on data characteristics
4. **Validation Strategy**: Design proper cross-validation approach

## Resources

This analysis was generated using patterns from successful Kaggle competitions and industry best practices.

---

*Generated by MLops Platform EDA Template Manager*
"""
        
        return nbformat.v4.new_markdown_cell(content)
    
    def save_template(self, notebook: nbformat.NotebookNode, filename: str) -> Path:
        """Save notebook template to file"""
        output_path = self.templates_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        log_data_action("TEMPLATE_SAVED", details=f"filename:{filename}")
        
        return output_path
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available template categories"""
        return self.TEMPLATE_CATEGORIES
    
    def get_kaggle_patterns(self, template_type: str) -> List[str]:
        """Get Kaggle patterns associated with a template type"""
        template = self.TEMPLATE_CATEGORIES.get(template_type, {})
        return template.get("kaggle_patterns", [])
    
    def get_domain_templates(self, domain: str) -> List[Dict[str, Any]]:
        """Get template recommendations for a specific domain"""
        domain_templates = []
        
        # Map domain to relevant template categories
        domain_mapping = {
            "healthcare": ["classification", "regression", "healthcare"],
            "finance": ["time_series", "regression", "finance", "fraud_detection"], 
            "retail": ["clustering", "classification", "regression", "descriptive_analysis"],
            "real_estate": ["regression", "clustering", "descriptive_analysis"],
            "tech": ["classification", "clustering", "time_series"],
            "marketing": ["clustering", "classification", "regression", "descriptive_analysis"],
            "time_series": ["time_series", "regression"],
            "nlp": ["nlp", "classification"],
            "computer_vision": ["computer_vision", "classification"],
            "fraud": ["fraud_detection", "classification", "clustering"],
            "general": ["basic_overview", "descriptive_analysis", "classification", "regression", "clustering"]
        }
        
        # Get relevant template types for domain
        template_types = domain_mapping.get(domain, domain_mapping["general"])
        
        for template_type in template_types:
            template_info = self.TEMPLATE_CATEGORIES.get(template_type, {})
            if template_info:
                domain_templates.append({
                    "template_id": template_type,
                    "name": template_info.get("name", template_type.title()),
                    "description": template_info.get("description", ""),
                    "complexity": self._get_complexity_string(template_info.get("complexity", "intermediate")),
                    "estimated_time": template_info.get("estimated_time", "15-30 minutes"),
                    "techniques": template_info.get("techniques", []),
                    "kaggle_patterns": template_info.get("kaggle_patterns", [])
                })
        
        return domain_templates
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with basic info"""
        templates = []
        
        for template_id, template_info in self.TEMPLATE_CATEGORIES.items():
            templates.append({
                "template_id": template_id,
                "name": template_info.get("name", template_id.title()),
                "description": template_info.get("description", ""),
                "complexity": self._get_complexity_string(template_info.get("complexity", "intermediate")),
                "domain_suitable": template_info.get("domain_suitable", ["general"])
            })
        
        return templates

    def _calculate_data_quality_score(self, data_preview: Dict[str, Any], metadata: Dict[str, Any] = None, analysis_depth: str = "intermediate") -> Tuple[str, List[str]]:
        """Calculate a real data quality score based on actual data characteristics."""
        try:
            columns = data_preview.get("columns", [])
            sample_data = data_preview.get("data", [])
            shape = data_preview.get("shape", (len(sample_data), len(columns)))
            
            quality_issues = []
            quality_score = 100
            
            # Get metadata for more detailed analysis
            if metadata:
                null_counts = metadata.get("null_counts", {})
                dtypes = metadata.get("dtypes", {})
                total_rows = shape[0] if shape else len(sample_data)
                total_columns = shape[1] if shape else len(columns)
                
                # Check for missing values
                if null_counts:
                    total_nulls = sum(null_counts.values())
                    null_percentage = (total_nulls / (total_rows * total_columns)) * 100
                    
                    if null_percentage > 30:
                        quality_score -= 25
                        quality_issues.append(f"High missing data: {null_percentage:.1f}%")
                    elif null_percentage > 15:
                        quality_score -= 15
                        quality_issues.append(f"Moderate missing data: {null_percentage:.1f}%")
                    elif null_percentage > 5:
                        quality_score -= 8
                        quality_issues.append(f"Low missing data: {null_percentage:.1f}%")
                    elif null_percentage > 1:
                        quality_score -= 3
                        quality_issues.append(f"Minor missing data: {null_percentage:.1f}%")
                
                # Check for data type consistency
                if dtypes:
                    object_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype)]
                    if len(object_cols) > len(columns) * 0.8:
                        quality_score -= 10
                        quality_issues.append("Many text columns - may need preprocessing")
                
                # Check dataset size
                if total_rows < 100:
                    quality_score -= 20
                    quality_issues.append("Small dataset - limited statistical power")
                elif total_rows < 1000:
                    quality_score -= 10
                    quality_issues.append("Moderate dataset size")
                
                # Advanced depth checks for expert/comprehensive analysis
                if analysis_depth in ["advanced", "expert"]:
                    # Check for potential duplicates (rough estimate)
                    if dtypes and len([col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]) < 2:
                        quality_score -= 5
                        quality_issues.append("Limited numeric features for statistical analysis")
                    
                    # Check column naming quality
                    if columns:
                        poor_names = [col for col in columns if len(col.strip()) < 2 or col.lower() in ['unnamed', 'index']]
                        if len(poor_names) > 0:
                            quality_score -= 5
                            quality_issues.append(f"Poor column naming detected")
                
            else:
                # Fallback analysis using sample data only
                if len(sample_data) < 10:
                    quality_score -= 30
                    quality_issues.append("Very small sample available")
                    
                # Check for empty columns in sample
                if sample_data:
                    empty_cols = 0
                    for col in columns:
                        col_values = [row.get(col) for row in sample_data if isinstance(row, dict)]
                        if all(v is None or v == '' or str(v).strip() == '' for v in col_values):
                            empty_cols += 1
                    
                    if empty_cols > 0:
                        quality_score -= min(20, empty_cols * 5)
                        quality_issues.append(f"{empty_cols} columns appear empty")
            
            # Ensure score is within bounds
            quality_score = max(0, min(100, quality_score))
            
            # Format score
            if quality_score >= 90:
                score_text = f"{quality_score}% ⭐"
            elif quality_score >= 75:
                score_text = f"{quality_score}%"
            elif quality_score >= 50:
                score_text = f"{quality_score}% ⚠️"
            else:
                score_text = f"{quality_score}% 🔴"
                
            return score_text, quality_issues
            
        except Exception as e:
            return "75%", [f"Quality assessment error: {str(e)}"]

    def _build_key_findings(self, template_info: Dict, template_name: str, 
                           actual_columns: int, actual_rows: int, 
                           quality_issues: List[str], analysis_depth: str) -> List[str]:
        """Build key findings based on analysis depth."""
        key_findings = [
            f"Dataset structure: {actual_columns} columns, {actual_rows:,} rows",
            f"Template complexity: {self._get_complexity_string(template_info.get('complexity', 'intermediate'))}",
            f"Analysis type: {template_name.replace('_', ' ').title()}"
        ]
        
        # Add findings based on depth
        if analysis_depth == "basic":
            # Basic: Just add top quality issue
            if quality_issues:
                key_findings.append(quality_issues[0])
        elif analysis_depth == "intermediate":
            # Standard: Add top 2 quality issues
            if quality_issues:
                key_findings.extend(quality_issues[:2])
        elif analysis_depth == "advanced":
            # Comprehensive: Add more quality issues + dataset insights
            if quality_issues:
                key_findings.extend(quality_issues[:3])
            key_findings.append(f"Ready for {template_name} modeling approach")
        elif analysis_depth == "expert":
            # Expert: Add all quality issues + advanced insights
            key_findings.extend(quality_issues)
            key_findings.extend([
                f"Recommended preprocessing steps will be detailed",
                f"Statistical assumptions will be validated"
            ])
            
        return key_findings

    def _get_complexity_string(self, complexity) -> str:
        """Convert AnalysisComplexity enum to string safely."""
        if hasattr(complexity, 'value'):
            return complexity.value
        elif complexity is None:
            return "intermediate"
        else:
            return str(complexity)

    def generate_template_analysis(
        self,
        template_name: str,
        data_preview: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        source_id: str = None,
        analysis_depth: str = "intermediate"  # New parameter
    ) -> Dict[str, Any]:
        """Generate analysis results using a specific template with configurable depth."""
        try:
            # Validate template exists
            if template_name not in self.TEMPLATE_CATEGORIES:
                return {
                    "success": False,
                    "error": f"Unknown template: {template_name}",
                    "available_templates": list(self.TEMPLATE_CATEGORIES.keys())
                }
            
            template_info = self.TEMPLATE_CATEGORIES[template_name]
            
            # Extract data information
            columns = data_preview.get("columns", [])
            sample_data = data_preview.get("data", [])
            shape = data_preview.get("shape", (len(sample_data), len(columns)))
            
            # Get actual dataset dimensions
            actual_rows = shape[0] if shape and len(shape) > 0 else len(sample_data)
            actual_columns = shape[1] if shape and len(shape) > 1 else len(columns)
            
            # Calculate real quality score (depth affects detail level)
            quality_score, quality_issues = self._calculate_data_quality_score(
                data_preview, metadata, analysis_depth
            )
            
            # Build key findings with real data (depth affects number of findings)
            key_findings = self._build_key_findings(
                template_info, template_name, actual_columns, actual_rows, 
                quality_issues, analysis_depth
            )
            
            # Generate comprehensive analysis results with actual data insights
            analysis_results = {
                "template_name": template_name,
                "analysis_depth": analysis_depth,
                "template_info": {
                    "name": template_info.get("name", template_name.title()),
                    "description": template_info.get("description", ""),
                    "complexity": self._get_complexity_string(template_info.get("complexity", "intermediate"))
                },
                "summary": {
                    "key_findings": key_findings,
                    "data_score": quality_score,
                    # Keep original fields for backwards compatibility
                    "total_columns": actual_columns,
                    "total_rows": actual_rows,
                    "template_name": template_info.get("name", template_name.title()),
                    "analysis_type": template_name,
                    "source_id": source_id
                },
                "visualizations": self._generate_template_visualizations(
                    template_name, columns, sample_data, metadata, analysis_depth
                )
            }
            
            return analysis_results
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis generation failed: {str(e)}"
            }

    def _generate_template_visualizations(
        self,
        template_name: str,
        columns: List[str],
        sample_data: List[Any],
        metadata: Dict[str, Any] = None,
        analysis_depth: str = "intermediate"
    ) -> List[Dict[str, Any]]:
        """Generate template-specific visualizations with actual data insights."""
        visualizations = []
        
        try:
            # Convert sample data to DataFrame for analysis
            if sample_data and columns:
                df = pd.DataFrame(sample_data, columns=columns)
                
                # Generate visualizations based on template type
                if template_name == "fraud_detection":
                    visualizations.extend(self._generate_fraud_visualizations(df, metadata))
                elif template_name == "classification":
                    visualizations.extend(self._generate_classification_visualizations(df, metadata))
                elif template_name == "regression":
                    visualizations.extend(self._generate_regression_visualizations(df, metadata))
                elif template_name == "time_series":
                    visualizations.extend(self._generate_timeseries_visualizations(df, metadata))
                elif template_name in ["basic_overview", "descriptive_analysis"]:
                    visualizations.extend(self._generate_basic_visualizations(df, metadata))
                else:
                    # Generic analysis for other templates
                    visualizations.extend(self._generate_generic_visualizations(df, metadata))
                
                # Add data quality visualization for all templates
                if analysis_depth in ["advanced", "expert"]:
                    visualizations.append(self._generate_data_quality_chart(df, metadata))
                
                # Ensure we always have at least basic visualizations
                if len(visualizations) == 0:
                    # Fall back to basic visualizations if template-specific ones failed
                    visualizations.extend(self._generate_basic_visualizations(df, metadata))
                    
            else:
                # Fallback when no sample data available
                actual_sample_rows = len(sample_data) if sample_data else 0
                visualizations.append({
                    "type": "warning",
                    "title": "Limited Data Available",
                    "description": "Analysis is based on column structure only. Upload more data for detailed insights.",
                    "data": {"columns": len(columns), "sample_rows": actual_sample_rows}
                })
                
        except Exception as e:
            visualizations.append({
                "type": "error",
                "title": "Visualization Generation Error",
                "description": f"Unable to generate visualizations: {str(e)}",
                "data": {}
            })
        
        return visualizations

    def _generate_fraud_visualizations(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate fraud detection specific visualizations."""
        visualizations = []
        
        # Find potential fraud indicator columns
        fraud_columns = []
        target_column = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['fraud', 'suspicious', 'flag', 'label', 'target']):
                if df[col].nunique() == 2:  # Binary column
                    target_column = col
            elif any(keyword in col_lower for keyword in ['amount', 'transaction', 'payment']):
                fraud_columns.append(col)
        
        # Class distribution chart (if target found)
        if target_column and not df[target_column].isnull().all():
            try:
                class_counts = df[target_column].value_counts()
                total = len(df)
                
                visualizations.append({
                    "type": "pie_chart",
                    "title": "Fraud vs Normal Transactions",
                    "description": f"Distribution of {target_column} in dataset",
                    "data": {
                        "labels": [str(label) for label in class_counts.index],
                        "values": class_counts.tolist(),
                        "colors": ["#28a745", "#dc3545"] if len(class_counts) == 2 else None,
                        "total_samples": total,
                        "percentages": [(count/total)*100 for count in class_counts.tolist()]
                    }
                })
            except Exception:
                pass
        
        # Amount distribution (if numeric amount column found)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if fraud_columns:
            amount_col = fraud_columns[0]  # First fraud-related numeric column
            if amount_col in numeric_cols and not df[amount_col].isnull().all():
                try:
                    amounts = df[amount_col].dropna()
                    if len(amounts) > 0:
                        visualizations.append({
                            "type": "histogram",
                            "title": f"{amount_col.title()} Distribution",
                            "description": f"Distribution of transaction amounts",
                            "data": {
                                "values": amounts.tolist()[:1000],  # Limit for performance
                                "bins": min(30, len(amounts.unique())),
                                "statistics": {
                                    "mean": float(amounts.mean()),
                                    "median": float(amounts.median()),
                                    "std": float(amounts.std()),
                                    "min": float(amounts.min()),
                                    "max": float(amounts.max())
                                }
                            }
                        })
                except Exception:
                    pass
        
        # Feature correlation heatmap (for numeric features)
        if len(numeric_cols) >= 2:
            try:
                correlation_data = df[numeric_cols].corr()
                
                visualizations.append({
                    "type": "heatmap",
                    "title": "Feature Correlation Matrix",
                    "description": "Correlation between numeric features",
                    "data": {
                        "correlation_matrix": correlation_data.values.tolist(),
                        "feature_names": correlation_data.columns.tolist(),
                        "colorscale": "RdBu"
                    }
                })
            except Exception:
                pass
        
        return visualizations

    def _generate_classification_visualizations(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate classification specific visualizations."""
        visualizations = []
        
        # Find target column (usually last column or contains 'target', 'class', 'label')
        target_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['target', 'class', 'label', 'y']):
                target_col = col
                break
        
        if not target_col:
            target_col = df.columns[-1]  # Default to last column
        
        # Class distribution
        if target_col in df.columns and not df[target_col].isnull().all():
            try:
                class_counts = df[target_col].value_counts()
                visualizations.append({
                    "type": "bar_chart",
                    "title": f"Target Variable Distribution ({target_col})",
                    "description": "Distribution of classes in the dataset",
                    "data": {
                        "labels": [str(label) for label in class_counts.index],
                        "values": class_counts.tolist(),
                        "total_samples": int(len(df))
                    }
                })
            except Exception:
                pass
        
        # Feature distributions by class (for numeric features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0 and target_col in df.columns:
            try:
                feature_col = numeric_cols[0]
                by_class_data = {}
                
                for class_val in df[target_col].unique():
                    if pd.notna(class_val):
                        subset = df[df[target_col] == class_val][feature_col].dropna()
                        if len(subset) > 0:
                            by_class_data[str(class_val)] = subset.tolist()[:100]  # Limit size
                
                if by_class_data:
                    visualizations.append({
                        "type": "box_plot",
                        "title": f"{feature_col} by {target_col}",
                        "description": f"Distribution of {feature_col} across different classes",
                        "data": by_class_data
                    })
            except Exception:
                pass
        
        return visualizations

    def _generate_regression_visualizations(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate regression specific visualizations."""
        visualizations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Scatter plot of two numeric features
            try:
                x_col, y_col = numeric_cols[:2]
                x_vals = df[x_col].dropna()
                y_vals = df[y_col].dropna()
                
                # Get common indices
                common_idx = x_vals.index.intersection(y_vals.index)
                if len(common_idx) > 0:
                    visualizations.append({
                        "type": "scatter_plot",
                        "title": f"{y_col} vs {x_col}",
                        "description": "Relationship between numeric features",
                        "data": {
                            "x": df.loc[common_idx, x_col].tolist()[:500],
                            "y": df.loc[common_idx, y_col].tolist()[:500],
                            "x_label": x_col,
                            "y_label": y_col
                        }
                    })
            except Exception:
                pass
        
        # Distribution of target variable (assuming last numeric column)
        if len(numeric_cols) > 0:
            try:
                target_col = numeric_cols[-1]
                target_vals = df[target_col].dropna()
                
                if len(target_vals) > 0:
                    visualizations.append({
                        "type": "histogram",
                        "title": f"Target Distribution ({target_col})",
                        "description": "Distribution of the target variable",
                        "data": {
                            "values": target_vals.tolist()[:1000],
                            "bins": min(30, len(target_vals.unique())),
                            "statistics": {
                                "mean": float(target_vals.mean()),
                                "std": float(target_vals.std()),
                                "median": float(target_vals.median())
                            }
                        }
                    })
            except Exception:
                pass
        
        return visualizations

    def _generate_basic_visualizations(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate basic overview visualizations."""
        visualizations = []
        
        # Data types distribution
        try:
            dtype_counts = {}
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                if 'int' in dtype_str or 'float' in dtype_str:
                    dtype_category = 'Numeric'
                elif 'object' in dtype_str:
                    dtype_category = 'Text/Categorical'
                elif 'datetime' in dtype_str:
                    dtype_category = 'DateTime'
                else:
                    dtype_category = 'Other'
                
                dtype_counts[dtype_category] = dtype_counts.get(dtype_category, 0) + 1
            
            if dtype_counts:
                visualizations.append({
                    "type": "pie_chart",
                    "title": "Data Types Distribution",
                    "description": "Distribution of column data types",
                    "data": {
                        "labels": list(dtype_counts.keys()),
                        "values": list(dtype_counts.values()),
                        "colors": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
                    }
                })
        except Exception:
            pass
        
        # Missing values chart
        try:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                visualizations.append({
                    "type": "bar_chart",
                    "title": "Missing Values by Column",
                    "description": "Columns with missing data",
                    "data": {
                        "labels": missing_data.index.tolist(),
                        "values": missing_data.values.tolist(),
                        "color": "#e74c3c"
                    }
                })
        except Exception:
            pass
        
        # Add basic dataset info visualization if no other visualizations were created
        if len(visualizations) == 0:
            # Always provide at least basic dataset overview
            visualizations.append({
                "type": "info",
                "title": "Dataset Overview",
                "description": f"Dataset contains {len(df.columns)} columns and {len(df)} rows",
                "data": {
                    "total_columns": int(len(df.columns)),
                    "total_rows": int(len(df)),
                    "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
                    "text_columns": int(len(df.select_dtypes(include=['object']).columns)),
                    "memory_usage": f"{float(df.memory_usage(deep=True).sum()) / 1024**2:.2f} MB"
                }
            })
        
        return visualizations

    def _generate_generic_visualizations(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate generic visualizations for other template types."""
        return self._generate_basic_visualizations(df, metadata)

    def _generate_data_quality_chart(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate data quality overview chart."""
        try:
            quality_metrics = {
                "Complete Data": int(len(df) - df.isnull().any(axis=1).sum()),
                "Missing Data": int(df.isnull().any(axis=1).sum()),
                "Numeric Columns": int(len(df.select_dtypes(include=[np.number]).columns)),
                "Text Columns": int(len(df.select_dtypes(include=['object']).columns))
            }
            
            return {
                "type": "bar_chart",
                "title": "Data Quality Overview",
                "description": "Overall data quality metrics",
                "data": {
                    "labels": list(quality_metrics.keys()),
                    "values": list(quality_metrics.values()),
                    "colors": ["#28a745", "#dc3545", "#17a2b8", "#ffc107"]
                }
            }
        except Exception:
            return {
                "type": "info",
                "title": "Data Quality",
                "description": "Unable to calculate quality metrics",
                "data": {}
            }
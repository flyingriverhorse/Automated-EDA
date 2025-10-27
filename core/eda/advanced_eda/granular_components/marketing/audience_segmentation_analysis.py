"""Audience Segmentation Analysis Component.

Focused analysis component for marketing audience segmentation
analyzing user demographics and behavior patterns.
"""
from typing import Dict, Any


class AudienceSegmentationAnalysis:
    """Focused component for audience segmentation analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "audience_segmentation_analysis",
            "display_name": "Audience Segmentation Analysis",
            "description": "Analyze audience segments: demographics, behavior patterns, and segment performance",
            "category": "marketing",
            "complexity": "advanced",
            "required_data_types": ["categorical", "numeric"],
            "estimated_runtime": "15-25 seconds",
            "icon": "👥",
            "tags": ["marketing", "audience", "segmentation", "demographics", "behavior"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has audience-related columns"""
        # For now, return True to allow all datasets (marketing analyses are flexible)
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate audience segmentation analysis code with column selection support"""
        
        # Check for column_mapping in data_preview (from modal UI)
        column_mapping = None
        if data_preview and 'column_mapping' in data_preview:
            column_mapping = data_preview['column_mapping']
        elif selected_columns:
            column_mapping = selected_columns
        
        # If column_mapping provided, use them; otherwise auto-detect
        column_mapping_code = ""
        if column_mapping:
            column_mapping_code = f"""
# User-selected column mappings
SEGMENT_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect segmentation columns (user can override these)
SEGMENT_MAPPING = {}
possible_mappings = {
    'age': ['age', 'user_age', 'age_group'],
    'gender': ['gender', 'sex', 'demographic_gender'],
    'location': ['location', 'country', 'city', 'region'],
    'purchase_frequency': ['purchase_frequency', 'order_count', 'frequency'],
    'total_spent': ['total_spent', 'lifetime_value', 'total_revenue']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in SEGMENT_MAPPING:
                SEGMENT_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== AUDIENCE SEGMENTATION ANALYSIS ===")
print("Audience demographics and behavior analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 COLUMN MAPPING:")
for metric, col in SEGMENT_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.title()}}: {{col}}")
print()

# Basic segmentation analysis with available columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"📊 Available columns for segmentation:")
print(f"   • Numeric: {{numeric_cols}}")  
print(f"   • Categorical: {{categorical_cols}}")

if SEGMENT_MAPPING:
    print("\\n🎯 AUDIENCE SEGMENTS:")
    # Basic demographic analysis if columns are available
    for segment_type, col in SEGMENT_MAPPING.items():
        if col in df.columns:
            print(f"\\n{{segment_type.upper()}} DISTRIBUTION:")
            if col in categorical_cols:
                counts = df[col].value_counts()
                print(counts.head())
            elif col in numeric_cols:
                print(f"   Mean: {{df[col].mean():.2f}}")
                print(f"   Median: {{df[col].median():.2f}}")
                print(f"   Range: {{df[col].min():.2f}} - {{df[col].max():.2f}}")
else:
    print("⚠️ No segment columns mapped. Please configure column mapping.")

print("\\n💡 SEGMENTATION RECOMMENDATIONS")
print("   • Create demographic segments for targeted campaigns")
print("   • Analyze behavioral patterns by segment")
print("   • Develop personalized marketing strategies")
print("   • Track segment performance over time")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "age": {
                "required": False,
                "description": "User age for demographic analysis",
                "data_type": "numeric",
                "examples": ["age", "age_group", "age_range"]
            },
            "gender": {
                "required": False,
                "description": "User gender for demographic analysis",
                "data_type": "categorical",
                "examples": ["gender", "sex"]
            },
            "location": {
                "required": False,
                "description": "User location for geographic segmentation",
                "data_type": "categorical",
                "examples": ["location", "city", "region", "country"]
            }
        }
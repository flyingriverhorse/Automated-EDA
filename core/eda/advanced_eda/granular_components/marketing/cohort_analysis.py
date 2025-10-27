"""Cohort Analysis Component.

Focused analysis component for marketing cohort analysis
analyzing user retention and behavior over time.
"""
from typing import Dict, Any


class CohortAnalysis:
    """Focused component for cohort analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "cohort_analysis",
            "display_name": "Cohort Analysis", 
            "description": "Analyze user cohorts: retention rates, behavior patterns, and lifetime value trends",
            "category": "marketing",
            "complexity": "advanced",
            "required_data_types": ["datetime", "categorical"],
            "estimated_runtime": "20-30 seconds",
            "icon": "📅",
            "tags": ["marketing", "cohort", "retention", "lifetime", "behavior"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has cohort-related columns"""
        return True  # Simplified for stub
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate cohort analysis code with column selection support"""
        
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
COHORT_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect cohort columns (user can override these)
COHORT_MAPPING = {}
possible_mappings = {
    'user_id': ['user_id', 'customer_id', 'account_id'],
    'signup_date': ['signup_date', 'registration_date', 'first_seen'],
    'activity_date': ['activity_date', 'last_seen', 'event_date'],
    'revenue': ['revenue', 'purchase_amount', 'value'],
    'active': ['active', 'is_active', 'retained']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in COHORT_MAPPING:
                COHORT_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== COHORT ANALYSIS ===")
print("User cohort retention and behavior analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 COLUMN MAPPING:")
for metric, col in COHORT_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.title()}}: {{col}}")
print()

# Basic cohort analysis
if COHORT_MAPPING:
    print("📊 COHORT INSIGHTS:")
    
    # User count analysis
    if 'user_id' in COHORT_MAPPING:
        user_col = COHORT_MAPPING['user_id']
        if user_col in df.columns:
            total_users = df[user_col].nunique()
            print(f"   • Total Users: {{total_users:,}}")
    
    # Activity analysis
    if 'signup_date' in COHORT_MAPPING and 'activity_date' in COHORT_MAPPING:
        signup_col = COHORT_MAPPING['signup_date']
        activity_col = COHORT_MAPPING['activity_date']
        
        if signup_col in df.columns and activity_col in df.columns:
            print("\\n📈 Basic Cohort Statistics:")
            try:
                df[signup_col] = pd.to_datetime(df[signup_col])
                df[activity_col] = pd.to_datetime(df[activity_col])
                
                # Group by signup month
                df['signup_month'] = df[signup_col].dt.to_period('M')
                df['activity_month'] = df[activity_col].dt.to_period('M')
                
                cohort_counts = df.groupby('signup_month')[user_col].nunique()
                print(f"   • Cohorts by signup month:")
                for month, count in cohort_counts.head().items():
                    print(f"     {{month}}: {{count}} users")
                    
            except Exception as e:
                print(f"   ⚠️ Date parsing issue: {{e}}")
    
    print("\\n💡 COHORT ANALYSIS RECOMMENDATIONS")
    print("   • Track user retention by acquisition cohort")
    print("   • Analyze lifetime value progression")
    print("   • Identify cohort behavior patterns")
    print("   • Measure retention improvements over time")
else:
    print("⚠️ No cohort columns mapped. Please configure column mapping.")
    print("\\n💡 COHORT ANALYSIS RECOMMENDATIONS")
    print("   • Map user ID and signup date columns")
    print("   • Include activity/event timestamps")
    print("   • Track revenue or value metrics")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {}
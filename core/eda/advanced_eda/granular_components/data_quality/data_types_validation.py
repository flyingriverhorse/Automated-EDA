"""Data Types Validation Component.

Validates data types and suggests corrections for columns.
"""
from typing import Dict, Any


class DataTypesValidationAnalysis:
    """Analyze and validate data types across all columns."""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "data_types_validation",
            "display_name": "Data Types Validation",
            "description": "Validate column data types and suggest improvements",
            "category": "data_quality",
            "complexity": "basic",
            "tags": ["data_types", "validation", "dtypes", "conversion"],
            "estimated_runtime": "2-5 seconds",
            "icon": "🔧"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview: Dict[str, Any] = None) -> bool:
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        return '''
# ===== DATA TYPES VALIDATION ANALYSIS =====

import pandas as pd
import numpy as np
import re
from datetime import datetime

print("="*60)
print("🔧 DATA TYPES VALIDATION")
print("="*60)

print(f"\\n📊 Dataset Overview:")
print(f"   • Shape: {df.shape}")
print(f"   • Total columns: {df.shape[1]}")

# Get current dtypes
print(f"\\n📋 Current Data Types:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   • {dtype}: {count} columns")

print(f"\\n🔍 COLUMN-BY-COLUMN ANALYSIS:")
print("-" * 60)

recommendations = []
potential_issues = []

for col in df.columns:
    print(f"\\n📈 Column: {col}")
    print(f"   Current dtype: {df[col].dtype}")
    print(f"   Non-null count: {df[col].count():,} / {len(df):,}")
    print(f"   Unique values: {df[col].nunique():,}")
    
    # Sample values for analysis
    non_null_values = df[col].dropna()
    if len(non_null_values) > 0:
        sample_values = non_null_values.head(10).tolist()
        print(f"   Sample values: {sample_values}")
        
        current_dtype = str(df[col].dtype)
        
        # Analyze potential type improvements
        if current_dtype == 'object':
            # Check if it could be numeric
            numeric_convertible = 0
            for val in non_null_values.head(100):
                try:
                    float(str(val).replace(',', '').replace('$', '').replace('%', ''))
                    numeric_convertible += 1
                except:
                    pass
            
            if numeric_convertible > len(non_null_values.head(100)) * 0.8:
                print("   💡 Suggestion: Could be converted to numeric")
                recommendations.append(f"{col}: object → numeric (remove special chars)")
            
            # Check if it could be datetime
            datetime_convertible = 0
            for val in non_null_values.head(50):
                try:
                    pd.to_datetime(str(val))
                    datetime_convertible += 1
                except:
                    pass
            
            if datetime_convertible > len(non_null_values.head(50)) * 0.7:
                print("   💡 Suggestion: Could be converted to datetime")
                recommendations.append(f"{col}: object → datetime")
            
            # Check if it could be categorical
            cardinality_ratio = df[col].nunique() / len(df[col])
            if cardinality_ratio < 0.1 and df[col].nunique() < 50:
                print("   💡 Suggestion: Consider converting to categorical")
                recommendations.append(f"{col}: object → category (low cardinality)")
        
        elif current_dtype in ['int64', 'float64']:
            # Check if integer could be boolean
            if df[col].nunique() == 2:
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}):
                    print("   💡 Suggestion: Could be boolean (0/1 values)")
                    recommendations.append(f"{col}: {current_dtype} → bool")
            
            # Check if float could be integer
            if current_dtype == 'float64':
                if df[col].dropna().apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    print("   💡 Suggestion: Could be integer (no decimals)")
                    recommendations.append(f"{col}: float64 → int64")
        
        # Check for potential data quality issues
        if df[col].isnull().sum() > len(df) * 0.5:
            potential_issues.append(f"{col}: High missing rate ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        
        if current_dtype == 'object':
            # Check for mixed types
            types_found = set()
            for val in non_null_values.head(100):
                if isinstance(val, (int, np.integer)):
                    types_found.add('int')
                elif isinstance(val, (float, np.floating)):
                    types_found.add('float')
                elif isinstance(val, str):
                    types_found.add('str')
                elif isinstance(val, bool):
                    types_found.add('bool')
            
            if len(types_found) > 1:
                potential_issues.append(f"{col}: Mixed data types detected: {types_found}")

# Summary
print(f"\\n" + "="*60)
print("📋 VALIDATION SUMMARY")
print("="*60)

if recommendations:
    print(f"\\n✅ TYPE CONVERSION RECOMMENDATIONS ({len(recommendations)}):")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\\n💡 Example conversion code:")
    print("   # Apply these conversions carefully after validation:")
    for rec in recommendations[:5]:  # Show first 5 examples (increased from 3)
        col_name, suggestion = rec.split(': ')
        if 'numeric' in suggestion:
            print(f"   df['{col_name}'] = pd.to_numeric(df['{col_name}'], errors='coerce')")
        elif 'datetime' in suggestion:
            print(f"   df['{col_name}'] = pd.to_datetime(df['{col_name}'], errors='coerce')")
        elif 'category' in suggestion:
            print(f"   df['{col_name}'] = df['{col_name}'].astype('category')")
        elif 'bool' in suggestion:
            print(f"   df['{col_name}'] = df['{col_name}'].astype('bool')")

if potential_issues:
    print(f"\\n⚠️  POTENTIAL DATA QUALITY ISSUES ({len(potential_issues)}):")
    for i, issue in enumerate(potential_issues, 1):
        print(f"   {i}. {issue}")

if not recommendations and not potential_issues:
    print(f"\\n✅ All data types appear to be appropriate!")
    print("   No obvious improvements or issues detected.")

print(f"\\n📊 Memory usage by dtype:")
memory_usage = df.memory_usage(deep=True)
total_memory = memory_usage.sum()
for dtype in df.dtypes.unique():
    dtype_cols = df.select_dtypes(include=[dtype]).columns
    dtype_memory = df[dtype_cols].memory_usage(deep=True).sum()
    print(f"   • {dtype}: {dtype_memory / 1024 / 1024:.2f} MB ({dtype_memory/total_memory*100:.1f}%)")

print("\\n" + "="*60)
print("✅ Data types validation complete!")
print("="*60)
'''

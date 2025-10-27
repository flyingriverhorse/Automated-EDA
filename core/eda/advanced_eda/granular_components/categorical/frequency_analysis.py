"""Categorical Frequency Analysis Component.

Provides frequency counts and distribution analysis for categorical variables.
"""
from typing import Dict, Any


class CategoricalFrequencyAnalysis:
    """Analyze frequency counts and distribution of categorical variables."""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "categorical_frequency_analysis",
            "display_name": "Categorical Frequency Analysis",
            "description": "Frequency counts and distribution analysis for categorical variables",
            "category": "univariate",
            "complexity": "basic",
            "tags": ["frequency", "categorical", "distribution", "counts"],
            "estimated_runtime": "1-2 seconds",
            "icon": "📊"
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any] = None) -> bool:
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        return len(data_preview.get('categorical_columns', [])) > 0 or len(data_preview.get('object_columns', [])) > 0
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate code for categorical frequency analysis."""
        
        return '''
# ===== CATEGORICAL FREQUENCY ANALYSIS =====

import pandas as pd
import numpy as np

print("="*60)
print("📊 CATEGORICAL FREQUENCY ANALYSIS")
print("="*60)

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"\\n📋 Found {len(categorical_cols)} categorical columns")

if len(categorical_cols) == 0:
    print("❌ No categorical columns found for analysis")
else:
    for col in categorical_cols:
        print(f"\\n{'='*50}")
        print(f"📈 COLUMN: {col}")
        print('='*50)
        
        # Basic info
        unique_count = df[col].nunique()
        total_count = len(df[col])
        null_count = df[col].isnull().sum()
        
        print(f"\\n📊 Basic Statistics:")
        print(f"   • Total values: {total_count:,}")
        print(f"   • Unique values: {unique_count:,}")
        print(f"   • Null values: {null_count:,} ({null_count/total_count*100:.1f}%)")
        print(f"   • Cardinality: {unique_count/total_count*100:.1f}%")
        
        # Frequency analysis
        if unique_count > 0:
            value_counts = df[col].value_counts(dropna=False)
            
            print(f"\\n🔢 Top Frequencies:")
            
            # Show top 15 most frequent values (increased from 10)
            top_values = value_counts.head(15)
            for idx, (value, count) in enumerate(top_values.items(), 1):
                percentage = count / total_count * 100
                value_str = str(value) if pd.notna(value) else "<NULL>"
                if len(value_str) > 30:
                    value_str = value_str[:27] + "..."
                print(f"   {idx:2d}. {value_str:<30} | {count:>8,} ({percentage:>5.1f}%)")
            
            if unique_count > 10:
                print(f"   ... and {unique_count - 10} more unique values")
            
            # Frequency distribution stats
            print(f"\\n📈 Frequency Distribution:")
            print(f"   • Most frequent: '{top_values.index[0]}' ({top_values.iloc[0]:,} times)")
            print(f"   • Least frequent: '{value_counts.index[-1]}' ({value_counts.iloc[-1]:,} times)")
            print(f"   • Average frequency: {total_count/unique_count:.1f} per unique value")
            
            # Frequency concentration
            top_1_pct = top_values.iloc[0] / total_count * 100
            top_5_pct = top_values.head(5).sum() / total_count * 100 if len(top_values) >= 5 else top_values.sum() / total_count * 100
            
            print(f"\\n🎯 Concentration Analysis:")
            print(f"   • Top 1 value covers: {top_1_pct:.1f}% of data")
            if len(top_values) >= 5:
                print(f"   • Top 5 values cover: {top_5_pct:.1f}% of data")
            
            # Identify potential data quality issues
            issues = []
            
            # Check for high cardinality (might be ID column)
            if unique_count > total_count * 0.9:
                issues.append("Very high cardinality - might be an ID column")
            
            # Check for very skewed distribution
            if top_1_pct > 90:
                issues.append(f"Very skewed - top value dominates {top_1_pct:.1f}% of data")
            
            # Check for potential encoding issues
            for val in value_counts.head(30).index:  # Increased from 20 to 30
                if pd.notna(val):
                    val_str = str(val)
                    if any(char in val_str for char in ['�', r'\\x', r'\\u']):
                        issues.append("Potential encoding issues detected")
                        break
            
            if issues:
                print(f"\\n⚠️  Potential Issues:")
                for issue in issues:
                    print(f"   • {issue}")
            else:
                print(f"\\n✅ No obvious data quality issues detected")

print("\\n" + "="*60) 
print("✅ Categorical frequency analysis complete!")
print("="*60)
'''
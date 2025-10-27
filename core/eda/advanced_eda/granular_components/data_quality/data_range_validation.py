"""Data Range Validation Analysis Component.

Validates logical ranges and consistency in data values.
"""


class DataRangeValidationAnalysis:
    """Validate data ranges and logical consistency."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "data_range_validation",
            "display_name": "Data Range Validation",
            "description": "Validate ranges and logical consistency (e.g., age ≥ 0, no negative prices, impossible timestamps)",
            "category": "data_quality", 
            "complexity": "intermediate",
            "tags": ["validation", "ranges", "consistency", "quality"],
            "estimated_runtime": "1-3 seconds",
            "icon": "✅"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        return len(data_preview.get('numeric_columns', [])) > 0
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for data range validation."""
        
        return '''
# ===== DATA RANGE VALIDATION ANALYSIS =====

import pandas as pd
import numpy as np
from datetime import datetime

print("="*60)
print("✅ DATA RANGE AND CONSISTENCY VALIDATION")
print("="*60)

# Get numeric columns for validation
numeric_cols = df.select_dtypes(include=[np.number]).columns
datetime_cols = df.select_dtypes(include=['datetime64']).columns
text_cols = df.select_dtypes(include=['object']).columns

validation_issues = []

print("\\n🔍 NUMERIC RANGE VALIDATION:")
print("-" * 40)

for col in numeric_cols:
    print(f"\\n📊 Column: {col}")
    
    # Basic range info
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"   Range: [{min_val:.3f}, {max_val:.3f}]")
    
    # Check for negative values where they might not make sense
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"   ⚠️  Negative values: {negative_count} ({negative_count/len(df)*100:.1f}%)")
        
        # Common sense checks for potentially problematic negative values
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['age', 'price', 'cost', 'amount', 'quantity', 'count', 'length', 'width', 'height', 'weight', 'distance']):
            validation_issues.append(f"{col}: Has {negative_count} negative values (may be logically incorrect)")
    
    # Check for extremely large values (potential outliers)
    q99 = df[col].quantile(0.99)
    q01 = df[col].quantile(0.01)
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    
    if max_val > q99 + 10 * iqr:  # Very extreme outlier
        extreme_count = (df[col] > q99 + 10 * iqr).sum()
        print(f"   🚨 Extreme high values: {extreme_count} values > {q99 + 10 * iqr:.2f}")
        validation_issues.append(f"{col}: Has {extreme_count} extremely high outliers")
    
    if min_val < q01 - 10 * iqr:  # Very extreme outlier
        extreme_count = (df[col] < q01 - 10 * iqr).sum()
        print(f"   🚨 Extreme low values: {extreme_count} values < {q01 - 10 * iqr:.2f}")
        validation_issues.append(f"{col}: Has {extreme_count} extremely low outliers")

print("\\n🗓️ DATETIME VALIDATION:")
print("-" * 40)

current_time = datetime.now()
for col in datetime_cols:
    if df[col].dtype == 'object':
        continue
        
    print(f"\\n📅 Column: {col}")
    
    try:
        min_date = df[col].min()
        max_date = df[col].max()
        print(f"   Date range: {min_date} to {max_date}")
        
        # Check for future dates (might be invalid)
        future_count = (df[col] > pd.Timestamp(current_time)).sum()
        if future_count > 0:
            print(f"   ⚠️  Future dates: {future_count} ({future_count/len(df)*100:.1f}%)")
            
        # Check for very old dates (might be invalid)
        very_old_count = (df[col] < pd.Timestamp('1900-01-01')).sum()
        if very_old_count > 0:
            print(f"   ⚠️  Very old dates (< 1900): {very_old_count}")
            validation_issues.append(f"{col}: Has {very_old_count} dates before 1900 (potentially invalid)")
            
    except Exception as e:
        print(f"   ❌ Error validating dates: {e}")
        validation_issues.append(f"{col}: DateTime validation failed - {str(e)}")

print("\\n📝 TEXT/CATEGORICAL VALIDATION:")
print("-" * 40)

for col in text_cols[:8]:  # Check first 8 text columns (increased from 5)
    print(f"\\n📋 Column: {col}")
    
    # Check for empty strings
    empty_strings = (df[col] == '').sum()
    if empty_strings > 0:
        print(f"   ⚠️  Empty strings: {empty_strings} ({empty_strings/len(df)*100:.1f}%)")
    
    # Check for whitespace-only values
    if df[col].dtype == 'object':
        try:
            whitespace_only = df[col].str.strip().eq('').sum()
            if whitespace_only > 0:
                print(f"   ⚠️  Whitespace-only values: {whitespace_only}")
                validation_issues.append(f"{col}: Has {whitespace_only} whitespace-only values")
        except:
            pass
    
    # Check for suspiciously long values
    if df[col].dtype == 'object':
        try:
            max_length = df[col].str.len().max()
            if max_length > 1000:
                long_count = (df[col].str.len() > 1000).sum()
                print(f"   📏 Very long values: {long_count} values > 1000 characters")
        except:
            pass

print("\\n📊 CROSS-COLUMN CONSISTENCY:")
print("-" * 40)

# Look for common consistency issues
col_names_lower = [col.lower() for col in df.columns]

# Check for start/end date consistency
start_cols = [col for col in df.columns if any(word in col.lower() for word in ['start', 'begin', 'from'])]
end_cols = [col for col in df.columns if any(word in col.lower() for word in ['end', 'finish', 'to', 'until'])]

for start_col in start_cols:
    for end_col in end_cols:
        if start_col in datetime_cols and end_col in datetime_cols:
            try:
                invalid_dates = (df[start_col] > df[end_col]).sum()
                if invalid_dates > 0:
                    print(f"   ⚠️  Start > End dates: {start_col} > {end_col} in {invalid_dates} rows")
                    validation_issues.append(f"Date consistency: {start_col} > {end_col} in {invalid_dates} rows")
            except:
                pass

print("\\n" + "="*60)
print("📋 VALIDATION SUMMARY:")
print("="*60)

if validation_issues:
    print(f"\\n❌ Found {len(validation_issues)} validation issues:")
    for i, issue in enumerate(validation_issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\\n✅ No major validation issues detected!")

print(f"\\n📊 Overall Data Quality Score: {max(0, 100 - len(validation_issues)*10)}/100")

print("\\n" + "="*60)
print("✅ Data range validation complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return DataRangeValidationAnalysis
"""Temporal Trend Analysis Component.

Provides analysis of time-based trends in data.
"""
from typing import Dict, Any


class TemporalTrendAnalysis:
    """Analyze temporal trends in time-series data."""
    
    def __init__(self):
        """Initialize the temporal trend analysis component."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this analysis."""
        return {
            "name": "Temporal Trend Analysis",
            "description": "Analyze trends and patterns over time in datetime columns",
            "category": "time-series",
            "complexity": "intermediate", 
            "tags": ["time-series", "trends", "temporal", "patterns"],
            "estimated_runtime": "3-8 seconds",
            "icon": "📅"
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any] = None) -> bool:
        """Check if analysis can be performed on the data."""
        # Be permissive for debugging - let it run and show debug info
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate code for temporal trend analysis."""
        if not data_preview:
            return "# No data preview provided for temporal trend analysis"
            
        datetime_cols = data_preview.get('datetime_columns', [])
        if not datetime_cols:
            # Try to be more flexible - check for object columns that might contain dates
            object_cols = data_preview.get('object_columns', [])
            return f'''
# TEMPORAL TREND ANALYSIS - DEBUG
print("DEBUG: No datetime columns detected in data_preview")
print(f"Available columns: {{list(df.columns)}}")
print(f"Column dtypes: {{df.dtypes.to_dict()}}")
print(f"Datetime columns from preview: {datetime_cols}")
print(f"Object columns from preview: {object_cols}")

# Check for potential datetime columns manually
print("\\nChecking for potential datetime columns...")
for col in df.columns:
    if df[col].dtype == 'object':
        sample = df[col].dropna().head(5)
        print(f"Column '{{col}}' samples: {{sample.tolist()}}")
        
        # Try to convert to datetime
        try:
            pd.to_datetime(sample)
            print(f"  -> Could be datetime!")
        except:
            print(f"  -> Not datetime")

print("\\n⚠️  No datetime columns found. Please ensure your data has datetime columns or they are properly formatted.")
print("💡 Tip: Convert string dates to datetime using pd.to_datetime() before analysis")
'''
        
        return f'''
# TEMPORAL TREND ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("📅 TEMPORAL TREND ANALYSIS")
print("="*50)

# Use detected datetime columns
datetime_cols = {datetime_cols}
print(f"\\nAnalyzing temporal trends for columns: {{', '.join(datetime_cols)}}")

# Also check for datetime-like object columns
object_cols_with_dates = []
for col in df.select_dtypes(include=['object']).columns:
    if col not in datetime_cols and len(df[col].dropna()) > 0:
        sample_values = df[col].dropna().head(20)
        datetime_like = 0
        
        for val in sample_values:
            try:
                pd.to_datetime(val)
                datetime_like += 1
            except:
                pass
        
        if datetime_like > len(sample_values) * 0.7:
            object_cols_with_dates.append(col)

all_datetime_cols = datetime_cols + object_cols_with_dates
print(f"\\n📊 Found {{len(datetime_cols)}} datetime columns and {{len(object_cols_with_dates)}} potential datetime object columns")

if len(all_datetime_cols) == 0:
    print("❌ No datetime columns found for temporal trend analysis")
    print("💡 Tip: Ensure datetime columns are properly formatted")
else:
    # Get numeric columns for trend analysis (analyze more columns now that we optimized datetime parsing)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = all_numeric_cols[:10]  # Limit to 10 for performance and readability
    
    print(f"📊 Found {{len(all_numeric_cols)}} numeric columns total")
    if len(all_numeric_cols) > 10:
        print(f"📊 Analyzing first {{len(numeric_cols)}} numeric columns for temporal trends")
        print(f"    (Columns: {{', '.join(numeric_cols)}})")
        print(f"💡 Tip: To analyze all columns, remove the [:10] limit in the code")
    else:
        print(f"📊 Analyzing all {{len(numeric_cols)}} numeric columns for temporal trends")
    
    max_datetime_cols = min(3, len(all_datetime_cols))
    print(f"📅 Analyzing {{max_datetime_cols}} datetime columns: {{', '.join(all_datetime_cols[:max_datetime_cols])}}")
    
    for dt_col in all_datetime_cols[:max_datetime_cols]:
        print(f"\\n{'='*60}")
        print(f"📅 ANALYZING DATETIME COLUMN: {{dt_col}}")
        print('='*60)
        
        try:
            print(f"\\n🔍 DEBUG: Processing column '{{dt_col}}'")
            print(f"   Column dtype: {{df[dt_col].dtype}}")
            print(f"   Sample values: {{df[dt_col].dropna().head(3).tolist()}}")
            
            # Always convert to datetime, regardless of current type
            print(f"🔄 Converting '{{dt_col}}' to datetime...")
            
            # Try common date formats first to avoid warnings
            datetime_series = None
            date_formats = [
                '%Y-%m-%d',           # 2025-05-21
                '%Y-%m-%d %H:%M:%S',  # 2025-05-21 14:30:00
                '%Y/%m/%d',           # 2025/05/21
                '%d/%m/%Y',           # 21/05/2025
                '%m/%d/%Y',           # 05/21/2025
                '%d-%m-%Y',           # 21-05-2025
                '%m-%d-%Y',           # 05-21-2025
                '%Y%m%d',             # 20250521
                '%d.%m.%Y',           # 21.05.2025
                '%Y.%m.%d'            # 2025.05.21
            ]
            
            for fmt in date_formats:
                try:
                    datetime_series = pd.to_datetime(df[dt_col], format=fmt, errors='raise')
                    print(f"✅ Successfully converted using format: {{fmt}}")
                    break
                except (ValueError, TypeError):
                    continue
            
            # If no format worked, try automatic parsing
            if datetime_series is None:
                print("🔄 Trying automatic date parsing...")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    datetime_series = pd.to_datetime(df[dt_col], errors='coerce')
            
            print(f"✅ Conversion result:")
            print(f"   New dtype: {{datetime_series.dtype}}")
            print(f"   Sample converted values: {{datetime_series.dropna().head(3).tolist()}}")
            
            # Remove NaT values
            valid_datetime_mask = datetime_series.notna()
            datetime_series = datetime_series[valid_datetime_mask]
            
            if len(datetime_series) < 10:
                print("❌ Insufficient datetime data for trend analysis (need ≥10 points)")
                continue
            
            print(f"\\n📊 Datetime Column Summary:")
            print(f"   • Total entries: {{len(df[dt_col]):,}}")
            print(f"   • Valid datetime entries: {{len(datetime_series):,}}")
            
            # Safely calculate date range
            min_date = datetime_series.min()
            max_date = datetime_series.max()
            
            print(f"   • Date range: {{min_date}} to {{max_date}}")
            
            # Calculate time span safely
            try:
                time_span_days = (max_date - min_date).days
                print(f"   • Time span: {{time_span_days}} days")
            except Exception as e:
                print(f"   • Time span: Unable to calculate ({{type(min_date).__name__}}, {{type(max_date).__name__}})")
                print(f"     Debug: min_date={{min_date}}, max_date={{max_date}}")
            
            # Create working dataframe
            work_df = df[valid_datetime_mask].copy()
            work_df['datetime_col'] = datetime_series
            work_df = work_df.sort_values('datetime_col')
            
            # Extract time components safely
            try:
                work_df['year'] = work_df['datetime_col'].dt.year
                work_df['month'] = work_df['datetime_col'].dt.month
                work_df['day_of_week'] = work_df['datetime_col'].dt.dayofweek
                work_df['quarter'] = work_df['datetime_col'].dt.quarter
                
                print(f"\\n📊 Temporal Coverage:")
                print(f"   • Years covered: {{work_df['year'].nunique()}}")
                print(f"   • Months with data: {{work_df['month'].nunique()}}")
                print(f"   • Days of week: {{work_df['day_of_week'].nunique()}}")
            except Exception as e:
                print(f"❌ Error extracting time components: {{e}}")
                print(f"   datetime_col dtype: {{work_df['datetime_col'].dtype}}")
                print(f"   datetime_col sample: {{work_df['datetime_col'].head(3).tolist()}}")
                continue
            
            # Analyze trends for numeric columns
            if len(numeric_cols) > 0:
                print(f"\\n📈 TEMPORAL TRENDS FOR NUMERIC VARIABLES:")
                print("-" * 50)
                
                for num_col in numeric_cols:
                    if num_col in work_df.columns and not work_df[num_col].isna().all():
                        print(f"\\n📊 Trends for: {{num_col}}")
                        
                        # Monthly trends
                        monthly_stats = work_df.groupby('month')[num_col].agg(['mean', 'count']).reset_index()
                        if len(monthly_stats) > 1:
                            print("   📅 Monthly averages:")
                            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            for _, row in monthly_stats.iterrows():
                                month_name = month_names[int(row['month'])-1]
                                print(f"     {{month_name}}: avg={{row['mean']:.2f}}, count={{int(row['count'])}}")
                        
                        # Day of week trends
                        dow_stats = work_df.groupby('day_of_week')[num_col].agg(['mean', 'count']).reset_index()
                        if len(dow_stats) > 1:
                            print("   📅 Day of week averages:")
                            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                            for _, row in dow_stats.iterrows():
                                dow_name = dow_names[int(row['day_of_week'])]
                                print(f"     {{dow_name}}: avg={{row['mean']:.2f}}, count={{int(row['count'])}}")
                        
                        # Yearly trends (if multiple years)
                        if work_df['year'].nunique() > 1:
                            yearly_stats = work_df.groupby('year')[num_col].agg(['mean', 'count']).reset_index()
                            print("   📅 Yearly averages:")
                            for _, row in yearly_stats.iterrows():
                                print(f"     {{int(row['year'])}}: avg={{row['mean']:.2f}}, count={{int(row['count'])}}")
        
        except Exception as e:
            print(f"❌ Error analyzing {{dt_col}}: {{str(e)}}")
            continue

print("\\n" + "="*50) 
print("✅ Temporal trend analysis complete!")
print("="*50)
'''


def get_component():
    """Return the analysis component."""
    return TemporalTrendAnalysis
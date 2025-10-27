"""Temporal Trend Analysis Component.

Provides analysis of time-based trends in data.
"""
from typing import Dict, Any


class TemporalTrendAnalysis:
    """Analyze temporal trends in time-series data."""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "temporal_trend_analysis",
            "display_name": "Temporal Trend Analysis",
            "description": "Analyze time-based trends (daily/weekly/monthly patterns)",
            "category": "time_series",
            "complexity": "intermediate", 
            "tags": ["time-series", "trends", "temporal", "patterns"],
            "estimated_runtime": "3-8 seconds",
            "icon": "📅"
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any] = None) -> bool:
        """Check if analysis can be performed on the data."""
        if not data_preview:
            return True
        datetime_cols = data_preview.get('datetime_columns', [])
        return len(datetime_cols) > 0
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate code for temporal trend analysis."""
        if not data_preview:
            return "# No data preview provided for temporal trend analysis"
            
        datetime_cols = data_preview.get('datetime_columns', [])
        if not datetime_cols:
            return "# No datetime columns found for temporal trend analysis"
        
        return f'''
# ===== TEMPORAL TREND ANALYSIS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📅 TEMPORAL TREND ANALYSIS")
print("="*60)

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

# Check for datetime-like object columns
for col in df.select_dtypes(include=['object']).columns:
    if len(df[col].dropna()) > 0:
        sample_values = df[col].dropna().head(20)
        datetime_like = 0
        
        for val in sample_values:
            try:
                pd.to_datetime(val)
                datetime_like += 1
            except:
                pass
        
        if datetime_like > len(sample_values) * 0.7:  # More than 70% seem to be datetime
            object_cols_with_dates.append(col)

all_datetime_cols = datetime_cols + object_cols_with_dates
print(f"\\n📊 Found {len(datetime_cols)} datetime columns and {len(object_cols_with_dates)} potential datetime object columns")

if len(all_datetime_cols) == 0:
    print("❌ No datetime columns found for temporal analysis")
    print("💡 Tip: Ensure datetime columns are properly formatted with pd.to_datetime()")
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Found {len(numeric_cols)} numeric columns for trend analysis")
    
    for dt_col in all_datetime_cols:
        print(f"\\n{'='*60}")
        print(f"📅 ANALYZING DATETIME COLUMN: {dt_col}")
        print('='*60)
        
        # Convert to datetime if needed
        try:
            if dt_col in object_cols_with_dates:
                datetime_series = pd.to_datetime(df[dt_col], errors='coerce')
            else:
                datetime_series = df[dt_col].copy()
            
            # Remove NaT values
            valid_datetime_mask = datetime_series.notna()
            datetime_series = datetime_series[valid_datetime_mask]
            
            if len(datetime_series) == 0:
                print("❌ No valid datetime values found")
                continue
            
            print(f"\\n📊 Datetime Column Overview:")
            print(f"   • Valid datetime entries: {len(datetime_series):,}")
            print(f"   • Date range: {datetime_series.min()} to {datetime_series.max()}")
            print(f"   • Time span: {(datetime_series.max() - datetime_series.min()).days} days")
            
            # Create a working dataframe with datetime and numeric columns
            work_df = df[valid_datetime_mask].copy()
            work_df['datetime_col'] = datetime_series
            
            # Basic temporal patterns
            work_df['year'] = work_df['datetime_col'].dt.year
            work_df['month'] = work_df['datetime_col'].dt.month
            work_df['day_of_week'] = work_df['datetime_col'].dt.dayofweek
            work_df['hour'] = work_df['datetime_col'].dt.hour
            
            print(f"\\n📈 Temporal Distribution:")
            print(f"   • Years covered: {work_df['year'].nunique()}")
            print(f"   • Months with data: {work_df['month'].nunique()}")
            print(f"   • Days of week: {work_df['day_of_week'].nunique()}")
            
            # Analyze trends with numeric columns
            if len(numeric_cols) > 0:
                print(f"\\n🔍 TREND ANALYSIS WITH NUMERIC COLUMNS:")
                print("-" * 50)
                
                for num_col in numeric_cols[:5]:  # Limit to first 5 for performance
                    if num_col in work_df.columns:
                        print(f"\\n📊 Trends for: {num_col}")
                        
                        # Monthly trends
                        monthly_stats = work_df.groupby('month')[num_col].agg(['mean', 'count']).round(2)
                        print("   Monthly patterns:")
                        for month, stats in monthly_stats.iterrows():
                            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
                            print(f"     {month_name}: avg={stats['mean']:.2f}, count={int(stats['count'])}")
                        
                        # Day of week patterns
                        dow_stats = work_df.groupby('day_of_week')[num_col].agg(['mean', 'count']).round(2)
                        print("   Day of week patterns:")
                        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        for dow, stats in dow_stats.iterrows():
                            print(f"     {dow_names[dow]}: avg={stats['mean']:.2f}, count={int(stats['count'])}")
                        
                        # Yearly trends if multiple years
                        if work_df['year'].nunique() > 1:
                            yearly_stats = work_df.groupby('year')[num_col].agg(['mean', 'count']).round(2)
                            print("   Yearly trends:")
                            for year, stats in yearly_stats.iterrows():
                                print(f"     {int(year)}: avg={stats['mean']:.2f}, count={int(stats['count'])}")
            
            # Time gaps analysis
            if len(datetime_series) > 1:
                datetime_sorted = datetime_series.sort_values()
                time_diffs = datetime_sorted.diff().dropna()
                
                print(f"\\n⏱️  TIME INTERVALS ANALYSIS:")
                print(f"   • Most common interval: {time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 'N/A'}")
                print(f"   • Average interval: {time_diffs.mean()}")
                print(f"   • Max gap: {time_diffs.max()}")
                print(f"   • Min gap: {time_diffs.min()}")
                
                # Identify large gaps
                large_gaps = time_diffs[time_diffs > time_diffs.quantile(0.95)]
                if len(large_gaps) > 0:
                    print(f"   • Large gaps found: {len(large_gaps)} intervals > 95th percentile")
        
        except Exception as e:
            print(f"❌ Error analyzing column {dt_col}: {str(e)}")
            continue

print("\\n" + "="*60) 
print("✅ Temporal trend analysis complete!")
print("="*60)
'''
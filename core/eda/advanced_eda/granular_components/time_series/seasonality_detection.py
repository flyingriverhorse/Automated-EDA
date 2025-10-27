"""Seasonality Detection Component.

Provides analysis of seasonal patterns in time-series data.
"""
from typing import Dict, Any, List, Optional


class SeasonalityDetectionAnalysis:
    """Detect and analyze seasonal patterns in time-series data."""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "seasonality_detection",
            "display_name": "Seasonality Detection",
            "description": "Detect seasonality, cycles, and recurring patterns in time-series data",
            "category": "time_series",
            "complexity": "advanced",
            "tags": ["seasonality", "cycles", "fourier", "decomposition", "patterns"],
            "estimated_runtime": "5-15 seconds",
            "icon": "🌊"
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any] = None) -> bool:
        """Check if analysis can be performed on the data."""
        # Be permissive for debugging - let it run and show debug info
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate code for seasonality detection."""
        if not data_preview:
            return "# No data preview provided for seasonality analysis"
            
        datetime_cols = data_preview.get('datetime_columns', [])
        object_cols = data_preview.get('object_columns', [])
        
        if not datetime_cols:
            return f'''
# SEASONALITY ANALYSIS - DEBUG
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

print("\\n⚠️  No datetime columns found for seasonality analysis.")
print("💡 Tip: Convert string dates to datetime using pd.to_datetime() before analysis")
'''
        
        return f'''
# SEASONALITY DETECTION ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*50)
print("🕐 SEASONALITY DETECTION ANALYSIS")
print("="*50)

datetime_cols = {datetime_cols}
print(f"\\nAnalyzing seasonality for columns: {{', '.join(datetime_cols)}}")

for col in datetime_cols:
    if col not in df.columns:
        print(f"\\n⚠️  Column '{{col}}' not found in dataset")
        continue
    
    print(f"\\n📅 ANALYZING: {{col}}")
    print("-" * 40)
    
    # Convert to datetime using efficient format detection
    print(f"🔄 Converting '{{col}}' to datetime...")
    
    # Try common date formats first to avoid warnings
    date_series = None
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
            date_series = pd.to_datetime(df[col], format=fmt, errors='raise')
            print(f"✅ Successfully converted using format: {{fmt}}")
            break
        except (ValueError, TypeError):
            continue
    
    # If no format worked, try automatic parsing
    if date_series is None:
        print("🔄 Trying automatic date parsing...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            date_series = pd.to_datetime(df[col], errors='coerce')
    
    # Remove any NaT (Not a Time) values
    date_series = date_series.dropna()
    
    if len(date_series) == 0:
        print(f"❌ No valid dates found in column '{{col}}'")
        continue
        
    print(f"📊 Valid dates: {{len(date_series):,}} / {{len(df):,}} ({{len(date_series)/len(df)*100:.1f}}%)")
    print(f"📅 Date range: {{date_series.min()}} to {{date_series.max()}}")
    print(f"⏱️  Time span: {{(date_series.max() - date_series.min()).days}} days")
    
    # Create seasonality visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Seasonality Analysis: {{col}}', fontsize=16, fontweight='bold')
    
    # 1. Monthly seasonality
    monthly_counts = date_series.dt.month.value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[0,0].bar(monthly_counts.index, monthly_counts.values, color='skyblue')
    axes[0,0].set_title('Monthly Seasonality')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xticks(range(1, 13))
    axes[0,0].set_xticklabels(month_names, rotation=45)
    
    # 2. Quarterly seasonality
    quarterly_counts = date_series.dt.quarter.value_counts().sort_index()
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
    axes[0,1].bar(quarterly_counts.index, quarterly_counts.values, color='lightcoral')
    axes[0,1].set_title('Quarterly Seasonality')
    axes[0,1].set_xlabel('Quarter')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_xticks(range(1, 5))
    axes[0,1].set_xticklabels(quarter_names)
    
    # 3. Day of week seasonality
    dow_counts = date_series.dt.day_name().value_counts()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = dow_counts.reindex([d for d in dow_order if d in dow_counts.index])
    axes[1,0].bar(range(len(dow_counts)), dow_counts.values, color='lightgreen')
    axes[1,0].set_title('Day of Week Seasonality')
    axes[1,0].set_xlabel('Day of Week')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_xticks(range(len(dow_counts)))
    axes[1,0].set_xticklabels([d[:3] for d in dow_counts.index], rotation=45)
    
    # 4. Hour seasonality (if available)
    if date_series.dt.hour.nunique() > 1:
        hour_counts = date_series.dt.hour.value_counts().sort_index()
        axes[1,1].bar(hour_counts.index, hour_counts.values, color='gold')
        axes[1,1].set_title('Hourly Seasonality')
        axes[1,1].set_xlabel('Hour of Day')
        axes[1,1].set_ylabel('Count')
    else:
        # Show day of month pattern instead
        dom_counts = date_series.dt.day.value_counts().sort_index()
        axes[1,1].bar(dom_counts.index, dom_counts.values, color='gold')
        axes[1,1].set_title('Day of Month Pattern')
        axes[1,1].set_xlabel('Day of Month')
        axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Seasonality statistics
    print(f"\\n🌟 SEASONALITY INSIGHTS:")
    print(f"   • Peak month: {{date_series.dt.month_name().mode().iloc[0] if not date_series.dt.month_name().mode().empty else 'N/A'}}")
    print(f"   • Peak quarter: Q{{date_series.dt.quarter.mode().iloc[0] if not date_series.dt.quarter.mode().empty else 'N/A'}}")
    print(f"   • Peak day of week: {{date_series.dt.day_name().mode().iloc[0] if not date_series.dt.day_name().mode().empty else 'N/A'}}")
    
    # Calculate seasonal variations
    monthly_std = monthly_counts.std()
    monthly_mean = monthly_counts.mean()
    seasonal_variation = (monthly_std / monthly_mean) * 100 if monthly_mean > 0 else 0
    
    print(f"   • Monthly variation: {{seasonal_variation:.1f}}% (higher = more seasonal)")
    
    if seasonal_variation > 50:
        print("   🔥 HIGH seasonality detected!")
    elif seasonal_variation > 20:
        print("   📊 MODERATE seasonality detected")
    else:
        print("   📏 LOW seasonality - fairly uniform distribution")

print("\\n✅ Seasonality analysis complete!")
'''
        return f'''
# SEASONALITY DETECTION ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*50)
print("🕐 SEASONALITY DETECTION ANALYSIS")
print("="*50)

datetime_cols = {datetime_cols}
print(f"\\nAnalyzing seasonality for columns: {{', '.join(datetime_cols)}}")

for col in datetime_cols:
    if col not in df.columns:
        print(f"\\n⚠️  Column '{{col}}' not found in dataset")
        continue
    
    print(f"\\n📅 ANALYZING: {{col}}")
    print("-" * 40)
    
    # Handle different datetime formats
    date_series = pd.to_datetime(df[col], errors='coerce')
    
    if date_series.isna().all():
        print(f"   ❌ Cannot parse '{{col}}' as datetime")
        continue
    
    # Remove missing dates
    valid_dates = date_series.dropna()
    
    if len(valid_dates) < 12:
        print(f"   ❌ Not enough data points ({{len(valid_dates)}}) for seasonality analysis")
        continue
    
    print(f"   ✅ Valid dates: {{len(valid_dates):,}} / {{len(df):,}}")
    print(f"   📆 Date range: {{valid_dates.min().date()}} to {{valid_dates.max().date()}}")
    print(f"   ⏰ Time span: {{(valid_dates.max() - valid_dates.min()).days}} days")
    
    # Create temporal features
    temporal_df = pd.DataFrame({{
        'date': valid_dates,
        'year': valid_dates.dt.year,
        'month': valid_dates.dt.month,
        'day': valid_dates.dt.day,
        'weekday': valid_dates.dt.dayofweek,
        'quarter': valid_dates.dt.quarter,
        'week': valid_dates.dt.isocalendar().week
    }})
    
    # Count occurrences for each temporal unit
    print("\\n📊 TEMPORAL PATTERN COUNTS:")
    
    # Monthly patterns
    monthly_counts = temporal_df['month'].value_counts().sort_index()
    print(f"   📅 Monthly distribution (Range: {{monthly_counts.min()}} - {{monthly_counts.max()}})")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month_num, count in monthly_counts.items():
        print(f"      {{month_names[month_num-1]}}: {{count:>4}}")
    
    # Day of week patterns
    weekday_counts = temporal_df['weekday'].value_counts().sort_index()
    print(f"\\n   📅 Weekday distribution (Range: {{weekday_counts.min()}} - {{weekday_counts.max()}})")
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day_num, count in weekday_counts.items():
        print(f"      {{weekday_names[day_num]}}: {{count:>4}}")
    
    # Quarterly patterns
    quarterly_counts = temporal_df['quarter'].value_counts().sort_index()
    print(f"\\n   📅 Quarterly distribution (Range: {{quarterly_counts.min()}} - {{quarterly_counts.max()}})")
    for quarter, count in quarterly_counts.items():
        print(f"      Q{{quarter}}: {{count:>4}}")
    
    # VISUALIZATION
    print("\\n📈 GENERATING SEASONALITY PLOTS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Seasonality Analysis: {{col}}', fontsize=16, fontweight='bold')
    
    # Monthly seasonality
    ax1 = axes[0, 0]
    monthly_counts.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_title('Monthly Pattern', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Count')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(month_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Weekday seasonality
    ax2 = axes[0, 1]
    weekday_counts.plot(kind='bar', ax=ax2, color='lightgreen', alpha=0.8)
    ax2.set_title('Day of Week Pattern', fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(weekday_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Quarterly seasonality
    ax3 = axes[1, 0]
    quarterly_counts.plot(kind='bar', ax=ax3, color='coral', alpha=0.8)
    ax3.set_title('Quarterly Pattern', fontweight='bold')
    ax3.set_xlabel('Quarter')
    ax3.set_ylabel('Count')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([f'Q{{i+1}}' for i in range(4)], rotation=0)
    ax3.grid(True, alpha=0.3)
    
    # Time series plot (sample if too many points)
    ax4 = axes[1, 1]
    
    # Group by date and count occurrences
    daily_counts = temporal_df.groupby(temporal_df['date'].dt.date).size()
    
    if len(daily_counts) > 365:
        # Sample data if too many points
        sample_size = min(365, len(daily_counts))
        daily_sample = daily_counts.sample(n=sample_size).sort_index()
        ax4.plot(daily_sample.index, daily_sample.values, alpha=0.7, linewidth=1)
        ax4.set_title(f'Time Series (Sample of {{sample_size}} days)', fontweight='bold')
    else:
        ax4.plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=1)
        ax4.set_title('Time Series - Daily Counts', fontweight='bold')
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Daily Count')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # SEASONALITY STRENGTH ANALYSIS
    print("\\n🔍 SEASONALITY STRENGTH ANALYSIS:")
    
    # Calculate coefficient of variation for each seasonal component
    monthly_cv = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
    weekday_cv = weekday_counts.std() / weekday_counts.mean() if weekday_counts.mean() > 0 else 0
    quarterly_cv = quarterly_counts.std() / quarterly_counts.mean() if quarterly_counts.mean() > 0 else 0
    
    print(f"   📊 Monthly seasonality strength: {{monthly_cv:.3f}}")
    print(f"   📊 Weekday seasonality strength: {{weekday_cv:.3f}}")
    print(f"   📊 Quarterly seasonality strength: {{quarterly_cv:.3f}}")
    
    # Interpretation
    def interpret_seasonality(cv_value):
        if cv_value > 0.3:
            return "Strong seasonality detected"
        elif cv_value > 0.15:
            return "Moderate seasonality detected"
        elif cv_value > 0.05:
            return "Weak seasonality detected"
        else:
            return "No clear seasonality"
    
    print("\\n💡 SEASONALITY INTERPRETATION:")
    print(f"   🗓️  Monthly: {{interpret_seasonality(monthly_cv)}}")
    print(f"   📅 Weekday: {{interpret_seasonality(weekday_cv)}}")
    print(f"   🗓️  Quarterly: {{interpret_seasonality(quarterly_cv)}}")

print("\\n" + "="*50)
print("✅ Seasonality analysis complete!")
print("="*50)
'''


def get_component():
    """Return the analysis component."""
    return SeasonalityDetectionAnalysis
    return SeasonalityDetectionAnalysis
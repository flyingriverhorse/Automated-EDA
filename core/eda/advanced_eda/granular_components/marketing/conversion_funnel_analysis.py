"""Conversion Funnel Analysis Component.

Focused analysis component for marketing conversion funnels
analyzing step-by-step user journey and drop-off rates.
"""
from typing import Dict, Any


class ConversionFunnelAnalysis:
    """Focused component for conversion funnel analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "conversion_funnel_analysis",
            "display_name": "Conversion Funnel Analysis",
            "description": "Analyze conversion funnels: step-by-step user journey, drop-off rates, and bottleneck identification",
            "category": "marketing",
            "complexity": "advanced",
            "required_data_types": ["numeric"],
            "estimated_runtime": "15-25 seconds",
            "icon": "🔽",
            "tags": ["marketing", "funnel", "conversion", "dropoff", "journey", "optimization"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has funnel-related columns"""
        # For now, return True to allow all datasets (marketing analyses are flexible)
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate conversion funnel analysis code with column selection support"""
        
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
FUNNEL_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect funnel columns (user can override these)
FUNNEL_MAPPING = {}
possible_mappings = {
    'step_1_views': ['landing_page_views', 'homepage_visits', 'impressions', 'visits', 'sessions'],
    'step_2_engagement': ['product_views', 'page_views', 'clicks', 'engagement'],
    'step_3_intent': ['add_to_cart', 'signups', 'leads', 'downloads'],
    'step_4_conversion': ['purchases', 'conversions', 'sales', 'completions'],
    'user_id': ['user_id', 'customer_id', 'session_id', 'visitor_id'],
    'timestamp': ['date', 'timestamp', 'created_at', 'event_time'],
    'campaign': ['campaign', 'source', 'utm_source', 'channel'],
    'stage': ['stage', 'step', 'funnel_step', 'phase'],
    'event_type': ['event', 'event_type', 'action', 'activity']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in FUNNEL_MAPPING:
                FUNNEL_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== CONVERSION FUNNEL ANALYSIS ===")
print("Marketing conversion funnel and drop-off analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 FUNNEL COLUMN MAPPING:")
for step, col in FUNNEL_MAPPING.items():
    if col in df.columns:
        print(f"   • {{step.replace('_', ' ').title()}}: {{col}}")
print()

# Get numeric columns for funnel metrics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   Funnel analysis requires numeric data.")
else:
    print(f"📊 ANALYZING {{len(numeric_cols)}} NUMERIC COLUMNS")
    print()
    
    # 1. Basic Funnel Metrics
    print("🔽 CONVERSION FUNNEL METRICS")
    
    funnel_steps = []
    funnel_data = {{}}
    
    # Try to identify funnel steps from data
    step_columns = []
    for i in range(1, 6):  # Look for up to 5 funnel steps
        step_key = f'step_{{i}}_views' if i == 1 else f'step_{{i}}_engagement' if i == 2 else f'step_{{i}}_intent' if i == 3 else f'step_{{i}}_conversion'
        if step_key in FUNNEL_MAPPING and FUNNEL_MAPPING[step_key] in df.columns:
            step_columns.append((f"Step {{i}}", FUNNEL_MAPPING[step_key]))
    
    # If no predefined steps, try to infer from numeric columns
    if not step_columns:
        # Look for columns that might represent funnel steps
        potential_steps = []
        funnel_keywords = ['view', 'visit', 'click', 'signup', 'purchase', 'conversion', 'lead', 'cart', 'checkout']
        
        for col in numeric_cols:
            col_lower = col.lower()
            for keyword in funnel_keywords:
                if keyword in col_lower:
                    potential_steps.append((col.replace('_', ' ').title(), col))
                    break
        
        # If we found potential steps, use top 5
        step_columns = potential_steps[:5]
    
    # If we still don't have steps, use top numeric columns
    if not step_columns:
        step_columns = [(col.replace('_', ' ').title(), col) for col in numeric_cols[:5]]
    
    print(f"   📊 Identified {{len(step_columns)}} funnel steps:")
    for step_name, col_name in step_columns:
        print(f"      • {{step_name}}: {{col_name}}")
    print()
    
    # Calculate funnel metrics
    if step_columns:
        print("📈 FUNNEL PERFORMANCE ANALYSIS")
        
        funnel_summary = []
        previous_step_value = None
        
        for i, (step_name, col_name) in enumerate(step_columns):
            step_total = df[col_name].sum()
            step_avg = df[col_name].mean()
            step_records = (df[col_name] > 0).sum()
            
            step_data = {{
                'step': step_name,
                'column': col_name,
                'total': step_total,
                'average': step_avg,
                'records_with_activity': step_records,
                'percentage_of_total_records': (step_records / len(df)) * 100
            }}
            
            # Calculate drop-off from previous step
            if previous_step_value is not None:
                drop_off = previous_step_value - step_total if previous_step_value > step_total else 0
                drop_off_rate = (drop_off / previous_step_value) * 100 if previous_step_value > 0 else 0
                conversion_rate = (step_total / previous_step_value) * 100 if previous_step_value > 0 else 0
                
                step_data['drop_off'] = drop_off
                step_data['drop_off_rate'] = drop_off_rate
                step_data['conversion_rate'] = conversion_rate
            
            funnel_summary.append(step_data)
            previous_step_value = step_total
        
        # Display funnel summary
        print("\\n   🎯 STEP-BY-STEP FUNNEL BREAKDOWN:")
        for i, step_data in enumerate(funnel_summary):
            print(f"\\n   {{i+1}}. {{step_data['step'].upper()}}")
            print(f"      Total: {{step_data['total']:,.0f}}")
            print(f"      Average per record: {{step_data['average']:,.1f}}")
            print(f"      Records with activity: {{step_data['records_with_activity']:,}} ({{step_data['percentage_of_total_records']:.1f}}%)")
            
            if 'drop_off_rate' in step_data:
                print(f"      Drop-off from previous: {{step_data['drop_off']:,.0f}} ({{step_data['drop_off_rate']:.1f}}%)")
                print(f"      Conversion from previous: {{step_data['conversion_rate']:.1f}}%")
        
        print()
        
        # Overall funnel performance
        if len(funnel_summary) >= 2:
            first_step = funnel_summary[0]['total']
            last_step = funnel_summary[-1]['total']
            overall_conversion = (last_step / first_step) * 100 if first_step > 0 else 0
            
            print(f"🏆 OVERALL FUNNEL PERFORMANCE")
            print(f"   📊 End-to-end conversion rate: {{overall_conversion:.2f}}%")
            print(f"   📉 Total drop-off: {{first_step - last_step:,.0f}} ({{100 - overall_conversion:.1f}}%)")
            
            # Identify biggest drop-off step
            max_drop_step = max([s for s in funnel_summary if 'drop_off_rate' in s], 
                              key=lambda x: x['drop_off_rate'], default=None)
            if max_drop_step:
                print(f"   ⚠️  Biggest drop-off: {{max_drop_step['step']}} ({{max_drop_step['drop_off_rate']:.1f}}%)")
            
            print()
    
    # 2. Funnel Efficiency Analysis
    print("⚡ FUNNEL EFFICIENCY ANALYSIS")
    
    # Calculate efficiency metrics for each step transition
    if len(step_columns) >= 2:
        efficiency_analysis = []
        
        for i in range(len(step_columns) - 1):
            current_step = step_columns[i]
            next_step = step_columns[i + 1]
            
            current_col = current_step[1]
            next_col = next_step[1]
            
            # Records where both steps have data
            both_steps_data = df[(df[current_col] > 0) & (df[next_col] > 0)]
            
            if not both_steps_data.empty:
                # Efficiency ratio: next_step / current_step for records with both
                efficiency_ratios = both_steps_data[next_col] / both_steps_data[current_col]
                avg_efficiency = efficiency_ratios.mean()
                median_efficiency = efficiency_ratios.median()
                
                efficiency_analysis.append({{
                    'transition': f"{{current_step[0]}} → {{next_step[0]}}",
                    'records_analyzed': len(both_steps_data),
                    'avg_efficiency': avg_efficiency,
                    'median_efficiency': median_efficiency,
                    'top_10_pct_efficiency': efficiency_ratios.quantile(0.9)
                }})
        
        if efficiency_analysis:
            print("   📊 Step-by-step efficiency ratios:")
            for analysis in efficiency_analysis:
                print(f"\\n   🔄 {{analysis['transition']}}:")
                print(f"      Records analyzed: {{analysis['records_analyzed']:,}}")
                print(f"      Average efficiency: {{analysis['avg_efficiency']:.3f}}")
                print(f"      Median efficiency: {{analysis['median_efficiency']:.3f}}")
                print(f"      Top 10% efficiency: {{analysis['top_10_pct_efficiency']:.3f}}")
            
            print()
    
    # 3. Segment Analysis (if campaign/source data available)
    segment_analysis_done = False
    
    if 'campaign' in FUNNEL_MAPPING and FUNNEL_MAPPING['campaign'] in df.columns:
        campaign_col = FUNNEL_MAPPING['campaign']
        print("🎯 FUNNEL PERFORMANCE BY CAMPAIGN")
        
        campaigns = df[campaign_col].value_counts().head(10).index
        
        for campaign in campaigns:
            campaign_data = df[df[campaign_col] == campaign]
            if not campaign_data.empty and len(step_columns) >= 2:
                first_step_total = campaign_data[step_columns[0][1]].sum()
                last_step_total = campaign_data[step_columns[-1][1]].sum()
                campaign_conversion = (last_step_total / first_step_total) * 100 if first_step_total > 0 else 0
                
                print(f"   📊 {{campaign}}: {{campaign_conversion:.1f}}% conversion ({{first_step_total:,.0f}} → {{last_step_total:,.0f}})")
        
        segment_analysis_done = True
        print()
    
    # 4. Time-based Funnel Analysis (if timestamp available)
    if 'timestamp' in FUNNEL_MAPPING and FUNNEL_MAPPING['timestamp'] in df.columns:
        timestamp_col = FUNNEL_MAPPING['timestamp']
        
        try:
            # Try to convert to datetime
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            if df[timestamp_col].notna().sum() > 0:
                print("📅 TIME-BASED FUNNEL TRENDS")
                
                # Group by time period (daily if we have enough data)
                df['date'] = df[timestamp_col].dt.date
                daily_funnel = df.groupby('date').agg({{
                    col[1]: 'sum' for col in step_columns
                }})
                
                if len(daily_funnel) > 1:
                    # Calculate daily conversion rates
                    if len(step_columns) >= 2:
                        first_step_col = step_columns[0][1]
                        last_step_col = step_columns[-1][1]
                        daily_funnel['conversion_rate'] = (daily_funnel[last_step_col] / daily_funnel[first_step_col]) * 100
                        
                        print(f"   📈 Daily conversion rate range: {{daily_funnel['conversion_rate'].min():.1f}}% - {{daily_funnel['conversion_rate'].max():.1f}}%")
                        print(f"   📊 Average daily conversion: {{daily_funnel['conversion_rate'].mean():.1f}}%")
                        
                        # Identify best and worst days
                        best_day = daily_funnel['conversion_rate'].idxmax()
                        worst_day = daily_funnel['conversion_rate'].idxmin()
                        print(f"   🏆 Best performing day: {{best_day}} ({{daily_funnel.loc[best_day, 'conversion_rate']:.1f}}%)")
                        print(f"   📉 Lowest performing day: {{worst_day}} ({{daily_funnel.loc[worst_day, 'conversion_rate']:.1f}}%)")
                
                print()
        except Exception as e:
            print(f"   ⚠️ Could not analyze time-based trends: {{str(e)}}")
            print()
    
    # 5. Advanced Funnel Insights
    print("🔍 ADVANCED FUNNEL INSIGHTS")
    
    # Identify high-performance records
    if len(step_columns) >= 2:
        first_col = step_columns[0][1]
        last_col = step_columns[-1][1]
        
        # Records with activity in first and last steps
        active_records = df[(df[first_col] > 0) & (df[last_col] > 0)]
        if not active_records.empty:
            conversion_efficiency = active_records[last_col] / active_records[first_col]
            
            # Top performers (top 10%)
            top_threshold = conversion_efficiency.quantile(0.9)
            top_performers = active_records[conversion_efficiency >= top_threshold]
            
            print(f"   🏆 High-converting records: {{len(top_performers)}} ({{len(top_performers)/len(active_records)*100:.1f}}%)")
            print(f"   💎 Top 10% conversion efficiency threshold: {{top_threshold:.3f}}")
            
            # Characteristics of top performers
            if len(top_performers) > 5:
                print(f"   📊 Top performers characteristics:")
                for col in numeric_cols[:5]:
                    if col in top_performers.columns:
                        avg_all = df[col].mean()
                        avg_top = top_performers[col].mean()
                        if avg_all > 0:
                            performance_lift = ((avg_top - avg_all) / avg_all) * 100
                            print(f"      • {{col}}: {{performance_lift:+.1f}}% vs average")
        
        print()
    
    # 6. Bottleneck Identification
    print("🚧 BOTTLENECK IDENTIFICATION")
    
    if len(funnel_summary) >= 2:
        # Find steps with highest drop-off rates
        steps_with_dropoff = [s for s in funnel_summary if 'drop_off_rate' in s]
        if steps_with_dropoff:
            # Sort by drop-off rate
            steps_with_dropoff.sort(key=lambda x: x['drop_off_rate'], reverse=True)
            
            print("   ⚠️ Major bottlenecks (highest drop-off rates):")
            for i, step in enumerate(steps_with_dropoff[:3]):
                print(f"      {{i+1}}. {{step['step']}}: {{step['drop_off_rate']:.1f}}% drop-off")
                if step['drop_off_rate'] > 70:
                    print(f"         🚨 CRITICAL: Very high drop-off rate!")
                elif step['drop_off_rate'] > 50:
                    print(f"         ⚠️ HIGH: Significant optimization opportunity")
        
        # Conversion rate benchmarks
        print("\\n   📊 Conversion rate benchmarks:")
        for step in steps_with_dropoff:
            if 'conversion_rate' in step:
                rate = step['conversion_rate']
                if rate > 80:
                    status = "🟢 Excellent"
                elif rate > 60:
                    status = "🟡 Good"
                elif rate > 40:
                    status = "🟠 Fair"
                else:
                    status = "🔴 Needs Improvement"
                
                print(f"      • {{step['step']}}: {{rate:.1f}}% {{status}}")
    
    print()
    
    # 7. Optimization Recommendations
    print("💡 FUNNEL OPTIMIZATION RECOMMENDATIONS")
    
    if len(funnel_summary) >= 2:
        # Performance-based recommendations
        steps_with_dropoff = [s for s in funnel_summary if 'drop_off_rate' in s and s['drop_off_rate'] > 0]
        
        if steps_with_dropoff:
            highest_dropoff = max(steps_with_dropoff, key=lambda x: x['drop_off_rate'])
            print(f"   🎯 Priority: Focus on {{highest_dropoff['step']}} ({{highest_dropoff['drop_off_rate']:.1f}}% drop-off)")
            
            # Step-specific recommendations
            step_name_lower = highest_dropoff['step'].lower()
            if 'view' in step_name_lower or 'visit' in step_name_lower:
                print("   💡 Improve: Landing page optimization, load speed, mobile responsiveness")
            elif 'click' in step_name_lower or 'engagement' in step_name_lower:
                print("   💡 Improve: Call-to-action placement, content relevance, user experience")
            elif 'signup' in step_name_lower or 'lead' in step_name_lower:
                print("   💡 Improve: Form optimization, value proposition, trust signals")
            elif 'purchase' in step_name_lower or 'conversion' in step_name_lower:
                print("   💡 Improve: Checkout process, payment options, security assurance")
        
        # General recommendations
        overall_conversion = (funnel_summary[-1]['total'] / funnel_summary[0]['total']) * 100 if funnel_summary[0]['total'] > 0 else 0
        
        if overall_conversion < 1:
            print("   📊 Overall conversion very low - consider end-to-end user journey audit")
        elif overall_conversion < 5:
            print("   📊 Overall conversion below average - focus on major bottlenecks")
        elif overall_conversion > 20:
            print("   🚀 Strong conversion rate - consider scaling successful elements")
        
        print("   🔍 Segment analysis by traffic source, device, or demographic")
        print("   📊 A/B test improvements at highest drop-off steps")
        print("   📈 Monitor funnel performance over time for trend analysis")
        print("   🎯 Set up alerts for significant drop-off rate changes")
    
    # Data quality recommendations
    if any('drop_off_rate' not in step for step in funnel_summary[1:]):
        print("   ⚠️ Some steps may have data quality issues - verify tracking implementation")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "step_1_views": {
                "required": False,
                "description": "First funnel step (e.g., landing page views, impressions)",
                "data_type": "numeric",
                "examples": ["landing_page_views", "visits", "sessions", "impressions"]
            },
            "step_2_engagement": {
                "required": False,
                "description": "Second funnel step (e.g., product views, clicks)",
                "data_type": "numeric",
                "examples": ["product_views", "page_views", "clicks", "engagement"]
            },
            "step_3_intent": {
                "required": False,
                "description": "Third funnel step (e.g., add to cart, signups)",
                "data_type": "numeric",
                "examples": ["add_to_cart", "signups", "leads", "downloads"]
            },
            "step_4_conversion": {
                "required": False,
                "description": "Final conversion step (e.g., purchases, completions)",
                "data_type": "numeric",
                "examples": ["purchases", "conversions", "sales", "completions"]
            },
            "user_id": {
                "required": False,
                "description": "User or session identifier for tracking",
                "data_type": "categorical",
                "examples": ["user_id", "session_id", "customer_id"]
            },
            "timestamp": {
                "required": False,
                "description": "Date/time for temporal analysis",
                "data_type": "datetime",
                "examples": ["date", "timestamp", "created_at"]
            },
            "campaign": {
                "required": False,
                "description": "Campaign or traffic source for segmentation",
                "data_type": "categorical",
                "examples": ["campaign", "source", "utm_source", "channel"]
            }
        }
"""Campaign Metrics Analysis Component.

Focused analysis component for marketing campaign performance metrics
including impressions, clicks, conversions, CTR, and other campaign KPIs.
"""
from typing import Dict, Any


class CampaignMetricsAnalysis:
    """Focused component for marketing campaign metrics analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "campaign_metrics_analysis",
            "display_name": "Campaign Metrics Analysis",
            "description": "Analyze marketing campaign performance metrics: impressions, clicks, conversions, CTR, CPC, ROAS",
            "category": "marketing",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-20 seconds",
            "icon": "📊",
            "tags": ["marketing", "campaigns", "metrics", "performance", "ctr", "conversion"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has marketing campaign-related columns - always returns True for flexibility"""
        # Always return True to allow users to configure column mapping via modal
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate campaign metrics analysis code with column selection support"""
        
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
COLUMN_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect marketing columns (user can override these)
COLUMN_MAPPING = {}
possible_mappings = {
    'impressions': ['impression', 'impressions', 'views', 'reach'],
    'clicks': ['click', 'clicks', 'click_count'],
    'conversions': ['conversion', 'conversions', 'conv', 'goal_completion'],
    'spend': ['spend', 'cost', 'budget', 'amount', 'investment'],
    'revenue': ['revenue', 'sales', 'value', 'income'],
    'campaign': ['campaign', 'campaign_name', 'ad_name', 'creative'],
    'channel': ['channel', 'source', 'medium', 'platform'],
    'date': ['date', 'day', 'timestamp', 'created_at']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in COLUMN_MAPPING:
                COLUMN_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== CAMPAIGN METRICS ANALYSIS ===")
print("Marketing campaign performance analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 COLUMN MAPPING:")
for metric, col in COLUMN_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.title()}}: {{col}}")
print()

# Get numeric columns for campaign metrics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   Campaign metrics analysis requires numeric data.")
else:
    print(f"📊 ANALYZING {{len(numeric_cols)}} NUMERIC COLUMNS")
    print(f"   Available columns: {{numeric_cols}}")
    print()
    
    # Core campaign metrics analysis
    campaign_metrics = {{}}
    
    # 1. Basic Performance Metrics
    print("🎯 CORE CAMPAIGN METRICS")
    
    if 'impressions' in COLUMN_MAPPING and COLUMN_MAPPING['impressions'] in df.columns:
        impressions_col = COLUMN_MAPPING['impressions']
        total_impressions = df[impressions_col].sum()
        avg_impressions = df[impressions_col].mean()
        campaign_metrics['total_impressions'] = total_impressions
        print(f"   👁️  Total Impressions: {{total_impressions:,.0f}}")
        print(f"   📈 Average Impressions per record: {{avg_impressions:,.0f}}")
    
    if 'clicks' in COLUMN_MAPPING and COLUMN_MAPPING['clicks'] in df.columns:
        clicks_col = COLUMN_MAPPING['clicks']
        total_clicks = df[clicks_col].sum()
        avg_clicks = df[clicks_col].mean()
        campaign_metrics['total_clicks'] = total_clicks
        print(f"   👆 Total Clicks: {{total_clicks:,.0f}}")
        print(f"   📈 Average Clicks per record: {{avg_clicks:,.0f}}")
    
    if 'conversions' in COLUMN_MAPPING and COLUMN_MAPPING['conversions'] in df.columns:
        conversions_col = COLUMN_MAPPING['conversions']
        total_conversions = df[conversions_col].sum()
        avg_conversions = df[conversions_col].mean()
        campaign_metrics['total_conversions'] = total_conversions
        print(f"   🎯 Total Conversions: {{total_conversions:,.0f}}")
        print(f"   📈 Average Conversions per record: {{avg_conversions:,.0f}}")
    
    if 'spend' in COLUMN_MAPPING and COLUMN_MAPPING['spend'] in df.columns:
        spend_col = COLUMN_MAPPING['spend']
        total_spend = df[spend_col].sum()
        avg_spend = df[spend_col].mean()
        campaign_metrics['total_spend'] = total_spend
        print(f"   💰 Total Spend: ${{total_spend:,.2f}}")
        print(f"   📈 Average Spend per record: ${{avg_spend:,.2f}}")
    
    if 'revenue' in COLUMN_MAPPING and COLUMN_MAPPING['revenue'] in df.columns:
        revenue_col = COLUMN_MAPPING['revenue']
        total_revenue = df[revenue_col].sum()
        avg_revenue = df[revenue_col].mean()
        campaign_metrics['total_revenue'] = total_revenue
        print(f"   💵 Total Revenue: ${{total_revenue:,.2f}}")
        print(f"   📈 Average Revenue per record: ${{avg_revenue:,.2f}}")
    
    print()
    
    # 2. Calculated KPIs
    print("🧮 CALCULATED KPIs")
    
    # Click-Through Rate (CTR)
    if 'impressions' in COLUMN_MAPPING and 'clicks' in COLUMN_MAPPING:
        impressions_col = COLUMN_MAPPING['impressions']
        clicks_col = COLUMN_MAPPING['clicks']
        
        # Calculate CTR for each record where impressions > 0
        df_with_impressions = df[df[impressions_col] > 0].copy()
        if not df_with_impressions.empty:
            df_with_impressions['ctr'] = (df_with_impressions[clicks_col] / df_with_impressions[impressions_col]) * 100
            avg_ctr = df_with_impressions['ctr'].mean()
            median_ctr = df_with_impressions['ctr'].median()
            campaign_metrics['avg_ctr'] = avg_ctr
            
            print(f"   📊 Average CTR: {{avg_ctr:.2f}}%")
            print(f"   📊 Median CTR: {{median_ctr:.2f}}%")
            
            # CTR performance categories
            high_ctr = (df_with_impressions['ctr'] > avg_ctr * 1.5).sum()
            low_ctr = (df_with_impressions['ctr'] < avg_ctr * 0.5).sum()
            print(f"   🔥 High-performing CTR (>{{avg_ctr*1.5:.1f}}%): {{high_ctr}} records")
            print(f"   ❄️  Low-performing CTR (<{{avg_ctr*0.5:.1f}}%): {{low_ctr}} records")
    
    # Conversion Rate
    if 'clicks' in COLUMN_MAPPING and 'conversions' in COLUMN_MAPPING:
        clicks_col = COLUMN_MAPPING['clicks']
        conversions_col = COLUMN_MAPPING['conversions']
        
        df_with_clicks = df[df[clicks_col] > 0].copy()
        if not df_with_clicks.empty:
            df_with_clicks['conversion_rate'] = (df_with_clicks[conversions_col] / df_with_clicks[clicks_col]) * 100
            avg_conv_rate = df_with_clicks['conversion_rate'].mean()
            median_conv_rate = df_with_clicks['conversion_rate'].median()
            campaign_metrics['avg_conversion_rate'] = avg_conv_rate
            
            print(f"   🎯 Average Conversion Rate: {{avg_conv_rate:.2f}}%")
            print(f"   🎯 Median Conversion Rate: {{median_conv_rate:.2f}}%")
    
    # Cost Per Click (CPC)
    if 'spend' in COLUMN_MAPPING and 'clicks' in COLUMN_MAPPING:
        spend_col = COLUMN_MAPPING['spend']
        clicks_col = COLUMN_MAPPING['clicks']
        
        df_with_clicks = df[df[clicks_col] > 0].copy()
        if not df_with_clicks.empty:
            df_with_clicks['cpc'] = df_with_clicks[spend_col] / df_with_clicks[clicks_col]
            avg_cpc = df_with_clicks['cpc'].mean()
            median_cpc = df_with_clicks['cpc'].median()
            campaign_metrics['avg_cpc'] = avg_cpc
            
            print(f"   💰 Average CPC: ${{avg_cpc:.2f}}")
            print(f"   💰 Median CPC: ${{median_cpc:.2f}}")
    
    # Cost Per Acquisition (CPA)
    if 'spend' in COLUMN_MAPPING and 'conversions' in COLUMN_MAPPING:
        spend_col = COLUMN_MAPPING['spend']
        conversions_col = COLUMN_MAPPING['conversions']
        
        df_with_conversions = df[df[conversions_col] > 0].copy()
        if not df_with_conversions.empty:
            df_with_conversions['cpa'] = df_with_conversions[spend_col] / df_with_conversions[conversions_col]
            avg_cpa = df_with_conversions['cpa'].mean()
            median_cpa = df_with_conversions['cpa'].median()
            campaign_metrics['avg_cpa'] = avg_cpa
            
            print(f"   🎯 Average CPA: ${{avg_cpa:.2f}}")
            print(f"   🎯 Median CPA: ${{median_cpa:.2f}}")
    
    # Return on Ad Spend (ROAS)
    if 'revenue' in COLUMN_MAPPING and 'spend' in COLUMN_MAPPING:
        revenue_col = COLUMN_MAPPING['revenue']
        spend_col = COLUMN_MAPPING['spend']
        
        df_with_spend = df[df[spend_col] > 0].copy()
        if not df_with_spend.empty:
            df_with_spend['roas'] = df_with_spend[revenue_col] / df_with_spend[spend_col]
            avg_roas = df_with_spend['roas'].mean()
            median_roas = df_with_spend['roas'].median()
            campaign_metrics['avg_roas'] = avg_roas
            
            print(f"   📊 Average ROAS: {{avg_roas:.2f}}x")
            print(f"   📊 Median ROAS: {{median_roas:.2f}}x")
            
            # ROAS performance categories
            profitable = (df_with_spend['roas'] > 1).sum()
            highly_profitable = (df_with_spend['roas'] > 3).sum()
            print(f"   ✅ Profitable campaigns (ROAS > 1): {{profitable}}/{{len(df_with_spend)}} ({{profitable/len(df_with_spend)*100:.1f}}%)")
            print(f"   🚀 Highly profitable (ROAS > 3): {{highly_profitable}}/{{len(df_with_spend)}} ({{highly_profitable/len(df_with_spend)*100:.1f}}%)")
    
    print()
    
    # 3. Performance Distribution Analysis
    print("📈 PERFORMANCE DISTRIBUTION ANALYSIS")
    
    # Find outliers in key metrics
    outlier_analysis = {{}}
    key_metrics = []
    
    for metric_name, col_name in COLUMN_MAPPING.items():
        if col_name in df.columns and df[col_name].dtype in ['int64', 'float64']:
            key_metrics.append((metric_name, col_name))
    
    for metric_name, col_name in key_metrics:
        col_data = df[col_name].dropna()
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_percentage = len(outliers) / len(col_data) * 100
            
            outlier_analysis[metric_name] = {{
                'count': len(outliers),
                'percentage': outlier_percentage,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            }}
            
            print(f"   📊 {{metric_name.title()}} outliers: {{len(outliers)}} ({{outlier_percentage:.1f}}%)")
            if len(outliers) > 0:
                print(f"      Range: {{col_data.min():.2f}} to {{col_data.max():.2f}}")
                print(f"      Normal range: {{lower_bound:.2f}} to {{upper_bound:.2f}}")
    
    print()
    
    # 4. Channel/Campaign Performance (if applicable)
    if 'campaign' in COLUMN_MAPPING and COLUMN_MAPPING['campaign'] in df.columns:
        campaign_col = COLUMN_MAPPING['campaign']
        print("🏆 TOP PERFORMING CAMPAIGNS")
        
        # Group by campaign
        campaign_performance = df.groupby(campaign_col).agg({{
            col: 'sum' for col in numeric_cols if col in df.columns
        }}).round(2)
        
        print("   📊 Campaign Summary (Top 10):")
        print(campaign_performance.head(10).to_string())
        
        # Best performing campaign by different metrics
        if 'clicks' in COLUMN_MAPPING and COLUMN_MAPPING['clicks'] in campaign_performance.columns:
            clicks_col = COLUMN_MAPPING['clicks']
            top_clicks_campaign = campaign_performance[clicks_col].idxmax()
            print(f"\\n   👆 Most clicks: {{top_clicks_campaign}} ({{campaign_performance.loc[top_clicks_campaign, clicks_col]:,.0f}} clicks)")
        
        if 'conversions' in COLUMN_MAPPING and COLUMN_MAPPING['conversions'] in campaign_performance.columns:
            conv_col = COLUMN_MAPPING['conversions']
            top_conv_campaign = campaign_performance[conv_col].idxmax()
            print(f"   🎯 Most conversions: {{top_conv_campaign}} ({{campaign_performance.loc[top_conv_campaign, conv_col]:,.0f}} conversions)")
    
    if 'channel' in COLUMN_MAPPING and COLUMN_MAPPING['channel'] in df.columns:
        channel_col = COLUMN_MAPPING['channel']
        print("\\n📺 CHANNEL PERFORMANCE")
        
        channel_performance = df.groupby(channel_col).agg({{
            col: ['sum', 'mean'] for col in numeric_cols if col in df.columns
        }}).round(2)
        
        print("   📊 Top channels by total performance:")
        # Flatten column names for better display
        channel_performance.columns = ['_'.join(col).strip() for col in channel_performance.columns]
        print(channel_performance.head(10).to_string())
    
    print()
    
    # 5. Data Quality Issues for Marketing Data
    print("✅ DATA QUALITY ASSESSMENT")
    quality_issues = []
    
    # Check for common marketing data issues
    for metric_name, col_name in COLUMN_MAPPING.items():
        if col_name in df.columns:
            col_data = df[col_name]
            
            # Missing values
            missing_pct = col_data.isnull().sum() / len(df) * 100
            if missing_pct > 10:
                quality_issues.append(f"{{metric_name.title()}}: High missing values ({{missing_pct:.1f}}%)")
            
            # Negative values where they shouldn't be
            if col_data.dtype in ['int64', 'float64']:
                if metric_name in ['impressions', 'clicks', 'conversions', 'spend'] and (col_data < 0).sum() > 0:
                    negative_count = (col_data < 0).sum()
                    quality_issues.append(f"{{metric_name.title()}}: {{negative_count}} negative values (should be positive)")
                
                # Zero values in key metrics
                if metric_name in ['impressions', 'spend'] and (col_data == 0).sum() > len(df) * 0.1:
                    zero_count = (col_data == 0).sum()
                    quality_issues.append(f"{{metric_name.title()}}: Many zero values ({{zero_count}}, {{zero_count/len(df)*100:.1f}}%)")
    
    # Logical inconsistencies
    if ('impressions' in COLUMN_MAPPING and 'clicks' in COLUMN_MAPPING and 
        COLUMN_MAPPING['impressions'] in df.columns and COLUMN_MAPPING['clicks'] in df.columns):
        
        impressions_col = COLUMN_MAPPING['impressions']
        clicks_col = COLUMN_MAPPING['clicks']
        
        # Clicks > Impressions (impossible)
        impossible_records = df[df[clicks_col] > df[impressions_col]]
        if len(impossible_records) > 0:
            quality_issues.append(f"Logical error: {{len(impossible_records)}} records have more clicks than impressions")
    
    if quality_issues:
        print("   ⚠️ Quality issues found:")
        for issue in quality_issues:
            print(f"      • {{issue}}")
    else:
        print("   ✅ No major data quality issues detected")
    
    print()
    
    # 6. Actionable Recommendations
    print("💡 MARKETING RECOMMENDATIONS")
    
    # Performance-based recommendations
    if 'avg_ctr' in campaign_metrics:
        if campaign_metrics['avg_ctr'] < 1.0:  # Industry benchmark varies
            print("   📊 CTR appears low - consider improving ad creatives or targeting")
        elif campaign_metrics['avg_ctr'] > 3.0:
            print("   🔥 Excellent CTR performance - scale successful campaigns")
    
    if 'avg_conversion_rate' in campaign_metrics:
        if campaign_metrics['avg_conversion_rate'] < 2.0:
            print("   🎯 Conversion rate could be improved - optimize landing pages")
        elif campaign_metrics['avg_conversion_rate'] > 5.0:
            print("   🚀 Strong conversion rate - consider increasing budget")
    
    if 'avg_roas' in campaign_metrics:
        if campaign_metrics['avg_roas'] < 1.0:
            print("   💰 ROAS below break-even - review targeting and bidding strategy")
        elif campaign_metrics['avg_roas'] > 4.0:
            print("   💎 Excellent ROAS - consider expanding to similar audiences")
    
    # Data quality recommendations
    print("   📊 Always segment analysis by time periods and channels")
    print("   🔍 Investigate outliers before making optimization decisions")
    print("   📈 Track trends over time, not just point-in-time metrics")
    print("   🎯 Focus on metrics that align with business objectives")
    
    if outlier_analysis:
        high_outlier_metrics = [k for k, v in outlier_analysis.items() if v['percentage'] > 5]
        if high_outlier_metrics:
            print(f"   ⚠️ High outlier metrics ({{', '.join(high_outlier_metrics)}}) - investigate data collection")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "impressions": {
                "required": False,
                "description": "Total impressions/views for the campaign",
                "data_type": "numeric",
                "examples": ["impressions", "views", "reach"]
            },
            "clicks": {
                "required": False,
                "description": "Total clicks received",
                "data_type": "numeric", 
                "examples": ["clicks", "click_count", "link_clicks"]
            },
            "conversions": {
                "required": False,
                "description": "Total conversions/goal completions",
                "data_type": "numeric",
                "examples": ["conversions", "purchases", "signups"]
            },
            "spend": {
                "required": False,
                "description": "Amount spent on the campaign",
                "data_type": "numeric",
                "examples": ["spend", "cost", "budget", "investment"]
            },
            "revenue": {
                "required": False,
                "description": "Revenue generated from the campaign",
                "data_type": "numeric",
                "examples": ["revenue", "sales", "value"]
            },
            "campaign": {
                "required": False,
                "description": "Campaign name or identifier",
                "data_type": "categorical",
                "examples": ["campaign_name", "ad_name", "creative_name"]
            },
            "channel": {
                "required": False,
                "description": "Marketing channel or platform",
                "data_type": "categorical",
                "examples": ["channel", "source", "platform", "medium"]
            }
        }
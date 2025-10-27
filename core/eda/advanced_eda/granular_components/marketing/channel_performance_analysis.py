"""Channel Performance Analysis Component.

Focused analysis component for marketing channel performance
comparing different marketing channels and their effectiveness.
"""
from typing import Dict, Any


class ChannelPerformanceAnalysis:
    """Focused component for marketing channel performance analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "channel_performance_analysis",
            "display_name": "Channel Performance Analysis",
            "description": "Compare marketing channel performance: ROI, conversion rates, cost efficiency across channels",
            "category": "marketing",
            "complexity": "intermediate", 
            "required_data_types": ["categorical", "numeric"],
            "estimated_runtime": "10-20 seconds",
            "icon": "📺",
            "tags": ["marketing", "channels", "performance", "comparison", "roi", "efficiency"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has channel-related columns - always returns True for flexibility"""
        # Always return True to allow users to configure column mapping via modal
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate channel performance analysis code with column selection support"""
        
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
CHANNEL_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect channel columns (user can override these)
CHANNEL_MAPPING = {}
possible_mappings = {
    'channel': ['channel', 'source', 'traffic_source', 'utm_source', 'medium', 'referrer'],
    'spend': ['spend', 'cost', 'budget', 'investment', 'ad_spend'],
    'impressions': ['impressions', 'views', 'reach', 'exposure'],
    'clicks': ['clicks', 'visits', 'sessions', 'traffic'],
    'conversions': ['conversions', 'sales', 'purchases', 'goals', 'leads'],
    'revenue': ['revenue', 'value', 'income', 'sales_amount'],
    'campaign': ['campaign', 'campaign_name', 'ad_name', 'creative'],
    'date': ['date', 'timestamp', 'day', 'created_at'],
    'device': ['device', 'device_type', 'platform', 'device_category'],
    'audience': ['audience', 'segment', 'demographic', 'target']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in CHANNEL_MAPPING:
                CHANNEL_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== CHANNEL PERFORMANCE ANALYSIS ===")
print("Marketing channel effectiveness comparison")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 CHANNEL MAPPING:")
for metric, col in CHANNEL_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.replace('_', ' ').title()}}: {{col}}")
print()

# Check if we have the required channel column
if 'channel' not in CHANNEL_MAPPING or CHANNEL_MAPPING['channel'] not in df.columns:
    print("❌ NO CHANNEL COLUMN FOUND")
    print("   Channel performance analysis requires a channel/source column.")
    print("   Available categorical columns:", df.select_dtypes(include=['object', 'category']).columns.tolist())
else:
    channel_col = CHANNEL_MAPPING['channel']
    print(f"📊 ANALYZING CHANNEL PERFORMANCE")
    print(f"   Channel column: {{channel_col}}")
    print()
    
    # Get available channels
    channels = df[channel_col].value_counts()
    print(f"   📺 Found {{len(channels)}} unique channels:")
    for i, (channel, count) in enumerate(channels.head(10).items()):
        print(f"      {{i+1}}. {{channel}}: {{count:,}} records ({{count/len(df)*100:.1f}}%)")
    
    if len(channels) > 10:
        print(f"      ... and {{len(channels) - 10}} more channels")
    print()
    
    # 1. Channel Volume Analysis
    print("📊 CHANNEL VOLUME ANALYSIS")
    
    channel_performance = {{}}
    
    for channel in channels.head(15).index:  # Analyze top 15 channels
        channel_data = df[df[channel_col] == channel]
        
        performance_metrics = {{
            'channel': channel,
            'records': len(channel_data),
            'percentage_of_total': len(channel_data) / len(df) * 100
        }}
        
        # Calculate metrics for each mapped column
        for metric, col_name in CHANNEL_MAPPING.items():
            if col_name in df.columns and df[col_name].dtype in ['int64', 'float64']:
                metric_data = channel_data[col_name].dropna()
                if not metric_data.empty:
                    performance_metrics[f'{{metric}}_total'] = metric_data.sum()
                    performance_metrics[f'{{metric}}_avg'] = metric_data.mean()
                    performance_metrics[f'{{metric}}_median'] = metric_data.median()
        
        channel_performance[channel] = performance_metrics
    
    # Display volume metrics
    print("   📈 Channel traffic and volume:")
    volume_data = []
    for channel, metrics in channel_performance.items():
        volume_data.append({{
            'Channel': channel,
            'Records': metrics['records'],
            'Percentage': f"{{metrics['percentage_of_total']:.1f}}%"
        }})
    
    volume_df = pd.DataFrame(volume_data)
    print(volume_df.to_string(index=False))
    print()
    
    # 2. Channel Efficiency Analysis
    print("⚡ CHANNEL EFFICIENCY ANALYSIS")
    
    efficiency_metrics = []
    
    for channel, metrics in channel_performance.items():
        efficiency = {{'Channel': channel}}
        
        # Click-Through Rate (if impressions and clicks available)
        if 'impressions_total' in metrics and 'clicks_total' in metrics:
            if metrics['impressions_total'] > 0:
                ctr = (metrics['clicks_total'] / metrics['impressions_total']) * 100
                efficiency['CTR'] = f"{{ctr:.2f}}%"
                metrics['ctr'] = ctr
        
        # Conversion Rate (if clicks and conversions available)
        if 'clicks_total' in metrics and 'conversions_total' in metrics:
            if metrics['clicks_total'] > 0:
                conv_rate = (metrics['conversions_total'] / metrics['clicks_total']) * 100
                efficiency['Conv_Rate'] = f"{{conv_rate:.2f}}%"
                metrics['conversion_rate'] = conv_rate
        
        # Cost Per Click (if spend and clicks available)
        if 'spend_total' in metrics and 'clicks_total' in metrics:
            if metrics['clicks_total'] > 0:
                cpc = metrics['spend_total'] / metrics['clicks_total']
                efficiency['CPC'] = f"${{cpc:.2f}}"
                metrics['cpc'] = cpc
        
        # Cost Per Acquisition (if spend and conversions available)
        if 'spend_total' in metrics and 'conversions_total' in metrics:
            if metrics['conversions_total'] > 0:
                cpa = metrics['spend_total'] / metrics['conversions_total']
                efficiency['CPA'] = f"${{cpa:.2f}}"
                metrics['cpa'] = cpa
        
        # Return on Ad Spend (if revenue and spend available)
        if 'revenue_total' in metrics and 'spend_total' in metrics:
            if metrics['spend_total'] > 0:
                roas = metrics['revenue_total'] / metrics['spend_total']
                efficiency['ROAS'] = f"{{roas:.2f}}x"
                metrics['roas'] = roas
        
        if len(efficiency) > 1:  # Has metrics beyond just channel name
            efficiency_metrics.append(efficiency)
    
    if efficiency_metrics:
        efficiency_df = pd.DataFrame(efficiency_metrics)
        print("   📊 Channel efficiency comparison:")
        print(efficiency_df.to_string(index=False))
        print()
        
        # Identify best and worst performing channels
        if 'ROAS' in efficiency_df.columns:
            # Find channels with ROAS data
            roas_data = []
            for channel, metrics in channel_performance.items():
                if 'roas' in metrics:
                    roas_data.append((channel, metrics['roas']))
            
            if roas_data:
                roas_data.sort(key=lambda x: x[1], reverse=True)
                best_roas_channel = roas_data[0]
                worst_roas_channel = roas_data[-1]
                
                print(f"   🏆 Best ROAS: {{best_roas_channel[0]}} ({{best_roas_channel[1]:.2f}}x)")
                print(f"   📉 Lowest ROAS: {{worst_roas_channel[0]}} ({{worst_roas_channel[1]:.2f}}x)")
        
        if 'CPA' in efficiency_df.columns:
            # Find channels with CPA data
            cpa_data = []
            for channel, metrics in channel_performance.items():
                if 'cpa' in metrics:
                    cpa_data.append((channel, metrics['cpa']))
            
            if cpa_data:
                cpa_data.sort(key=lambda x: x[1])  # Lower CPA is better
                best_cpa_channel = cpa_data[0]
                worst_cpa_channel = cpa_data[-1]
                
                print(f"   💰 Lowest CPA: {{best_cpa_channel[0]}} (${{best_cpa_channel[1]:.2f}})")
                print(f"   💸 Highest CPA: {{worst_cpa_channel[0]}} (${{worst_cpa_channel[1]:.2f}})")
        
        print()
    else:
        print("   ⚠️ Insufficient data for efficiency calculations")
        print("   Ensure you have spend, clicks, conversions, or revenue data")
        print()
    
    # 3. Channel Profitability Analysis
    print("💰 CHANNEL PROFITABILITY ANALYSIS")
    
    if 'revenue' in CHANNEL_MAPPING and 'spend' in CHANNEL_MAPPING:
        revenue_col = CHANNEL_MAPPING['revenue']
        spend_col = CHANNEL_MAPPING['spend']
        
        profitability_data = []
        
        for channel in channels.head(10).index:
            channel_data = df[df[channel_col] == channel]
            
            total_revenue = channel_data[revenue_col].sum()
            total_spend = channel_data[spend_col].sum()
            profit = total_revenue - total_spend
            profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
            roi_percentage = (profit / total_spend) * 100 if total_spend > 0 else 0
            
            profitability_data.append({{
                'Channel': channel,
                'Revenue': f"${{total_revenue:,.0f}}",
                'Spend': f"${{total_spend:,.0f}}",
                'Profit': f"${{profit:,.0f}}",
                'Margin': f"{{profit_margin:.1f}}%",
                'ROI': f"{{roi_percentage:.1f}}%"
            }})
        
        profit_df = pd.DataFrame(profitability_data)
        print("   💵 Channel profitability summary:")
        print(profit_df.to_string(index=False))
        
        # Calculate total profitability
        total_revenue_all = df[revenue_col].sum()
        total_spend_all = df[spend_col].sum()
        total_profit = total_revenue_all - total_spend_all
        
        print(f"\\n   📊 Overall Performance:")
        print(f"      Total Revenue: ${{total_revenue_all:,.0f}}")
        print(f"      Total Spend: ${{total_spend_all:,.0f}}")
        print(f"      Total Profit: ${{total_profit:,.0f}}")
        print(f"      Overall ROI: {{(total_profit/total_spend_all)*100:.1f}}%" if total_spend_all > 0 else "")
        
        print()
    else:
        print("   ⚠️ Revenue and spend data required for profitability analysis")
        print()
    
    # 4. Channel Quality Analysis
    print("🎯 CHANNEL QUALITY ANALYSIS")
    
    quality_scores = {{}}
    
    for channel, metrics in channel_performance.items():
        quality_score = 0
        quality_factors = 0
        
        # Volume quality (scale consideration)
        volume_pct = metrics['percentage_of_total']
        if volume_pct > 10:
            quality_score += 3  # High volume
        elif volume_pct > 5:
            quality_score += 2  # Medium volume
        elif volume_pct > 1:
            quality_score += 1  # Low but meaningful volume
        quality_factors += 3
        
        # Efficiency quality
        if 'ctr' in metrics:
            ctr = metrics['ctr']
            if ctr > 3:
                quality_score += 3
            elif ctr > 1:
                quality_score += 2
            elif ctr > 0.5:
                quality_score += 1
            quality_factors += 3
        
        if 'conversion_rate' in metrics:
            conv_rate = metrics['conversion_rate']
            if conv_rate > 5:
                quality_score += 3
            elif conv_rate > 2:
                quality_score += 2
            elif conv_rate > 1:
                quality_score += 1
            quality_factors += 3
        
        if 'roas' in metrics:
            roas = metrics['roas']
            if roas > 4:
                quality_score += 3
            elif roas > 2:
                quality_score += 2
            elif roas > 1:
                quality_score += 1
            quality_factors += 3
        
        if quality_factors > 0:
            normalized_score = (quality_score / quality_factors) * 100
            quality_scores[channel] = normalized_score
    
    if quality_scores:
        print("   🏆 Channel quality scores (0-100):")
        sorted_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (channel, score) in enumerate(sorted_quality[:10]):
            if score >= 80:
                emoji = "🟢"
                status = "Excellent"
            elif score >= 60:
                emoji = "🟡" 
                status = "Good"
            elif score >= 40:
                emoji = "🟠"
                status = "Fair"
            else:
                emoji = "🔴"
                status = "Poor"
            
            print(f"      {{i+1}}. {{emoji}} {{channel}}: {{score:.0f}} ({{status}})")
        
        print()
    
    # 5. Channel Mix Analysis
    print("📊 CHANNEL MIX ANALYSIS")
    
    # Calculate channel concentration
    top_3_share = sum(channels.head(3).values) / len(df) * 100
    top_5_share = sum(channels.head(5).values) / len(df) * 100
    
    print(f"   📈 Channel concentration:")
    print(f"      Top 3 channels: {{top_3_share:.1f}}% of total traffic")
    print(f"      Top 5 channels: {{top_5_share:.1f}}% of total traffic")
    
    # Diversification assessment
    if top_3_share > 80:
        diversification = "🔴 Highly concentrated (risky)"
    elif top_3_share > 60:
        diversification = "🟠 Moderately concentrated"
    elif top_3_share > 40:
        diversification = "🟡 Balanced mix"
    else:
        diversification = "🟢 Well diversified"
    
    print(f"      Diversification: {{diversification}}")
    
    # Channel lifecycle analysis
    print("\\n   📅 Channel maturity analysis:")
    for channel in channels.head(5).index:
        channel_records = channels[channel]
        if channel_records > len(df) * 0.1:
            maturity = "🌳 Mature"
        elif channel_records > len(df) * 0.05:
            maturity = "🌱 Growing"
        else:
            maturity = "🌰 Emerging"
        
        print(f"      {{channel}}: {{maturity}} ({{channel_records:,}} records)")
    
    print()
    
    # 6. Device/Audience Breakdown by Channel (if available)
    if 'device' in CHANNEL_MAPPING and CHANNEL_MAPPING['device'] in df.columns:
        device_col = CHANNEL_MAPPING['device']
        print("📱 CHANNEL PERFORMANCE BY DEVICE")
        
        # Create cross-tabulation
        channel_device = pd.crosstab(df[channel_col], df[device_col], normalize='index') * 100
        
        print("   📊 Device distribution by channel (%):")
        print(channel_device.round(1).head(8).to_string())
        print()
    
    # 7. Recommendations
    print("💡 CHANNEL OPTIMIZATION RECOMMENDATIONS")
    
    # Budget allocation recommendations
    if quality_scores:
        best_channels = [ch for ch, score in sorted_quality[:3]]
        worst_channels = [ch for ch, score in sorted_quality[-2:]]
        
        print("   💰 Budget allocation recommendations:")
        print(f"      ✅ Increase investment: {{', '.join(best_channels)}}")
        print(f"      ❌ Consider reducing/optimizing: {{', '.join(worst_channels)}}")
    
    # Diversification recommendations
    if top_3_share > 70:
        print("\\n   🎯 Diversification recommendations:")
        print("      • Reduce dependency on top-performing channels")
        print("      • Test and develop emerging channels")
        print("      • Create contingency plans for channel disruptions")
    
    # Performance optimization
    performance_issues = []
    for channel, metrics in channel_performance.items():
        if 'ctr' in metrics and metrics['ctr'] < 1.0:
            performance_issues.append(f"{{channel}}: Low CTR")
        if 'conversion_rate' in metrics and metrics['conversion_rate'] < 2.0:
            performance_issues.append(f"{{channel}}: Low conversion rate")
        if 'roas' in metrics and metrics['roas'] < 1.0:
            performance_issues.append(f"{{channel}}: Negative ROAS")
    
    if performance_issues:
        print("\\n   🔧 Performance optimization priorities:")
        for issue in performance_issues[:5]:
            print(f"      • {{issue}}")
    
    # General recommendations
    print("\\n   📊 General channel management:")
    print("      • Monitor channel performance trends over time")
    print("      • A/B test creative and messaging by channel")
    print("      • Implement cross-channel attribution modeling")
    print("      • Regular competitive analysis for each channel")
    print("      • Set up alerts for significant performance changes")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "channel": {
                "required": True,
                "description": "Marketing channel or traffic source",
                "data_type": "categorical",
                "examples": ["channel", "source", "utm_source", "medium"]
            },
            "spend": {
                "required": False,
                "description": "Amount spent per channel",
                "data_type": "numeric",
                "examples": ["spend", "cost", "budget", "investment"]
            },
            "impressions": {
                "required": False,
                "description": "Impressions or reach per channel",
                "data_type": "numeric",
                "examples": ["impressions", "views", "reach", "exposure"]
            },
            "clicks": {
                "required": False,
                "description": "Clicks or visits per channel",
                "data_type": "numeric",
                "examples": ["clicks", "visits", "sessions", "traffic"]
            },
            "conversions": {
                "required": False,
                "description": "Conversions or sales per channel",
                "data_type": "numeric",
                "examples": ["conversions", "sales", "purchases", "leads"]
            },
            "revenue": {
                "required": False,
                "description": "Revenue generated per channel",
                "data_type": "numeric",
                "examples": ["revenue", "value", "income", "sales_amount"]
            },
            "device": {
                "required": False,
                "description": "Device type for cross-analysis",
                "data_type": "categorical",
                "examples": ["device", "device_type", "platform"]
            }
        }
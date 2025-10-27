"""Engagement Analysis Component.

Focused analysis component for marketing engagement metrics
analyzing user interaction patterns and engagement quality.
"""
from typing import Dict, Any


class EngagementAnalysis:
    """Focused component for marketing engagement analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "engagement_analysis",
            "display_name": "Engagement Analysis", 
            "description": "Analyze engagement metrics: time spent, interactions, bounce rate, session depth",
            "category": "marketing",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "🎯",
            "tags": ["marketing", "engagement", "interaction", "behavior", "session", "bounce"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has engagement-related columns"""
        if not data_preview:
            return True
        
        columns = data_preview.get("columns", [])
        if not columns:
            return True
        
        # Look for engagement indicators
        engagement_keywords = [
            "time", "duration", "session", "bounce", "page", "interaction",
            "engagement", "depth", "scroll", "dwell", "visit", "return",
            "frequency", "recency", "activity", "behavior", "action"
        ]
        
        columns_lower = [col.lower() for col in columns]
        for keyword in engagement_keywords:
            for col in columns_lower:
                if keyword in col:
                    return True
        return False
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate engagement analysis code with column selection support"""
        
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
ENGAGEMENT_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect engagement columns (user can override these)
ENGAGEMENT_MAPPING = {}
possible_mappings = {
    'session_duration': ['session_duration', 'time_on_site', 'visit_duration', 'session_time'],
    'page_views': ['page_views', 'pageviews', 'pages_per_session', 'page_count'],
    'bounce_rate': ['bounce_rate', 'bounced', 'single_page_session', 'bounce'],
    'time_on_page': ['time_on_page', 'page_duration', 'dwell_time', 'avg_time_on_page'],
    'interactions': ['interactions', 'clicks', 'actions', 'events', 'engagements'],
    'scroll_depth': ['scroll_depth', 'scroll_percentage', 'page_scroll', 'scroll'],
    'return_visits': ['return_visits', 'returning_users', 'repeat_visits', 'frequency'],
    'session_depth': ['session_depth', 'pages_per_visit', 'page_depth'],
    'user_id': ['user_id', 'visitor_id', 'session_id', 'customer_id'],
    'device_type': ['device', 'device_type', 'platform', 'device_category'],
    'traffic_source': ['source', 'traffic_source', 'referrer', 'channel']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in ENGAGEMENT_MAPPING:
                ENGAGEMENT_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== ENGAGEMENT ANALYSIS ===")
print("User engagement and behavior analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 ENGAGEMENT COLUMN MAPPING:")
for metric, col in ENGAGEMENT_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.replace('_', ' ').title()}}: {{col}}")
print()

# Get numeric columns for engagement metrics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    print("❌ NO NUMERIC COLUMNS FOUND")
    print("   Engagement analysis requires numeric data.")
else:
    print(f"📊 ANALYZING {{len(numeric_cols)}} NUMERIC COLUMNS")
    print()
    
    # 1. Session Quality Metrics
    print("⏱️ SESSION QUALITY METRICS")
    
    engagement_summary = {{}}
    
    # Session Duration Analysis
    if 'session_duration' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['session_duration'] in df.columns:
        duration_col = ENGAGEMENT_MAPPING['session_duration']
        duration_data = df[duration_col].dropna()
        
        if not duration_data.empty:
            avg_duration = duration_data.mean()
            median_duration = duration_data.median()
            total_time = duration_data.sum()
            
            # Convert to minutes if values seem to be in seconds
            time_unit = "seconds"
            display_avg = avg_duration
            display_median = median_duration
            display_total = total_time
            
            if avg_duration > 300:  # Likely seconds if > 5 minutes
                display_avg = avg_duration / 60
                display_median = median_duration / 60  
                display_total = total_time / 60
                time_unit = "minutes"
            
            engagement_summary['avg_session_duration'] = display_avg
            
            print(f"   ⏰ Average session duration: {{display_avg:.1f}} {{time_unit}}")
            print(f"   ⏰ Median session duration: {{display_median:.1f}} {{time_unit}}")
            print(f"   ⏰ Total time across all sessions: {{display_total:,.0f}} {{time_unit}}")
            
            # Session duration categories
            if time_unit == "minutes":
                short_sessions = (duration_data < 60).sum()  # < 1 minute
                medium_sessions = ((duration_data >= 60) & (duration_data < 300)).sum()  # 1-5 minutes
                long_sessions = (duration_data >= 300).sum()  # > 5 minutes
            else:
                short_sessions = (duration_data < 30).sum()  # < 30 seconds
                medium_sessions = ((duration_data >= 30) & (duration_data < 120)).sum()  # 30-120 seconds
                long_sessions = (duration_data >= 120).sum()  # > 2 minutes
            
            total_sessions = len(duration_data)
            print(f"   📊 Short sessions: {{short_sessions}} ({{short_sessions/total_sessions*100:.1f}}%)")
            print(f"   📊 Medium sessions: {{medium_sessions}} ({{medium_sessions/total_sessions*100:.1f}}%)")
            print(f"   📊 Long sessions: {{long_sessions}} ({{long_sessions/total_sessions*100:.1f}}%)")
    
    # Page Views per Session
    if 'page_views' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['page_views'] in df.columns:
        pv_col = ENGAGEMENT_MAPPING['page_views']
        pv_data = df[pv_col].dropna()
        
        if not pv_data.empty:
            avg_pageviews = pv_data.mean()
            median_pageviews = pv_data.median()
            total_pageviews = pv_data.sum()
            
            engagement_summary['avg_pageviews'] = avg_pageviews
            
            print(f"\\n   📄 Average pages per session: {{avg_pageviews:.1f}}")
            print(f"   📄 Median pages per session: {{median_pageviews:.1f}}")
            print(f"   📄 Total page views: {{total_pageviews:,.0f}}")
            
            # Page depth categories
            single_page = (pv_data == 1).sum()
            shallow_browsing = ((pv_data >= 2) & (pv_data <= 3)).sum()
            deep_browsing = (pv_data > 3).sum()
            
            print(f"   📊 Single page sessions: {{single_page}} ({{single_page/len(pv_data)*100:.1f}}%)")
            print(f"   📊 Shallow browsing (2-3 pages): {{shallow_browsing}} ({{shallow_browsing/len(pv_data)*100:.1f}}%)")
            print(f"   📊 Deep browsing (>3 pages): {{deep_browsing}} ({{deep_browsing/len(pv_data)*100:.1f}}%)")
    
    # Bounce Rate Analysis
    if 'bounce_rate' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['bounce_rate'] in df.columns:
        bounce_col = ENGAGEMENT_MAPPING['bounce_rate']
        bounce_data = df[bounce_col].dropna()
        
        if not bounce_data.empty:
            # Check if bounce rate is percentage (0-100) or binary (0-1)
            if bounce_data.max() <= 1.0:
                # Binary or decimal format
                avg_bounce = bounce_data.mean() * 100
                total_bounces = bounce_data.sum()
            else:
                # Percentage format
                avg_bounce = bounce_data.mean()
                total_bounces = (bounce_data > 0).sum()
            
            engagement_summary['bounce_rate'] = avg_bounce
            
            print(f"\\n   ⚡ Average bounce rate: {{avg_bounce:.1f}}%")
            print(f"   ⚡ Total bounced sessions: {{total_bounces:,.0f}}")
            
            # Bounce rate assessment
            if avg_bounce < 20:
                bounce_assessment = "🟢 Excellent"
            elif avg_bounce < 40:
                bounce_assessment = "🟡 Good"
            elif avg_bounce < 60:
                bounce_assessment = "🟠 Fair"
            else:
                bounce_assessment = "🔴 High - Needs Improvement"
                
            print(f"   📊 Bounce rate assessment: {{bounce_assessment}}")
    
    print()
    
    # 2. Interaction Depth Analysis
    print("🎯 INTERACTION DEPTH ANALYSIS")
    
    # Interactions per session
    if 'interactions' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['interactions'] in df.columns:
        interactions_col = ENGAGEMENT_MAPPING['interactions']
        interactions_data = df[interactions_col].dropna()
        
        if not interactions_data.empty:
            avg_interactions = interactions_data.mean()
            median_interactions = interactions_data.median()
            total_interactions = interactions_data.sum()
            
            engagement_summary['avg_interactions'] = avg_interactions
            
            print(f"   👆 Average interactions per session: {{avg_interactions:.1f}}")
            print(f"   👆 Median interactions per session: {{median_interactions:.1f}}")
            print(f"   👆 Total interactions: {{total_interactions:,.0f}}")
            
            # Interaction categories
            no_interaction = (interactions_data == 0).sum()
            low_interaction = ((interactions_data > 0) & (interactions_data <= 3)).sum()
            high_interaction = (interactions_data > 3).sum()
            
            print(f"   📊 No interaction sessions: {{no_interaction}} ({{no_interaction/len(interactions_data)*100:.1f}}%)")
            print(f"   📊 Low interaction (1-3): {{low_interaction}} ({{low_interaction/len(interactions_data)*100:.1f}}%)")
            print(f"   📊 High interaction (>3): {{high_interaction}} ({{high_interaction/len(interactions_data)*100:.1f}}%)")
    
    # Scroll depth analysis
    if 'scroll_depth' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['scroll_depth'] in df.columns:
        scroll_col = ENGAGEMENT_MAPPING['scroll_depth']
        scroll_data = df[scroll_col].dropna()
        
        if not scroll_data.empty:
            avg_scroll = scroll_data.mean()
            median_scroll = scroll_data.median()
            
            # Assume scroll depth is percentage if max > 1, otherwise convert
            if scroll_data.max() <= 1.0:
                avg_scroll *= 100
                median_scroll *= 100
                scroll_data_pct = scroll_data * 100
            else:
                scroll_data_pct = scroll_data
            
            engagement_summary['avg_scroll_depth'] = avg_scroll
            
            print(f"\\n   📜 Average scroll depth: {{avg_scroll:.1f}}%")
            print(f"   📜 Median scroll depth: {{median_scroll:.1f}}%")
            
            # Scroll categories
            shallow_scroll = (scroll_data_pct < 25).sum()
            medium_scroll = ((scroll_data_pct >= 25) & (scroll_data_pct < 75)).sum()
            deep_scroll = (scroll_data_pct >= 75).sum()
            
            print(f"   📊 Shallow scroll (<25%): {{shallow_scroll}} ({{shallow_scroll/len(scroll_data)*100:.1f}}%)")
            print(f"   📊 Medium scroll (25-75%): {{medium_scroll}} ({{medium_scroll/len(scroll_data)*100:.1f}}%)")
            print(f"   📊 Deep scroll (>75%): {{deep_scroll}} ({{deep_scroll/len(scroll_data)*100:.1f}}%)")
    
    print()
    
    # 3. User Loyalty & Return Behavior
    print("🔄 USER LOYALTY ANALYSIS")
    
    if 'return_visits' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['return_visits'] in df.columns:
        return_col = ENGAGEMENT_MAPPING['return_visits']
        return_data = df[return_col].dropna()
        
        if not return_data.empty:
            avg_return_visits = return_data.mean()
            total_return_visits = return_data.sum()
            returning_users = (return_data > 0).sum()
            
            engagement_summary['avg_return_visits'] = avg_return_visits
            engagement_summary['returning_users_pct'] = (returning_users / len(return_data)) * 100
            
            print(f"   🔄 Average return visits per user: {{avg_return_visits:.1f}}")
            print(f"   🔄 Total return visits: {{total_return_visits:,.0f}}")
            print(f"   🔄 Users with return visits: {{returning_users}} ({{returning_users/len(return_data)*100:.1f}}%)")
            
            # User loyalty segments
            new_users = (return_data == 0).sum()
            occasional_returners = ((return_data >= 1) & (return_data <= 3)).sum()
            regular_returners = ((return_data > 3) & (return_data <= 10)).sum()
            loyal_users = (return_data > 10).sum()
            
            print(f"   👶 New users (0 returns): {{new_users}} ({{new_users/len(return_data)*100:.1f}}%)")
            print(f"   🎯 Occasional returners (1-3): {{occasional_returners}} ({{occasional_returners/len(return_data)*100:.1f}}%)")
            print(f"   🎖️  Regular returners (4-10): {{regular_returners}} ({{regular_returners/len(return_data)*100:.1f}}%)")
            print(f"   👑 Loyal users (>10): {{loyal_users}} ({{loyal_users/len(return_data)*100:.1f}}%)")
    
    print()
    
    # 4. Segmentation Analysis
    segment_analyses = []
    
    # Device-based engagement
    if 'device_type' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['device_type'] in df.columns:
        device_col = ENGAGEMENT_MAPPING['device_type']
        print("📱 DEVICE-BASED ENGAGEMENT")
        
        device_engagement = {{}}
        for device in df[device_col].value_counts().head(5).index:
            device_data = df[df[device_col] == device]
            
            device_metrics = {{'device': device, 'sessions': len(device_data)}}
            
            # Calculate key metrics for each device
            for metric, col_name in ENGAGEMENT_MAPPING.items():
                if col_name in device_data.columns and device_data[col_name].dtype in ['int64', 'float64']:
                    device_metrics[metric] = device_data[col_name].mean()
            
            device_engagement[device] = device_metrics
        
        # Display device comparison
        print("   📊 Average engagement by device:")
        for device, metrics in device_engagement.items():
            print(f"\\n   📱 {{device}}:")
            print(f"      Sessions: {{metrics['sessions']:,}}")
            if 'session_duration' in metrics:
                print(f"      Avg session duration: {{metrics['session_duration']:.1f}}")
            if 'page_views' in metrics:
                print(f"      Avg page views: {{metrics['page_views']:.1f}}")
            if 'bounce_rate' in metrics:
                print(f"      Bounce rate: {{metrics['bounce_rate']:.1f}}%")
        
        segment_analyses.append('device')
    
    # Traffic source engagement
    if 'traffic_source' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['traffic_source'] in df.columns:
        source_col = ENGAGEMENT_MAPPING['traffic_source']
        print("\\n🌐 TRAFFIC SOURCE ENGAGEMENT")
        
        source_engagement = {{}}
        for source in df[source_col].value_counts().head(5).index:
            source_data = df[df[source_col] == source]
            
            source_metrics = {{'source': source, 'sessions': len(source_data)}}
            
            for metric, col_name in ENGAGEMENT_MAPPING.items():
                if col_name in source_data.columns and source_data[col_name].dtype in ['int64', 'float64']:
                    source_metrics[metric] = source_data[col_name].mean()
            
            source_engagement[source] = source_metrics
        
        # Display source comparison
        print("   📊 Average engagement by traffic source:")
        for source, metrics in source_engagement.items():
            print(f"\\n   🌐 {{source}}:")
            print(f"      Sessions: {{metrics['sessions']:,}}")
            if 'session_duration' in metrics:
                print(f"      Avg session duration: {{metrics['session_duration']:.1f}}")
            if 'page_views' in metrics:
                print(f"      Avg page views: {{metrics['page_views']:.1f}}")
            if 'bounce_rate' in metrics:
                print(f"      Bounce rate: {{metrics['bounce_rate']:.1f}}%")
        
        segment_analyses.append('traffic_source')
    
    if not segment_analyses:
        print("📊 GENERAL ENGAGEMENT SEGMENTS")
        
        # Create engagement score based on available metrics
        engagement_score = pd.Series(index=df.index, dtype=float)
        score_components = 0
        
        if 'session_duration' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['session_duration'] in df.columns:
            duration_col = ENGAGEMENT_MAPPING['session_duration']
            normalized_duration = df[duration_col] / df[duration_col].max()
            engagement_score += normalized_duration
            score_components += 1
        
        if 'page_views' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['page_views'] in df.columns:
            pv_col = ENGAGEMENT_MAPPING['page_views']
            normalized_pv = df[pv_col] / df[pv_col].max()
            engagement_score += normalized_pv
            score_components += 1
        
        if 'interactions' in ENGAGEMENT_MAPPING and ENGAGEMENT_MAPPING['interactions'] in df.columns:
            int_col = ENGAGEMENT_MAPPING['interactions']
            normalized_int = df[int_col] / df[int_col].max()
            engagement_score += normalized_int
            score_components += 1
        
        if score_components > 0:
            engagement_score = engagement_score / score_components
            
            # Segment users by engagement score
            high_engagement = (engagement_score >= 0.7).sum()
            medium_engagement = ((engagement_score >= 0.3) & (engagement_score < 0.7)).sum()
            low_engagement = (engagement_score < 0.3).sum()
            
            print(f"   🔥 High engagement users: {{high_engagement}} ({{high_engagement/len(df)*100:.1f}}%)")
            print(f"   😐 Medium engagement users: {{medium_engagement}} ({{medium_engagement/len(df)*100:.1f}}%)")
            print(f"   😴 Low engagement users: {{low_engagement}} ({{low_engagement/len(df)*100:.1f}}%)")
    
    print()
    
    # 5. Engagement Quality Assessment
    print("✅ ENGAGEMENT QUALITY ASSESSMENT")
    
    quality_scores = {{}}
    quality_issues = []
    
    # Session duration quality
    if 'avg_session_duration' in engagement_summary:
        duration = engagement_summary['avg_session_duration']
        if duration > 5:  # Assuming minutes
            quality_scores['session_duration'] = 'Good'
        elif duration > 2:
            quality_scores['session_duration'] = 'Fair'
        else:
            quality_scores['session_duration'] = 'Poor'
            quality_issues.append("Short average session duration - improve content engagement")
    
    # Bounce rate quality  
    if 'bounce_rate' in engagement_summary:
        bounce = engagement_summary['bounce_rate']
        if bounce < 30:
            quality_scores['bounce_rate'] = 'Excellent'
        elif bounce < 50:
            quality_scores['bounce_rate'] = 'Good'
        elif bounce < 70:
            quality_scores['bounce_rate'] = 'Fair'
        else:
            quality_scores['bounce_rate'] = 'Poor'
            quality_issues.append("High bounce rate - improve landing page relevance")
    
    # Page views quality
    if 'avg_pageviews' in engagement_summary:
        pageviews = engagement_summary['avg_pageviews']
        if pageviews > 5:
            quality_scores['page_depth'] = 'Excellent'
        elif pageviews > 3:
            quality_scores['page_depth'] = 'Good'
        elif pageviews > 1.5:
            quality_scores['page_depth'] = 'Fair'
        else:
            quality_scores['page_depth'] = 'Poor'
            quality_issues.append("Low page depth - improve internal linking and navigation")
    
    print("   📊 Engagement quality scores:")
    for metric, score in quality_scores.items():
        emoji = "🟢" if score in ['Excellent', 'Good'] else "🟡" if score == 'Fair' else "🔴"
        print(f"      {{emoji}} {{metric.replace('_', ' ').title()}}: {{score}}")
    
    if quality_issues:
        print("\\n   ⚠️ Areas for improvement:")
        for issue in quality_issues:
            print(f"      • {{issue}}")
    else:
        print("\\n   ✅ Overall engagement quality looks good!")
    
    print()
    
    # 6. Actionable Recommendations
    print("💡 ENGAGEMENT OPTIMIZATION RECOMMENDATIONS")
    
    # Performance-based recommendations
    if 'bounce_rate' in engagement_summary and engagement_summary['bounce_rate'] > 60:
        print("   🎯 High Priority: Reduce bounce rate through landing page optimization")
        print("      • Improve page load speed")
        print("      • Ensure content matches user intent")
        print("      • Add compelling calls-to-action")
    
    if 'avg_pageviews' in engagement_summary and engagement_summary['avg_pageviews'] < 2:
        print("   📄 Improve page depth and site exploration:")
        print("      • Add related content recommendations")
        print("      • Improve internal linking structure")
        print("      • Create content series or multi-part guides")
    
    if 'avg_session_duration' in engagement_summary and engagement_summary['avg_session_duration'] < 2:
        print("   ⏰ Increase session duration:")
        print("      • Add engaging multimedia content")
        print("      • Improve content readability and structure")
        print("      • Implement progressive disclosure techniques")
    
    if 'returning_users_pct' in engagement_summary and engagement_summary['returning_users_pct'] < 30:
        print("   🔄 Improve user retention:")
        print("      • Implement email newsletter or notifications")
        print("      • Create personalized user experiences")
        print("      • Add bookmark or save functionality")
    
    # General recommendations
    print("\\n   📊 General engagement optimization:")
    print("      • A/B test different page layouts and content formats")
    print("      • Implement user behavior tracking for deeper insights")
    print("      • Create user personas based on engagement patterns")
    print("      • Monitor engagement metrics by traffic source and device")
    
    # Segmentation recommendations
    if segment_analyses:
        print("\\n   🎯 Segment-specific optimization:")
        print("      • Create device-specific user experiences")
        print("      • Tailor content for different traffic sources")
        print("      • Implement dynamic content based on user behavior")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "session_duration": {
                "required": False,
                "description": "Time spent in session (seconds or minutes)",
                "data_type": "numeric",
                "examples": ["session_duration", "time_on_site", "visit_duration"]
            },
            "page_views": {
                "required": False,
                "description": "Number of pages viewed per session",
                "data_type": "numeric",
                "examples": ["page_views", "pages_per_session", "pageviews"]
            },
            "bounce_rate": {
                "required": False,
                "description": "Bounce rate (0-1 or 0-100)",
                "data_type": "numeric",
                "examples": ["bounce_rate", "bounced", "single_page_session"]
            },
            "interactions": {
                "required": False,
                "description": "Number of interactions or clicks per session",
                "data_type": "numeric",
                "examples": ["interactions", "clicks", "actions", "events"]
            },
            "scroll_depth": {
                "required": False,
                "description": "How far users scroll (percentage or ratio)",
                "data_type": "numeric",
                "examples": ["scroll_depth", "scroll_percentage", "page_scroll"]
            },
            "return_visits": {
                "required": False,
                "description": "Number of return visits per user",
                "data_type": "numeric",
                "examples": ["return_visits", "repeat_visits", "frequency"]
            },
            "device_type": {
                "required": False,
                "description": "Device type for segmentation",
                "data_type": "categorical",
                "examples": ["device", "device_type", "platform"]
            },
            "traffic_source": {
                "required": False,
                "description": "Traffic source for segmentation",
                "data_type": "categorical",
                "examples": ["source", "traffic_source", "referrer", "channel"]
            }
        }
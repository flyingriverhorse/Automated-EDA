"""ROI Analysis Component.

Focused analysis component for marketing ROI analysis
calculating return on investment across campaigns and channels.
"""
from typing import Dict, Any


class ROIAnalysis:
    """Focused component for ROI analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "roi_analysis",
            "display_name": "ROI Analysis",
            "description": "Calculate ROI, ROAS, and profitability metrics across marketing activities",
            "category": "marketing",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "💰",
            "tags": ["marketing", "roi", "roas", "profitability", "financial"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has ROI-related columns"""
        # For now, return True to allow all datasets (marketing analyses are flexible)
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None, selected_columns: Dict[str, str] = None) -> str:
        """Generate ROI analysis code with column selection support"""
        
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
ROI_MAPPING = {column_mapping}
"""
        else:
            column_mapping_code = """
# Auto-detect ROI columns (user can override these)
ROI_MAPPING = {}
possible_mappings = {
    'revenue': ['revenue', 'sales', 'income', 'earnings'],
    'cost': ['cost', 'spend', 'investment', 'expense'],
    'profit': ['profit', 'net_income', 'margin'],
    'impressions': ['impressions', 'views', 'reach'],
    'clicks': ['clicks', 'click_count'],
    'conversions': ['conversions', 'sales_count']
}

# Auto-detect columns
columns_lower = {col: col.lower() for col in df.columns}
for metric, possible_names in possible_mappings.items():
    for col_name, col_lower in columns_lower.items():
        for possible_name in possible_names:
            if possible_name in col_lower and metric not in ROI_MAPPING:
                ROI_MAPPING[metric] = col_name
                break
"""
        
        return f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== ROI ANALYSIS ===")
print("Return on Investment and profitability analysis")
print()

{column_mapping_code}

# Display detected/selected columns
print("📋 COLUMN MAPPING:")
for metric, col in ROI_MAPPING.items():
    if col in df.columns:
        print(f"   • {{metric.title()}}: {{col}}")
print()

# Calculate ROI metrics if columns are available
if 'revenue' in ROI_MAPPING and 'cost' in ROI_MAPPING:
    revenue_col = ROI_MAPPING['revenue']
    cost_col = ROI_MAPPING['cost']
    
    if revenue_col in df.columns and cost_col in df.columns:
        print("📊 ROI CALCULATIONS:")
        df['roi'] = ((df[revenue_col] - df[cost_col]) / df[cost_col]) * 100
        df['profit'] = df[revenue_col] - df[cost_col]
        
        print(f"   • Average ROI: {{df['roi'].mean():.2f}}%")
        print(f"   • Total Profit: ${{df['profit'].sum():,.2f}}")
        print(f"   • Total Revenue: ${{df[revenue_col].sum():,.2f}}")
        print(f"   • Total Cost: ${{df[cost_col].sum():,.2f}}")
        print()
        
        # ROI distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(df['roi'], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title('ROI Distribution')
        plt.xlabel('ROI (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
else:
    print("⚠️ Please map revenue and cost columns for ROI analysis")
    
    # Fallback: auto-detect potential columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 Available numeric columns: {{numeric_cols}}")
    
    # Look for potential revenue and cost columns
    revenue_candidates = [col for col in numeric_cols if any(keyword in col.lower() 
                         for keyword in ['revenue', 'sales', 'income', 'value'])]
    cost_candidates = [col for col in numeric_cols if any(keyword in col.lower() 
                      for keyword in ['cost', 'spend', 'budget', 'investment'])]
    
    if revenue_candidates:
        print(f"📈 Potential revenue columns: {{revenue_candidates}}")
    if cost_candidates:
        print(f"💰 Potential cost columns: {{cost_candidates}}")

print("\\n💡 ROI ANALYSIS RECOMMENDATIONS")
print("   • Calculate ROI by campaign, channel, and time period")
print("   • Set up ROI benchmarks and targets")
print("   • Analyze customer lifetime value vs acquisition cost")
print("   • Monitor profitability trends over time")

print("\\n" + "="*60)
'''
    
    def get_required_columns(self) -> Dict[str, Dict[str, Any]]:
        """Return information about columns this analysis can work with"""
        return {
            "revenue": {
                "required": False,
                "description": "Revenue generated",
                "data_type": "numeric",
                "examples": ["revenue", "sales", "income", "value"]
            },
            "spend": {
                "required": False,
                "description": "Amount spent or invested",
                "data_type": "numeric",
                "examples": ["spend", "cost", "budget", "investment"]
            }
        }
"""PCA Cumulative Variance Component.

Focused analysis component for PCA cumulative variance analysis only.
"""
from typing import Dict, Any


class PCACumulativeVarianceAnalysis:
    """Focused component for PCA cumulative variance visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "pca_cumulative_variance",
            "display_name": "PCA Cumulative Variance",
            "description": "Cumulative variance analysis for optimal PCA component selection",
            "category": "relationships",
            "subcategory": "pca",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "5-10 seconds",
            "icon": "chart-area",
            "tags": ["pca", "cumulative", "variance", "selection", "threshold"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has enough numeric columns for PCA"""
        if not data_preview:
            return True
        
        data = data_preview.get("data", [])
        if not data:
            return True
            
        # Check if we have at least 2 numeric columns
        numeric_count = 0
        for row in data[:5]:
            for value in row:
                try:
                    float(str(value))
                    numeric_count += 1
                    if numeric_count >= 2:
                        return True
                except (ValueError, TypeError):
                    continue
        return numeric_count >= 2
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate focused PCA cumulative variance analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== PCA CUMULATIVE VARIANCE ANALYSIS ===")
print("Optimal component selection based on variance thresholds")
print()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    print("❌ INSUFFICIENT NUMERIC COLUMNS")
    print("   PCA requires at least 2 numeric variables.")
else:
    print(f"📊 CUMULATIVE VARIANCE ANALYSIS FOR {len(numeric_cols)} VARIABLES")
    print()
    
    # Remove columns with too many missing values or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and df[col].notna().sum() >= max(10, len(df) * 0.3):
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns (removing columns with >70% missing or constant values)")
    
    if len(valid_cols) < 3:
        print("❌ INSUFFICIENT VALID COLUMNS")
        print("   Need at least 3 columns with sufficient data for PCA.")
    else:
        # Prepare data with imputation for missing values
        X = df[valid_cols].copy()
        
        # Handle missing values with imputation
        total_missing = X.isnull().sum().sum()
        if total_missing > 0:
            print(f"   • Found {total_missing:,} missing values")
            print(f"   • Imputing missing values with column means")
            X = X.fillna(X.mean())
        else:
            print(f"   • No missing values to handle")
        
        print(f"   • Final dataset: {len(X):,} rows × {len(valid_cols)} columns")
        
        if len(X) < 10:
            print("❌ INSUFFICIENT DATA POINTS")
            print("   Need at least 10 data points for reliable PCA.")
        else:
            print(f"   • Using {len(X):,} observations for PCA cumulative variance analysis")
            print()
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA with all components
            n_components = min(len(valid_cols), len(X) - 1)
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
        
        # Extract variance information
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # Create comprehensive cumulative variance visualization
        fig = plt.figure(figsize=(16, 10))
        
        # Main cumulative variance plot
        ax1 = plt.subplot(2, 2, 1)
        components = range(1, len(explained_var_ratio) + 1)
        
        # Plot cumulative variance
        ax1.plot(components, cumulative_var_ratio * 100, 'bo-', linewidth=3, markersize=8, label='Cumulative Variance')
        ax1.fill_between(components, cumulative_var_ratio * 100, alpha=0.3, color='skyblue')
        
        # Add threshold lines
        thresholds = [50, 60, 70, 80, 85, 90, 95, 99]
        colors = ['gray', 'orange', 'yellow', 'green', 'lightgreen', 'blue', 'red', 'purple']
        
        for threshold, color in zip(thresholds, colors):
            ax1.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f'{threshold}%')
            
            # Find component that reaches this threshold
            if np.any(cumulative_var_ratio >= threshold/100):
                threshold_comp = np.argmax(cumulative_var_ratio >= threshold/100) + 1
                ax1.axvline(x=threshold_comp, color=color, linestyle=':', alpha=0.5)
                ax1.plot(threshold_comp, threshold, 'o', color=color, markersize=8, markeredgecolor='black')
        
        ax1.set_title('Cumulative Explained Variance by Component', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 105)
        ax1.set_xticks(components)
        
        # Individual variance contribution
        ax2 = plt.subplot(2, 2, 2)
        bars = ax2.bar(components, explained_var_ratio * 100, alpha=0.7, color='lightcoral')
        ax2.set_title('Individual Component Variance Contribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(components)
        
        # Add value labels on bars
        for i, (bar, var_pct) in enumerate(zip(bars, explained_var_ratio * 100)):
            if var_pct > 1:  # Only label bars with >1% variance
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{var_pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Variance increment analysis
        ax3 = plt.subplot(2, 2, 3)
        if len(explained_var_ratio) > 1:
            variance_increments = np.diff(cumulative_var_ratio * 100)
            ax3.plot(components[1:], variance_increments, 'go-', linewidth=2, markersize=6)
            ax3.set_title('Variance Increment Between Components', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Component (n to n+1)')
            ax3.set_ylabel('Variance Increment (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(components[1:])
        
        # Component selection recommendations table
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        # Create recommendation table
        table_data = []
        headers = ['Threshold', 'Components', 'Actual %', 'Reduction']
        
        for threshold in [80, 85, 90, 95, 99]:
            if np.any(cumulative_var_ratio >= threshold/100):
                comp_needed = np.argmax(cumulative_var_ratio >= threshold/100) + 1
                actual_var = cumulative_var_ratio[comp_needed-1] * 100
                reduction = (1 - comp_needed/len(numeric_cols)) * 100
                table_data.append([f'{threshold}%', f'{comp_needed}/{len(numeric_cols)}', 
                                 f'{actual_var:.1f}%', f'{reduction:.1f}%'])
            else:
                table_data.append([f'{threshold}%', f'{len(numeric_cols)}/{len(numeric_cols)}', 
                                 f'{cumulative_var_ratio[-1]*100:.1f}%', '0.0%'])
        
        table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Component Selection Recommendations', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        print()
        
        # Detailed analysis output
        print("📊 CUMULATIVE VARIANCE ANALYSIS RESULTS:")
        print()
        
        # Summary statistics
        print("📈 VARIANCE SUMMARY:")
        print(f"   • Total components available: {n_components}")
        print(f"   • First component explains: {explained_var_ratio[0]*100:.1f}% variance")
        print(f"   • First two components explain: {cumulative_var_ratio[1]*100:.1f}% variance")
        print(f"   • All components explain: {cumulative_var_ratio[-1]*100:.1f}% variance")
        print()
        
        # Threshold analysis
        print("🎯 THRESHOLD ANALYSIS:")
        for threshold in [70, 80, 85, 90, 95, 99]:
            if np.any(cumulative_var_ratio >= threshold/100):
                comp_needed = np.argmax(cumulative_var_ratio >= threshold/100) + 1
                actual_var = cumulative_var_ratio[comp_needed-1] * 100
                reduction = (1 - comp_needed/len(numeric_cols)) * 100
                
                print(f"   • {threshold}% variance: {comp_needed} components (actual: {actual_var:.1f}%)")
                print(f"     → Dimensionality reduction: {reduction:.1f}%")
                print(f"     → Components retained: {comp_needed}/{len(numeric_cols)}")
            else:
                print(f"   • {threshold}% variance: Requires all {n_components} components")
            print()
        
        # Efficiency analysis
        print("⚡ EFFICIENCY ANALYSIS:")
        
        # Most efficient component ranges
        if len(explained_var_ratio) >= 3:
            top_3_var = cumulative_var_ratio[2] * 100
            top_5_var = cumulative_var_ratio[min(4, len(cumulative_var_ratio)-1)] * 100
            
            print(f"   • First 3 components capture: {top_3_var:.1f}% variance")
            print(f"   • First 5 components capture: {top_5_var:.1f}% variance")
            
            # Variance per component ratio
            efficiency = explained_var_ratio / np.arange(1, len(explained_var_ratio) + 1) * 100
            most_efficient = np.argmax(efficiency) + 1
            print(f"   • Most efficient single component: PC{most_efficient} ({explained_var_ratio[most_efficient-1]*100:.1f}%)")
        
        print()
        
        # Recommendations based on common use cases
        print("💡 USE CASE RECOMMENDATIONS:")
        print()
        
        # For visualization (2-3 components)
        if len(cumulative_var_ratio) >= 2:
            viz_var = cumulative_var_ratio[1] * 100
            print(f"📊 VISUALIZATION (2-3 components):")
            print(f"   • 2D visualization: {viz_var:.1f}% variance retained")
            if len(cumulative_var_ratio) >= 3:
                viz_3d_var = cumulative_var_ratio[2] * 100
                print(f"   • 3D visualization: {viz_3d_var:.1f}% variance retained")
            print()
        
        # For machine learning (80-95% variance)
        ml_80_comp = np.argmax(cumulative_var_ratio >= 0.8) + 1 if np.any(cumulative_var_ratio >= 0.8) else n_components
        ml_95_comp = np.argmax(cumulative_var_ratio >= 0.95) + 1 if np.any(cumulative_var_ratio >= 0.95) else n_components
        
        print(f"🤖 MACHINE LEARNING:")
        print(f"   • Standard preprocessing (80%): {ml_80_comp} components")
        print(f"   • High fidelity (95%): {ml_95_comp} components")
        print()
        
        # For data compression
        compression_90 = np.argmax(cumulative_var_ratio >= 0.9) + 1 if np.any(cumulative_var_ratio >= 0.9) else n_components
        compression_ratio = (1 - compression_90/len(numeric_cols)) * 100
        
        print(f"🗜️  DATA COMPRESSION (90% variance):")
        print(f"   • Components needed: {compression_90}")
        print(f"   • Compression ratio: {compression_ratio:.1f}%")
        print(f"   • Storage savings: {compression_ratio:.1f}% reduction")
        
        print()
        print("📖 INTERPRETATION GUIDE:")
        print("   • Steep rise → High information components")
        print("   • Plateau region → Diminishing returns")
        print("   • 80-95% variance typical for most applications")
        print("   • Higher thresholds preserve more original information")
        print("   • Lower thresholds provide more dimensionality reduction")

print("\\n" + "="*50)
print("✅ PCA CUMULATIVE VARIANCE ANALYSIS COMPLETE")
print("="*50)
'''
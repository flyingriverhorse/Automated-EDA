"""PCA Scree Plot Component.

Focused analysis component for creating PCA scree plots only.
"""
from typing import Dict, Any


class PCAScreePlotAnalysis:
    """Focused component for PCA scree plot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "pca_scree_plot",
            "display_name": "PCA Scree Plot",
            "description": "Scree plot showing eigenvalues for PCA component selection",
            "category": "relationships",
            "subcategory": "pca",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-15 seconds",
            "icon": "chart-line-down",
            "tags": ["pca", "scree", "eigenvalues", "dimensionality", "components"]
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
        """Generate focused PCA scree plot analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== PCA SCREE PLOT ANALYSIS ===")
print("Eigenvalue visualization for component selection")
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
    print(f"📊 PERFORMING PCA SCREE ANALYSIS ON {len(numeric_cols)} VARIABLES")
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
            print(f"   • Using {len(X):,} observations for PCA scree plot")
            print()
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA with all components
            n_components = min(len(valid_cols), len(X) - 1)
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
        
        # Extract eigenvalues and explained variance
        eigenvalues = pca.explained_variance_
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        print(f"🔍 PCA SCREE ANALYSIS RESULTS:")
        print(f"   • Total components: {n_components}")
        print(f"   • Total variance explained: {cumulative_var_ratio[-1]:.3f} (100%)")
        print()
        
        # Create the scree plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Classic Scree Plot (Eigenvalues)
        components = range(1, len(eigenvalues) + 1)
        ax1.plot(components, eigenvalues, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser Criterion (eigenvalue = 1)')
        
        # Highlight Kaiser criterion components
        kaiser_components = eigenvalues > 1
        if kaiser_components.sum() > 0:
            ax1.fill_between(components, eigenvalues, 1, where=(eigenvalues > 1), 
                           alpha=0.3, color='green', label=f'Retained ({kaiser_components.sum()} components)')
        
        ax1.set_title('PCA Scree Plot - Eigenvalues', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(components)
        
        # Add eigenvalue annotations
        for i, (comp, eigenval) in enumerate(zip(components, eigenvalues)):
            ax1.annotate(f'{eigenval:.2f}', (comp, eigenval), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        # Plot 2: Explained Variance Scree Plot
        ax2.plot(components, explained_var_ratio * 100, 'go-', linewidth=2, markersize=8, label='Individual')
        ax2.plot(components, cumulative_var_ratio * 100, 'ro-', linewidth=2, markersize=8, label='Cumulative')
        
        # Add threshold lines for common variance levels
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(y=95, color='purple', linestyle='--', alpha=0.7, label='95% threshold')
        
        ax2.set_title('PCA Scree Plot - Explained Variance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(components)
        ax2.set_ylim(0, 105)
        
        # Add percentage annotations
        for i, (comp, var_ratio) in enumerate(zip(components, explained_var_ratio)):
            ax2.annotate(f'{var_ratio*100:.1f}%', (comp, var_ratio*100), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        print()
        
        # Component selection recommendations
        print("🎯 COMPONENT SELECTION RECOMMENDATIONS:")
        print()
        
        # Kaiser criterion
        kaiser_count = (eigenvalues > 1).sum()
        print(f"📏 KAISER CRITERION (eigenvalue > 1):")
        print(f"   • Recommended components: {kaiser_count}")
        print(f"   • Variance explained: {cumulative_var_ratio[kaiser_count-1]*100:.1f}%")
        print()
        
        # Elbow method suggestion
        print("💪 ELBOW METHOD:")
        print("   • Look for the 'elbow' in the scree plot")
        print("   • Components before the sharp drop-off")
        
        # Find approximate elbow using second derivative
        if len(eigenvalues) > 2:
            second_derivatives = []
            for i in range(1, len(eigenvalues) - 1):
                second_deriv = eigenvalues[i-1] - 2*eigenvalues[i] + eigenvalues[i+1]
                second_derivatives.append(second_deriv)
            
            if second_derivatives:
                elbow_index = np.argmax(second_derivatives) + 2  # +2 because we start from index 1
                print(f"   • Suggested elbow at component: {elbow_index}")
                print(f"   • Variance explained: {cumulative_var_ratio[elbow_index-1]*100:.1f}%")
        print()
        
        # Variance thresholds
        print("📊 VARIANCE THRESHOLD RECOMMENDATIONS:")
        
        # Find components for different variance levels
        var_80_idx = np.argmax(cumulative_var_ratio >= 0.8) + 1 if np.any(cumulative_var_ratio >= 0.8) else n_components
        var_90_idx = np.argmax(cumulative_var_ratio >= 0.9) + 1 if np.any(cumulative_var_ratio >= 0.9) else n_components
        var_95_idx = np.argmax(cumulative_var_ratio >= 0.95) + 1 if np.any(cumulative_var_ratio >= 0.95) else n_components
        
        print(f"   • For 80% variance: {var_80_idx} components ({cumulative_var_ratio[var_80_idx-1]*100:.1f}% actual)")
        print(f"   • For 90% variance: {var_90_idx} components ({cumulative_var_ratio[var_90_idx-1]*100:.1f}% actual)")
        print(f"   • For 95% variance: {var_95_idx} components ({cumulative_var_ratio[var_95_idx-1]*100:.1f}% actual)")
        print()
        
        # Dimensionality reduction benefits
        print("🎯 DIMENSIONALITY REDUCTION BENEFITS:")
        for threshold in [80, 90, 95]:
            var_idx = np.argmax(cumulative_var_ratio >= threshold/100) + 1 if np.any(cumulative_var_ratio >= threshold/100) else n_components
            reduction = (1 - var_idx / len(numeric_cols)) * 100
            print(f"   • {threshold}% variance: {reduction:.1f}% dimension reduction ({var_idx}/{len(numeric_cols)} components)")
        print()
        
        # Detailed component table
        print("📋 DETAILED COMPONENT ANALYSIS:")
        print(f"{'PC':>4} | {'Eigenval':>9} | {'Var%':>7} | {'Cum%':>7}")
        print("-" * 35)
        
        for i in range(min(10, len(eigenvalues))):  # Show first 10 components
            eigenval = eigenvalues[i]
            var_pct = explained_var_ratio[i] * 100
            cum_pct = cumulative_var_ratio[i] * 100
            print(f"PC{i+1:>2} | {eigenval:>8.3f} | {var_pct:>6.1f} | {cum_pct:>6.1f}")
        
        if len(eigenvalues) > 10:
            print(f"... ({len(eigenvalues) - 10} more components)")
        
        print()
        print("📖 SCREE PLOT INTERPRETATION:")
        print("   • Steep slope → High variance components")
        print("   • Gradual slope → Low variance components") 
        print("   • Elbow point → Optimal number of components")
        print("   • Kaiser criterion → Keep eigenvalues > 1")
        print("   • Choose based on analysis goals and interpretability")

print("\\n" + "="*50)
print("✅ PCA SCREE PLOT ANALYSIS COMPLETE")
print("="*50)
'''
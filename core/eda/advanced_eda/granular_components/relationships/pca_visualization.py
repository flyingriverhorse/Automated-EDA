"""PCA Visualization Component.

Focused analysis component for PCA scatter plot visualizations only.
"""
from typing import Dict, Any


class PCAVisualizationAnalysis:
    """Focused component for PCA scatter plot visualization"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "pca_visualization",
            "display_name": "PCA Visualization",
            "description": "2D and 3D PCA scatter plot visualizations of transformed data",
            "category": "relationships",
            "subcategory": "pca",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "10-20 seconds",
            "icon": "chart-scatter-3d",
            "tags": ["pca", "visualization", "scatter", "2d", "3d", "transformation"]
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview: Dict[str, Any] = None) -> bool:
        """Check if dataset has enough numeric columns for PCA"""
        if not data_preview:
            return True
        
        sample_data = data_preview.get("data", [])
        if not sample_data:
            return True
            
        headers = data_preview.get("columns", [])
        numeric_cols = []
        
        # Check each column for numeric data
        for i, col in enumerate(headers):
            try:
                values = [row[i] for row in sample_data[:5] if len(row) > i]
                numeric_values = 0
                for v in values:
                    try:
                        float(str(v))
                        numeric_values += 1
                    except (ValueError, TypeError):
                        continue
                # If most sampled values are numeric, consider column numeric
                if numeric_values >= len(values) * 0.5:
                    numeric_cols.append(col)
            except:
                continue
        
        # PCA visualization needs at least 2 numeric columns
        return len(numeric_cols) >= 2
    
    def analyze(self, df, **kwargs):
        """
        Perform PCA visualization analysis.
        
        Args:
            df: DataFrame to analyze  
            **kwargs: Additional parameters
                
        Returns:
            Dict with analysis results
        """
        try:
            print("="*60)
            print("📊 PCA VISUALIZATION ANALYSIS")
            print("="*60)
            
            # Execute the generated code logic with proper scope
            exec(self.generate_code(), {'df': df, 'plt': __import__('matplotlib.pyplot'), 'pd': __import__('pandas'), 
                                       'np': __import__('numpy'), 'sns': __import__('seaborn'),
                                       'PCA': __import__('sklearn.decomposition', fromlist=['PCA']).PCA,
                                       'StandardScaler': __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler,
                                       'warnings': __import__('warnings')})
            
            return {"status": "completed", "message": "PCA visualization analysis completed successfully"}
            
        except Exception as e:
            error_msg = f"Error in PCA visualization analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate focused PCA visualization code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("=== PCA VISUALIZATION ANALYSIS ===")
print("2D and 3D visualizations of PCA-transformed data")
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
    print(f"📊 PCA VISUALIZATION FOR {len(numeric_cols)} VARIABLES")
    print()
    
    # Remove columns with too many missing values or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and df[col].notna().sum() >= max(10, len(df) * 0.3):
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns (removing columns with >70% missing or constant values)")
    
    # Limit dimensions for better visualization and performance
    max_dimensions = 20  # Reasonable limit for PCA visualization
    if len(valid_cols) > max_dimensions:
        print(f"⚠️  LIMITING DIMENSIONS: Using top {max_dimensions} columns by variance for better visualization")
        # Calculate variance for each column and select top ones
        col_variances = [(col, df[col].var()) for col in valid_cols]
        col_variances.sort(key=lambda x: x[1], reverse=True)
        valid_cols = [col for col, _ in col_variances[:max_dimensions]]
        print(f"📋 Selected columns: {valid_cols[:5]}{'...' if len(valid_cols) > 5 else ''}")
    
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
            print(f"   • Using {len(X):,} observations for PCA visualization")
            print()
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA for visualization (2D and 3D)
            n_components_viz = min(3, len(valid_cols), len(X) - 1)
            pca_viz = PCA(n_components=n_components_viz)
            X_pca = pca_viz.fit_transform(X_scaled)
        
        # Get variance explained by visualization components
        var_explained = pca_viz.explained_variance_ratio_
        cumulative_var = np.cumsum(var_explained)
        
        print(f"🎯 VISUALIZATION COMPONENTS:")
        print(f"   • PC1 explains: {var_explained[0]*100:.1f}% variance")
        if len(var_explained) > 1:
            print(f"   • PC2 explains: {var_explained[1]*100:.1f}% variance")
            print(f"   • PC1 + PC2 explain: {cumulative_var[1]*100:.1f}% variance")
        if len(var_explained) > 2:
            print(f"   • PC3 explains: {var_explained[2]*100:.1f}% variance")
            print(f"   • PC1 + PC2 + PC3 explain: {cumulative_var[2]*100:.1f}% variance")
        print()
        
        # Check if we have categorical columns for coloring
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_column = None
        
        if categorical_cols:
            # Find the categorical column with reasonable number of categories
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 10:  # Good range for visualization
                    color_column = col
                    break
        
        if color_column:
            # Use original DataFrame index for categorical data
            color_data = df[color_column]
            unique_categories = color_data.unique()
            print(f"   Using '{color_column}' for coloring ({len(unique_categories)} categories)")
        else:
            color_data = None
            print("   No suitable categorical column found for coloring")
        
        print()
        
        # Create visualization plots
        if n_components_viz >= 2:
            # 2D PCA Plot
            fig, axes = plt.subplots(1, 2 if n_components_viz >= 3 else 1, figsize=(15 if n_components_viz >= 3 else 8, 6))
            if n_components_viz < 3:
                axes = [axes]
            
            # Plot 1: 2D PCA
            ax1 = axes[0]
            
            if color_data is not None:
                # Color by categorical variable
                for i, category in enumerate(unique_categories):
                    mask = color_data == category
                    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              label=f'{category}', alpha=0.7, s=50)
                ax1.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Single color
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50, color='skyblue')
            
            ax1.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)')
            ax1.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)')
            ax1.set_title(f'PCA 2D Visualization\\n({cumulative_var[1]*100:.1f}% total variance)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add origin lines
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # 3D PCA Plot (if available)
            if n_components_viz >= 3:
                ax2 = axes[1]
                ax2.remove()  # Remove the 2D axis
                ax2 = fig.add_subplot(122, projection='3d')  # Add 3D axis
                
                if color_data is not None:
                    # Color by categorical variable
                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
                    for i, category in enumerate(unique_categories):
                        mask = color_data == category
                        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                                  label=f'{category}', alpha=0.7, s=50, color=colors[i])
                    ax2.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Single color
                    ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                               alpha=0.7, s=50, color='skyblue')
                
                ax2.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
                ax2.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
                ax2.set_zlabel(f'PC3 ({var_explained[2]*100:.1f}%)')
                ax2.set_title(f'PCA 3D Visualization\\n({cumulative_var[2]*100:.1f}% total variance)', 
                             fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            print()
            
        # Component loading analysis
        print("🔍 PRINCIPAL COMPONENT LOADINGS:")
        print("   (Contribution of original variables to each PC)")
        print()
        
        # Get component loadings
        loadings = pca_viz.components_.T * np.sqrt(pca_viz.explained_variance_)
        loadings_df = pd.DataFrame(loadings, 
                                 columns=[f'PC{i+1}' for i in range(n_components_viz)],
                                 index=valid_cols)
        
        print(loadings_df.round(3).to_string())
        print()
        
        # Identify most important variables for each component
        print("🎯 KEY VARIABLES FOR EACH COMPONENT:")
        for i in range(n_components_viz):
            pc_loadings = np.abs(loadings[:, i])
            top_vars_idx = np.argsort(pc_loadings)[-3:][::-1]  # Top 3
            
            print(f"\\n   PC{i+1} (most influenced by):")
            for idx in top_vars_idx:
                var_name = valid_cols[idx]
                loading_val = loadings[idx, i]
                contribution = pc_loadings[idx] / np.sum(pc_loadings) * 100
                print(f"      • {var_name}: {loading_val:.3f} ({contribution:.1f}% contribution)")
        
        print()
        
        print("📖 VISUALIZATION INTERPRETATION:")
        print("   • PC1 (x-axis) captures the most variance")
        print("   • PC2 (y-axis) captures the second most variance") 
        print("   • Clusters indicate similar data patterns")
        print("   • Distance from origin indicates variability")
        print("   • Loading vectors show variable influence")
        print("   • Longer arrows = stronger influence on that PC")
        print("   • Arrow direction shows positive/negative correlation")

print("\\n" + "="*50)
print("✅ PCA VISUALIZATION ANALYSIS COMPLETE")
print("="*50)
'''


def get_component():
    """Return the analysis component."""
    return PCAVisualizationAnalysis
"""PCA Heatmap Analysis Component.

Focused component specifically for PCA component loadings heatmap visualization.
"""
from typing import Dict, Any


class PCAHeatmapAnalysis:
    """Focused component for PCA loadings heatmap visualization"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_metadata():
        """Return metadata for this analysis component."""
        return {
            "name": "pca_heatmap",
            "display_name": "PCA Heatmap",
            "description": "Heatmap visualization of PCA component loadings showing variable contributions",
            "category": "relationships",
            "subcategory": "PCA Analysis",
            "complexity": "intermediate",
            "tags": ["pca", "heatmap", "loadings", "visualization", "components"],
            "estimated_runtime": "10-20 seconds",
            "required_data_types": ["numeric"],
            "data_requirements": {
                "min_numeric_columns": 3,
                "min_rows": 10,
                "handles_missing_values": True,
                "requires_target": False
            }
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if dataset is compatible with PCA heatmap analysis"""
        if data_preview is None:
            return True
        
        # Handle DataFrame input (for direct testing)
        if hasattr(data_preview, 'select_dtypes'):  # It's a DataFrame
            numeric_cols = data_preview.select_dtypes(include=['int64', 'float64']).columns.tolist()
            return len(numeric_cols) >= 2
        
        # Handle service data preview format (dictionary)
        if isinstance(data_preview, dict):
            if data_preview.get('empty', False):
                return True
            
            # Use numeric_columns provided by the service if available
            numeric_cols = data_preview.get("numeric_columns", [])
            
            # Fallback: manually check sample data if numeric_columns not provided
            if not numeric_cols:
                sample_data = data_preview.get("sample_data", []) or data_preview.get("data", [])
                if sample_data and len(sample_data) > 0:
                    headers = data_preview.get("columns", [])
                    for i, col in enumerate(headers):
                        try:
                            # Sample a few values to check if numeric
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
            
            # PCA heatmap requires at least 2 numeric columns
            return len(numeric_cols) >= 2
        
        return True
    
    def analyze(self, df, **kwargs):
        """
        Perform PCA heatmap analysis.
        
        Args:
            df: DataFrame to analyze
            **kwargs: Additional parameters
                
        Returns:
            Dict with analysis results
        """
        try:
            print("="*60)
            print("🔥 PCA LOADINGS HEATMAP ANALYSIS")
            print("="*60)
            
            # Execute the generated code logic with proper scope
            exec(self.generate_code(), {'df': df, 'plt': __import__('matplotlib.pyplot'), 'pd': __import__('pandas'), 
                                       'np': __import__('numpy'), 'sns': __import__('seaborn'),
                                       'PCA': __import__('sklearn.decomposition', fromlist=['PCA']).PCA,
                                       'StandardScaler': __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler,
                                       'warnings': __import__('warnings')})
            
            return {"status": "completed", "message": "PCA heatmap analysis completed successfully"}
            
        except Exception as e:
            error_msg = f"Error in PCA heatmap analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def generate_code(self, data_preview=None):
        """Generate PCA heatmap analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔥 PCA LOADINGS HEATMAP ANALYSIS")
print("="*60)
print("Heatmap visualization of variable contributions to principal components")
print()

# Setup matplotlib style
plt.style.use('default')

# Get numeric columns (respecting any column filtering that was applied)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"📊 CREATING PCA HEATMAP FOR {len(numeric_cols)} VARIABLES")
print()

if len(numeric_cols) < 3:
    print("❌ INSUFFICIENT NUMERIC COLUMNS")
    print("   Need at least 3 numeric columns for PCA heatmap.")
else:
    # Remove columns with too many missing values or constant values
    valid_cols = []
    for col in numeric_cols:
        if df[col].nunique() > 1 and df[col].notna().sum() >= max(10, len(df) * 0.3):
            valid_cols.append(col)
    
    print(f"📋 Using {len(valid_cols)} valid columns (removing columns with >70% missing or constant values)")
    
    # Limit dimensions for better visualization and performance
    max_dimensions = 25  # Slightly higher for heatmap since it can handle more
    if len(valid_cols) > max_dimensions:
        print(f"⚠️  LIMITING DIMENSIONS: Using top {max_dimensions} columns by variance for better heatmap")
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
            print(f"   • Using {len(X):,} observations for PCA heatmap")
            print()
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            n_components = min(len(valid_cols), len(X) - 1, 10)  # Max 10 components for visualization
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            
            # Get component loadings
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            print(f"🎯 HEATMAP COMPONENTS:")
            print(f"   • Showing first {n_components} principal components")
            for i in range(n_components):
                print(f"   • PC{i+1} explains: {pca.explained_variance_ratio_[i]*100:.1f}% variance")
            print()
            
            # Create simplified heatmap visualization (single plot)
            print("🎨 Creating PCA loadings heatmap...")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, max(8, len(valid_cols) * 0.25)))
            
            # Create the main loadings heatmap
            im = ax.imshow(loadings, cmap='RdBu_r', aspect='auto', 
                          vmin=-1, vmax=1, interpolation='nearest')
            
            # Set title and labels
            ax.set_title('🔥 PCA Component Loadings Heatmap\\n' +
                        f'Variables vs Principal Components (n_components={n_components})', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Principal Components', fontweight='bold')
            ax.set_ylabel('Variables', fontweight='bold')
            
            # Set ticks and labels
            ax.set_xticks(range(n_components))
            ax.set_xticklabels([f'PC{i+1}\\n({pca.explained_variance_ratio_[i]*100:.1f}%)' 
                               for i in range(n_components)])
            ax.set_yticks(range(len(valid_cols)))
            ax.set_yticklabels(valid_cols, fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Loading Value', rotation=270, labelpad=20, fontweight='bold')
            
            # Add minimal annotations only for strong loadings to avoid clutter
            if len(valid_cols) <= 30:  # Only annotate if manageable number of variables
                for i in range(len(valid_cols)):
                    for j in range(n_components):
                        loading_val = loadings[i, j]
                        if abs(loading_val) > 0.4:  # Only annotate strong loadings
                            text_color = 'white' if abs(loading_val) > 0.6 else 'black'
                            ax.text(j, i, f'{loading_val:.2f}', ha='center', va='center',
                                   color=text_color, fontweight='bold', fontsize=7)
            
            plt.tight_layout()
            plt.show()
            print()
            
            # Brief summary
            print("� HEATMAP SUMMARY:")
            print(f"   • Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
            print(f"   • Components analyzed: {n_components}")
            print(f"   • Variables included: {len(valid_cols)}")
            
            # Find most influential variable
            max_idx = np.unravel_index(np.argmax(np.abs(loadings)), loadings.shape)
            most_influential_var = valid_cols[max_idx[0]]
            most_influential_pc = f'PC{max_idx[1]+1}'
            most_influential_value = loadings[max_idx[0], max_idx[1]]
            
            print(f"   • Most influential: {most_influential_var} → {most_influential_pc}: {most_influential_value:.3f}")
            print("   • 🔴 Red: negative correlation, 🔵 Blue: positive correlation")

        print("\\n" + "="*60)
        print("✅ PCA loadings heatmap analysis complete!")
        print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return PCAHeatmapAnalysis
"""Outlier Scatter Plot Matrix Component.

Focused analysis component for visualizing multivariate outliers using scatter plot matrices.
"""
from typing import Dict, Any


class OutlierScatterMatrixVisualization:
    """Focused component for outlier visualization in scatter plots"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "outlier_scatter_matrix",
            "display_name": "Outlier Scatter Matrix",
            "description": "Scatter plot matrix for detecting multivariate outliers",
            "category": "outliers",
            "subcategory": "visualization",
            "complexity": "intermediate",
            "required_data_types": ["numeric"],
            "estimated_runtime": "15-30 seconds",
            "icon": "chart-scatter",
            "tags": ["visualization", "outliers", "scatter", "multivariate", "mahalanobis"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if dataset has enough numeric columns"""
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
        """Generate focused outlier scatter matrix visualization code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis
import warnings
warnings.filterwarnings('ignore')

print("=== OUTLIER SCATTER MATRIX VISUALIZATION ===")
print("Multivariate outlier detection using scatter plot matrix")
print()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    print("❌ INSUFFICIENT NUMERIC COLUMNS")
    print("   This analysis requires at least 2 numeric columns.")
else:
    print(f"📊 CREATING SCATTER MATRIX FOR MULTIVARIATE OUTLIER DETECTION")
    print()
    
    # Select subset for scatter matrix (max 6 variables for readability)
    scatter_cols = numeric_cols[:6]
    print(f"   Using {len(scatter_cols)} columns: {', '.join(scatter_cols)}")
    
    # Prepare data
    scatter_data = df[scatter_cols].dropna()
    
    if len(scatter_data) < 10:
        print("❌ INSUFFICIENT DATA POINTS")
        print("   Need at least 10 complete cases for meaningful analysis.")
    else:
        print(f"   Complete cases: {len(scatter_data)}")
        print()
        
        # Calculate Mahalanobis distance for multivariate outliers
        multivariate_outliers = np.zeros(len(scatter_data), dtype=bool)
        
        try:
            # Compute covariance matrix and its inverse
            cov_matrix = np.cov(scatter_data.T)
            
            # Check if covariance matrix is invertible
            if np.linalg.det(cov_matrix) == 0:
                print("⚠️  Covariance matrix is singular, using pseudo-inverse")
                cov_inv = np.linalg.pinv(cov_matrix)
            else:
                cov_inv = np.linalg.inv(cov_matrix)
            
            # Calculate Mahalanobis distances
            mean_vec = np.mean(scatter_data, axis=0)
            mahal_distances = []
            
            for idx, (_, row) in enumerate(scatter_data.iterrows()):
                try:
                    dist = mahalanobis(row.values, mean_vec, cov_inv)
                    mahal_distances.append(dist)
                except Exception as e:
                    mahal_distances.append(0)
            
            mahal_distances = np.array(mahal_distances)
            
            # Identify multivariate outliers using chi-square threshold
            # Degrees of freedom = number of variables
            threshold = stats.chi2.ppf(0.95, df=len(scatter_cols))  # 95% confidence
            multivariate_outliers = mahal_distances > threshold
            
            n_multivariate_outliers = multivariate_outliers.sum()
            print(f"🎯 MULTIVARIATE OUTLIER DETECTION RESULTS:")
            print(f"   • Chi-square threshold (95%): {threshold:.2f}")
            print(f"   • Multivariate outliers detected: {n_multivariate_outliers} ({n_multivariate_outliers/len(scatter_data)*100:.1f}%)")
            print(f"   • Maximum Mahalanobis distance: {np.max(mahal_distances):.2f}")
            print(f"   • Mean Mahalanobis distance: {np.mean(mahal_distances):.2f}")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not calculate multivariate outliers: {e}")
            print("   Using univariate outlier detection instead.")
            
        # Create scatter plot matrix
        n_vars = len(scatter_cols)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(3*n_vars, 3*n_vars))
        fig.suptitle('Scatter Plot Matrix with Multivariate Outlier Highlighting', fontsize=16, fontweight='bold')
        
        # Ensure axes is always 2D
        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_vars == 2:
            if axes.ndim == 1:
                axes = axes.reshape(2, 1) if len(axes) == 2 else axes.reshape(1, 2)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if n_vars == 1:
                    ax = axes[0, 0]
                else:
                    ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram with outlier highlighting
                    col_data = scatter_data.iloc[:, i]
                    
                    # Plot all data
                    ax.hist(col_data, bins=20, alpha=0.6, color='lightblue', label='Normal')
                    
                    # Highlight multivariate outliers
                    if n_multivariate_outliers > 0:
                        outlier_data = col_data[multivariate_outliers]
                        if len(outlier_data) > 0:
                            ax.hist(outlier_data, bins=20, alpha=0.8, color='red', label='M-Outliers')
                    
                    ax.set_title(scatter_cols[i], fontweight='bold')
                    if n_multivariate_outliers > 0:
                        ax.legend(fontsize=8)
                    
                else:
                    # Off-diagonal: scatter plot
                    x_data = scatter_data.iloc[:, j]
                    y_data = scatter_data.iloc[:, i]
                    
                    # Plot normal points
                    normal_mask = ~multivariate_outliers
                    if normal_mask.sum() > 0:
                        ax.scatter(x_data[normal_mask], y_data[normal_mask], 
                                 alpha=0.6, s=20, color='blue', label='Normal')
                    
                    # Highlight multivariate outliers
                    if n_multivariate_outliers > 0:
                        ax.scatter(x_data[multivariate_outliers], y_data[multivariate_outliers], 
                                 alpha=0.8, s=40, color='red', marker='x', linewidth=2, label='M-Outliers')
                    
                    # Add correlation coefficient
                    try:
                        corr_coef = x_data.corr(y_data)
                        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    if i == n_vars - 1:
                        ax.set_xlabel(scatter_cols[j])
                    if j == 0:
                        ax.set_ylabel(scatter_cols[i])
                    
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print()
        
        # Create a separate plot showing Mahalanobis distances
        if len(mahal_distances) > 0:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Mahalanobis distance distribution
            ax1.hist(mahal_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
            ax1.set_title('Distribution of Mahalanobis Distances')
            ax1.set_xlabel('Mahalanobis Distance')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Index plot of Mahalanobis distances
            ax2.scatter(range(len(mahal_distances)), mahal_distances, alpha=0.6, color='blue')
            ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
            
            # Highlight outliers
            if n_multivariate_outliers > 0:
                outlier_indices = np.where(multivariate_outliers)[0]
                ax2.scatter(outlier_indices, mahal_distances[outlier_indices], 
                           color='red', s=50, alpha=0.8, label='Outliers')
            
            ax2.set_title('Mahalanobis Distance by Data Point')
            ax2.set_xlabel('Data Point Index')
            ax2.set_ylabel('Mahalanobis Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            print()
        
        # Summary of outlier indices
        if n_multivariate_outliers > 0:
            outlier_indices = scatter_data.index[multivariate_outliers].tolist()
            print("🚨 IDENTIFIED MULTIVARIATE OUTLIERS:")
            print(f"   Row indices: {outlier_indices[:20]}{'...' if len(outlier_indices) > 20 else ''}")
            print()
            
            # Show actual values for first few outliers
            if len(outlier_indices) > 0:
                print("📋 SAMPLE OUTLIER VALUES:")
                sample_outliers = scatter_data.loc[outlier_indices[:5]]
                print(sample_outliers.round(3).to_string())
                print()
        
        print("📊 SCATTER MATRIX INTERPRETATION:")
        print("   • Red X marks → Multivariate outliers")
        print("   • Blue dots → Normal data points")
        print("   • Diagonal histograms → Distribution of each variable")
        print("   • Off-diagonal → Pairwise relationships")
        print("   • Correlation coefficients → Strength of linear relationships")
        print("   • Multivariate outliers may not be univariate outliers")

print("\\n" + "="*50)
print("✅ OUTLIER SCATTER MATRIX VISUALIZATION COMPLETE")
print("="*50)
'''
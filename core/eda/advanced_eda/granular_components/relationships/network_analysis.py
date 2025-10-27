"""Network Analysis Component for Relationship Exploration.

This component creates network graphs showing connections between entities,
variables, or observations based on different relationship metrics.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class NetworkAnalysis:
    """Generate network graphs showing relationships between entities."""
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return {
            "display_name": "Network Relationship Analysis",
            "description": "Create network graphs showing connections between entities, variables, or observations",
            "category": "relationship_exploration",
            "complexity": "intermediate",
            "estimated_runtime": "5-15 seconds",
            "tags": ["network", "graph", "relationships", "connections", "entities"],
            "icon": "🕸️",
            "data_requirements": {
                "min_rows": 10,
                "min_columns": 2,
                "data_types": ["numeric", "categorical"]
            },
            "outputs": ["network_graph", "adjacency_matrix", "network_metrics"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if data is compatible with network analysis."""
        if not data_preview:
            return True
        
        # Need at least 2 columns for relationships
        numeric_cols = data_preview.get("numeric_columns", [])
        categorical_cols = data_preview.get("categorical_columns", [])
        total_cols = len(numeric_cols) + len(categorical_cols)
        
        return total_cols >= 2
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate network analysis code."""
        return '''
print("🕸️ NETWORK RELATIONSHIP ANALYSIS")
print("=" * 50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not available. Using simplified network visualization.")

# Get data info
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
all_cols = numeric_cols + categorical_cols

print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🔢 Numeric columns: {len(numeric_cols)}")
print(f"📝 Categorical columns: {len(categorical_cols)}")

if len(all_cols) < 2:
    print("❌ Need at least 2 columns for network analysis")
else:
    # 1. VARIABLE-VARIABLE CORRELATION NETWORK
    if len(numeric_cols) >= 2:
        print("\\n🔗 VARIABLE CORRELATION NETWORK")
        print("-" * 30)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create adjacency matrix (strong correlations only)
        threshold = 0.5
        adj_matrix = np.abs(corr_matrix) >= threshold
        np.fill_diagonal(adj_matrix.values, False)  # Remove self-connections
        
        # Count connections
        connections = np.sum(adj_matrix.values) // 2  # Undirected graph
        print(f"Variables with |correlation| >= {threshold}: {connections} connections")
        
        if connections > 0:
            # Create network visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Correlation heatmap with threshold
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[0], fmt='.2f')
            axes[0].set_title(f'Variable Correlation Matrix\\n(Network threshold: |r| ≥ {threshold})')
            
            # Plot 2: Network-style visualization
            if NETWORKX_AVAILABLE:
                G = nx.Graph()
                # Add nodes
                for col in numeric_cols:
                    G.add_node(col)
                
                # Add edges for strong correlations
                edge_weights = []
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                        corr_val = corr_matrix.loc[col1, col2]
                        if abs(corr_val) >= threshold:
                            G.add_edge(col1, col2, weight=abs(corr_val), correlation=corr_val)
                            edge_weights.append(abs(corr_val))
                
                if len(G.edges()) > 0:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    
                    # Draw network
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                         node_size=1000, ax=axes[1])
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[1])
                    
                    # Color edges by correlation strength
                    edge_colors = [G[u][v]['correlation'] for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                         edge_cmap=plt.cm.RdBu_r, edge_vmin=-1, edge_vmax=1,
                                         width=[w*3 for w in edge_weights], ax=axes[1])
                    
                    axes[1].set_title(f'Variable Correlation Network\\n({len(G.edges())} connections)')
                else:
                    axes[1].text(0.5, 0.5, f'No correlations ≥ {threshold}\\nfound between variables', 
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Variable Network (No Strong Correlations)')
            else:
                # Fallback visualization without NetworkX
                axes[1].text(0.5, 0.5, 'NetworkX required for\\nnetwork visualization', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Network Visualization (NetworkX Required)')
            
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
            
            # Print strong correlations
            print("\\n🔗 Strong Variable Connections:")
            strong_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        strong_pairs.append((col1, col2, corr_val))
            
            strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in strong_pairs:
                direction = "+" if corr > 0 else "-"
                print(f"   {col1} ↔ {col2}: {corr:.3f} ({direction})")
        else:
            print(f"   No correlations ≥ {threshold} found between variables")
    
    # 2. OBSERVATION-OBSERVATION SIMILARITY NETWORK
    if len(numeric_cols) >= 2 and df.shape[0] <= 100:  # Limit for performance
        print("\\n👥 OBSERVATION SIMILARITY NETWORK")
        print("-" * 35)
        
        # Sample data for large datasets
        sample_size = min(50, df.shape[0])
        df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)
        
        # Calculate pairwise distances
        distances = pdist(df_sample.fillna(df_sample.mean()), metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Convert to similarity (inverse distance)
        similarity_matrix = 1 / (1 + distance_matrix)
        np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity
        
        # Create connections for most similar observations
        similarity_threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 80)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Similarity matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(similarity_matrix, cmap='viridis', square=True, cbar_kws={"shrink": .8})
        plt.title(f'Observation Similarity Matrix\\n(Sample of {sample_size} observations)')
        
        # Plot 2: Network visualization
        plt.subplot(1, 2, 2)
        if NETWORKX_AVAILABLE:
            G_obs = nx.Graph()
            
            # Add nodes (observations)
            for i in range(sample_size):
                G_obs.add_node(i, label=f'Obs_{i}')
            
            # Add edges for similar observations
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    if similarity_matrix[i, j] >= similarity_threshold:
                        G_obs.add_edge(i, j, weight=similarity_matrix[i, j])
            
            if len(G_obs.edges()) > 0:
                pos = nx.spring_layout(G_obs, k=2, iterations=50)
                
                # Node colors based on a clustering-like measure (degree)
                node_colors = [G_obs.degree(node) for node in G_obs.nodes()]
                
                nx.draw_networkx_nodes(G_obs, pos, node_color=node_colors, 
                                     cmap=plt.cm.Set3, node_size=300)
                nx.draw_networkx_edges(G_obs, pos, alpha=0.6, width=0.5)
                
                plt.title(f'Observation Similarity Network\\n({len(G_obs.edges())} connections)')
            else:
                plt.text(0.5, 0.5, f'No similarities ≥ {similarity_threshold:.3f}\\nfound', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Observation Network (No Strong Similarities)')
        else:
            plt.text(0.5, 0.5, 'NetworkX required for\\nnetwork visualization', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Network Visualization (NetworkX Required)')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"Similarity threshold (80th percentile): {similarity_threshold:.3f}")
        if NETWORKX_AVAILABLE and len(G_obs.edges()) > 0:
            print(f"Network has {len(G_obs.nodes())} nodes and {len(G_obs.edges())} edges")
            print(f"Average degree: {2*len(G_obs.edges())/len(G_obs.nodes()):.1f}")
    
    # 3. CATEGORICAL VARIABLE CO-OCCURRENCE NETWORK
    if len(categorical_cols) >= 2:
        print("\\n🏷️ CATEGORICAL CO-OCCURRENCE NETWORK")
        print("-" * 40)
        
        # Create co-occurrence matrix for categorical values
        cat_values = []
        cat_columns = []
        
        # Get unique values from each categorical column
        for col in categorical_cols[:3]:  # Limit to first 3 for performance
            unique_vals = df[col].dropna().unique()[:5]  # Top 5 values per column
            for val in unique_vals:
                cat_values.append(f"{col}_{val}")
                cat_columns.append(col)
        
        if len(cat_values) >= 2:
            # Calculate co-occurrence (how often values appear together)
            cooccur_matrix = np.zeros((len(cat_values), len(cat_values)))
            
            for i, val1 in enumerate(cat_values):
                col1, actual_val1 = val1.split('_', 1)
                for j, val2 in enumerate(cat_values):
                    if i != j:
                        col2, actual_val2 = val2.split('_', 1)
                        if col1 != col2:  # Different columns
                            # Count co-occurrences
                            mask1 = df[col1].astype(str) == actual_val1
                            mask2 = df[col2].astype(str) == actual_val2
                            cooccur_count = (mask1 & mask2).sum()
                            cooccur_matrix[i, j] = cooccur_count
            
            # Normalize by total rows
            cooccur_matrix = cooccur_matrix / len(df)
            
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Co-occurrence heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(cooccur_matrix, 
                       xticklabels=cat_values, yticklabels=cat_values,
                       annot=True, fmt='.3f', cmap='Blues', square=True)
            plt.title('Categorical Co-occurrence Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Plot 2: Network visualization
            plt.subplot(1, 2, 2)
            if NETWORKX_AVAILABLE:
                G_cat = nx.Graph()
                
                # Add nodes
                for val in cat_values:
                    G_cat.add_node(val)
                
                # Add edges for significant co-occurrences
                cooccur_threshold = 0.05  # 5% co-occurrence threshold
                for i, val1 in enumerate(cat_values):
                    for j, val2 in enumerate(cat_values[i+1:], i+1):
                        if cooccur_matrix[i, j] >= cooccur_threshold:
                            G_cat.add_edge(val1, val2, weight=cooccur_matrix[i, j])
                
                if len(G_cat.edges()) > 0:
                    pos = nx.spring_layout(G_cat, k=3, iterations=50)
                    
                    # Color nodes by original column
                    node_colors = []
                    for node in G_cat.nodes():
                        col_name = node.split('_', 1)[0]
                        node_colors.append(hash(col_name) % 10)
                    
                    nx.draw_networkx_nodes(G_cat, pos, node_color=node_colors, 
                                         cmap=plt.cm.Set3, node_size=500)
                    nx.draw_networkx_labels(G_cat, pos, font_size=6)
                    nx.draw_networkx_edges(G_cat, pos, alpha=0.6)
                    
                    plt.title(f'Categorical Co-occurrence Network\\n({len(G_cat.edges())} connections)')
                else:
                    plt.text(0.5, 0.5, f'No co-occurrences ≥ {cooccur_threshold}\\nfound', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Categorical Network (No Strong Co-occurrences)')
            else:
                plt.text(0.5, 0.5, 'NetworkX required for\\nnetwork visualization', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Network Visualization (NetworkX Required)')
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    # 4. NETWORK SUMMARY STATISTICS
    print("\\n📊 NETWORK ANALYSIS SUMMARY")
    print("-" * 30)
    
    if len(numeric_cols) >= 2:
        # Variable network metrics
        strong_correlations = 0
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if abs(corr_matrix.loc[col1, col2]) >= 0.5:
                    strong_correlations += 1
        
        print(f"🔗 Variable Network:")
        print(f"   • Strong correlations (|r| ≥ 0.5): {strong_correlations}")
        print(f"   • Network density: {strong_correlations / (len(numeric_cols)*(len(numeric_cols)-1)/2):.3f}")
    
    if len(categorical_cols) >= 2:
        print(f"🏷️ Categorical Network:")
        print(f"   • Categorical columns analyzed: {min(3, len(categorical_cols))}")
        print(f"   • Unique value combinations: {len(cat_values) if 'cat_values' in locals() else 'N/A'}")
    
    print("\\n💡 INSIGHTS & RECOMMENDATIONS")
    print("-" * 35)
    
    if len(numeric_cols) >= 2:
        if strong_correlations > len(numeric_cols) * 0.5:
            print("✅ High variable connectivity detected")
            print("   → Consider dimensionality reduction (PCA)")
            print("   → Check for multicollinearity issues")
        elif strong_correlations == 0:
            print("⚠️ Low variable connectivity")
            print("   → Variables may be independent")
            print("   → Consider different analysis approaches")
        else:
            print("✅ Moderate variable connectivity")
            print("   → Good balance of relationships")
    
    if len(all_cols) >= 3:
        print("\\n🔍 Next steps:")
        print("   • Try different correlation methods (Spearman, Kendall)")
        print("   • Analyze subgroups or clusters separately") 
        print("   • Consider temporal relationships if time data exists")
        print("   • Use network centrality measures for feature selection")

print("\\n✅ Network relationship analysis complete!")
'''


def get_component():
    """Return the analysis component."""
    return NetworkAnalysis
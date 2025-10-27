"""Entity Relationship Network Component.

This component creates network graphs specifically for entity relationships,
identifying and visualizing connections between different entities in the dataset.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class EntityRelationshipNetwork:
    """Create network graphs showing relationships between entities."""
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata."""
        return {
            "display_name": "Entity Relationship Network",
            "description": "Analyze and visualize relationships between different entities (customers, products, locations, etc.)",
            "category": "relationship_exploration",
            "complexity": "advanced",
            "estimated_runtime": "10-30 seconds",
            "tags": ["entities", "relationships", "network", "graph", "connections", "business"],
            "icon": "🌐",
            "data_requirements": {
                "min_rows": 20,
                "min_columns": 2,
                "data_types": ["categorical", "numeric"]
            },
            "outputs": ["entity_network", "relationship_metrics", "centrality_analysis"]
        }
    
    def validate_data_compatibility(self, data_preview: Dict[str, Any]) -> bool:
        """Check if data is compatible with entity relationship analysis."""
        if not data_preview:
            return True
        
        # Need at least 2 categorical columns for entity relationships
        categorical_cols = data_preview.get("categorical_columns", [])
        return len(categorical_cols) >= 2
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate entity relationship network analysis code."""
        return '''
print("🌐 ENTITY RELATIONSHIP NETWORK ANALYSIS")
print("=" * 50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX not available. Using simplified analysis.")

# Get data info
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🏷️ Categorical columns: {len(categorical_cols)}")
print(f"🔢 Numeric columns: {len(numeric_cols)}")

if len(categorical_cols) < 2:
    print("❌ Need at least 2 categorical columns for entity relationship analysis")
else:
    # Identify potential entity columns
    entity_candidates = {}
    for col in categorical_cols:
        unique_count = df[col].nunique()
        total_rows = len(df)
        uniqueness_ratio = unique_count / total_rows
        
        # Classify columns by uniqueness
        if uniqueness_ratio > 0.8:
            entity_type = "High Unique (IDs, Names)"
        elif uniqueness_ratio > 0.5:
            entity_type = "Medium Unique (Categories)"
        elif uniqueness_ratio > 0.1:
            entity_type = "Low Unique (Groups)"
        else:
            entity_type = "Very Low Unique (Labels)"
        
        entity_candidates[col] = {
            'unique_count': unique_count,
            'uniqueness_ratio': uniqueness_ratio,
            'entity_type': entity_type
        }
    
    print("\\n🔍 ENTITY COLUMN ANALYSIS")
    print("-" * 30)
    for col, info in entity_candidates.items():
        print(f"{col}:")
        print(f"   • Unique values: {info['unique_count']}")
        print(f"   • Uniqueness ratio: {info['uniqueness_ratio']:.3f}")
        print(f"   • Likely type: {info['entity_type']}")
    
    # Select best entity columns for network analysis
    entity_cols = []
    for col, info in entity_candidates.items():
        # Select columns with reasonable uniqueness for network analysis
        if 0.05 <= info['uniqueness_ratio'] <= 0.8 and info['unique_count'] <= 50:
            entity_cols.append(col)
    
    # If no suitable columns, use top categorical columns
    if len(entity_cols) < 2:
        entity_cols = categorical_cols[:3]
    
    print(f"\\n🎯 Selected entity columns for network: {entity_cols[:3]}")
    
    # 1. BIPARTITE ENTITY NETWORKS
    if len(entity_cols) >= 2:
        print("\\n🔗 BIPARTITE ENTITY RELATIONSHIPS")
        print("-" * 35)
        
        # Analyze first two entity columns
        col1, col2 = entity_cols[0], entity_cols[1]
        
        # Create relationship counts
        relationship_counts = df.groupby([col1, col2]).size().reset_index(name='count')
        relationship_counts = relationship_counts[relationship_counts['count'] >= 2]  # Filter weak connections
        
        print(f"Analyzing relationships between {col1} and {col2}")
        print(f"Total unique relationships: {len(relationship_counts)}")
        print(f"Relationships with 2+ connections: {len(relationship_counts)}")
        
        if len(relationship_counts) > 0:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Relationship strength heatmap
            pivot_table = df.pivot_table(values=numeric_cols[0] if numeric_cols else None, 
                                       index=col1, columns=col2, 
                                       aggfunc='count' if not numeric_cols else 'mean',
                                       fill_value=0)
            
            # Limit size for visualization
            if pivot_table.shape[0] > 15:
                top_entities1 = df[col1].value_counts().head(15).index
                pivot_table = pivot_table.loc[top_entities1]
            if pivot_table.shape[1] > 15:
                top_entities2 = df[col2].value_counts().head(15).index
                pivot_table = pivot_table[top_entities2]
            
            sns.heatmap(pivot_table, cmap='Blues', annot=False, 
                       ax=axes[0,0], cbar_kws={"shrink": .8})
            axes[0,0].set_title(f'{col1} vs {col2}\\nRelationship Matrix')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].tick_params(axis='y', rotation=0)
            
            # Plot 2: Network visualization
            if NETWORKX_AVAILABLE:
                G = nx.Graph()
                
                # Add nodes with types
                entities1 = relationship_counts[col1].unique()[:20]  # Limit for performance
                entities2 = relationship_counts[col2].unique()[:20]
                
                for entity in entities1:
                    G.add_node(f"{col1}_{entity}", type=col1, entity=entity)
                for entity in entities2:
                    G.add_node(f"{col2}_{entity}", type=col2, entity=entity)
                
                # Add edges
                edge_weights = []
                for _, row in relationship_counts.iterrows():
                    entity1 = f"{col1}_{row[col1]}"
                    entity2 = f"{col2}_{row[col2]}"
                    weight = row['count']
                    
                    if entity1 in G.nodes() and entity2 in G.nodes():
                        G.add_edge(entity1, entity2, weight=weight)
                        edge_weights.append(weight)
                
                if len(G.edges()) > 0:
                    # Create layout
                    pos = nx.spring_layout(G, k=2, iterations=50)
                    
                    # Separate node types for coloring
                    nodes1 = [n for n in G.nodes() if n.startswith(f"{col1}_")]
                    nodes2 = [n for n in G.nodes() if n.startswith(f"{col2}_")]
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes1, 
                                         node_color='lightblue', node_size=300, 
                                         label=col1, ax=axes[0,1])
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes2, 
                                         node_color='lightcoral', node_size=300, 
                                         label=col2, ax=axes[0,1])
                    
                    # Draw edges with thickness based on weight
                    edge_widths = [(w/max(edge_weights))*3 for w in edge_weights]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=axes[0,1])
                    
                    # Add labels (limited)
                    limited_labels = {}
                    for node in list(G.nodes())[:10]:  # Show only first 10 labels
                        entity_name = node.split('_', 1)[1][:8]  # Truncate long names
                        limited_labels[node] = entity_name
                    nx.draw_networkx_labels(G, pos, limited_labels, font_size=6, ax=axes[0,1])
                    
                    axes[0,1].set_title(f'Entity Relationship Network\\n{len(G.nodes())} nodes, {len(G.edges())} edges')
                    axes[0,1].legend()
                else:
                    axes[0,1].text(0.5, 0.5, 'No significant relationships\\nfor network visualization', 
                                  ha='center', va='center', transform=axes[0,1].transAxes)
                    axes[0,1].set_title('Entity Network (No Strong Relationships)')
            else:
                axes[0,1].text(0.5, 0.5, 'NetworkX required for\\nnetwork visualization', 
                              ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_title('Network Visualization (NetworkX Required)')
            
            axes[0,1].axis('off')
            
            # Plot 3: Top relationships
            top_relationships = relationship_counts.nlargest(15, 'count')
            
            y_pos = range(len(top_relationships))
            axes[1,0].barh(y_pos, top_relationships['count'])
            axes[1,0].set_yticks(y_pos)
            axes[1,0].set_yticklabels([f"{row[col1]} → {row[col2]}" 
                                      for _, row in top_relationships.iterrows()], 
                                     fontsize=8)
            axes[1,0].set_title('Top Entity Relationships')
            axes[1,0].set_xlabel('Connection Strength')
            
            # Plot 4: Entity degree distribution
            entity_degrees = defaultdict(int)
            for _, row in relationship_counts.iterrows():
                entity_degrees[f"{col1}_{row[col1]}"] += row['count']
                entity_degrees[f"{col2}_{row[col2]}"] += row['count']
            
            degrees = list(entity_degrees.values())
            axes[1,1].hist(degrees, bins=min(20, len(degrees)//2), alpha=0.7, color='skyblue')
            axes[1,1].set_title('Entity Degree Distribution')
            axes[1,1].set_xlabel('Total Connections')
            axes[1,1].set_ylabel('Number of Entities')
            
            plt.tight_layout()
            plt.show()
            
            # Network metrics
            if NETWORKX_AVAILABLE and len(G.edges()) > 0:
                print("\\n📊 NETWORK METRICS")
                print("-" * 20)
                print(f"Nodes: {len(G.nodes())}")
                print(f"Edges: {len(G.edges())}")
                print(f"Density: {nx.density(G):.4f}")
                print(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
                
                # Central entities
                centrality = nx.degree_centrality(G)
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print("\\n🌟 Most Central Entities:")
                for entity, centrality_score in top_central:
                    clean_name = entity.split('_', 1)[1]
                    entity_type = entity.split('_', 1)[0]
                    print(f"   • {clean_name} ({entity_type}): {centrality_score:.3f}")
    
    # 2. MULTI-ENTITY RELATIONSHIPS
    if len(entity_cols) >= 3:
        print("\\n🔗 MULTI-ENTITY RELATIONSHIPS")
        print("-" * 32)
        
        # Analyze three-way relationships
        col1, col2, col3 = entity_cols[0], entity_cols[1], entity_cols[2]
        
        # Find common entities across three dimensions
        three_way_counts = df.groupby([col1, col2, col3]).size().reset_index(name='count')
        three_way_counts = three_way_counts[three_way_counts['count'] >= 2]
        
        if len(three_way_counts) > 0:
            print(f"Three-way relationships found: {len(three_way_counts)}")
            
            # Show top relationships
            top_three_way = three_way_counts.nlargest(10, 'count')
            print("\\n🔝 Top Three-way Relationships:")
            for _, row in top_three_way.iterrows():
                print(f"   • {row[col1]} + {row[col2]} + {row[col3]}: {row['count']} occurrences")
        else:
            print("No significant three-way relationships found")
    
    # 3. ENTITY ATTRIBUTE ANALYSIS
    if numeric_cols:
        print("\\n💰 ENTITY ATTRIBUTE ANALYSIS")
        print("-" * 32)
        
        # Analyze numeric attributes by entity
        primary_entity = entity_cols[0]
        primary_metric = numeric_cols[0]
        
        entity_metrics = df.groupby(primary_entity)[primary_metric].agg(['count', 'mean', 'sum', 'std']).fillna(0)
        entity_metrics = entity_metrics.sort_values('sum', ascending=False)
        
        print(f"Analyzing {primary_metric} by {primary_entity}")
        print(f"Top entities by total {primary_metric}:")
        
        top_entities = entity_metrics.head(10)
        for entity, metrics in top_entities.iterrows():
            print(f"   • {entity}: {metrics['sum']:.2f} total, {metrics['mean']:.2f} avg ({metrics['count']} records)")
        
        # Visualize entity performance
        if len(entity_metrics) <= 20:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            top_10 = entity_metrics.head(10)
            plt.barh(range(len(top_10)), top_10['sum'])
            plt.yticks(range(len(top_10)), top_10.index)
            plt.title(f'Top 10 {primary_entity} by Total {primary_metric}')
            plt.xlabel(f'Total {primary_metric}')
            
            plt.subplot(1, 2, 2)
            plt.scatter(entity_metrics['count'], entity_metrics['mean'], 
                       s=entity_metrics['sum']/entity_metrics['sum'].max()*200, alpha=0.6)
            plt.xlabel(f'Number of Records')
            plt.ylabel(f'Average {primary_metric}')
            plt.title(f'{primary_entity} Performance\\n(Bubble size = Total {primary_metric})')
            
            plt.tight_layout()
            plt.show()
    
    # 4. RECOMMENDATIONS
    print("\\n💡 ENTITY RELATIONSHIP INSIGHTS")
    print("-" * 35)
    
    # Connection insights
    if len(relationship_counts) > 0:
        max_connections = relationship_counts['count'].max()
        avg_connections = relationship_counts['count'].mean()
        
        if max_connections > avg_connections * 3:
            print("✅ Hub entities detected (some entities are much more connected)")
            print("   → Consider hub-based analysis or clustering")
        
        if len(relationship_counts) > len(df) * 0.1:
            print("✅ Dense entity network")
            print("   → Good for community detection algorithms")
        else:
            print("⚠️ Sparse entity network")
            print("   → Focus on strongest relationships")
    
    # Entity type insights
    high_unique_cols = [col for col, info in entity_candidates.items() 
                       if info['uniqueness_ratio'] > 0.8]
    if high_unique_cols:
        print(f"\\n🆔 Potential ID columns detected: {high_unique_cols}")
        print("   → Use for linking records or deduplication")
    
    group_cols = [col for col, info in entity_candidates.items() 
                 if 0.1 <= info['uniqueness_ratio'] <= 0.5]
    if group_cols:
        print(f"\\n👥 Grouping columns detected: {group_cols}")
        print("   → Good for segmentation analysis")
    
    print("\\n🔍 Next steps:")
    print("   • Apply community detection algorithms")
    print("   • Analyze temporal changes in relationships") 
    print("   • Create entity embeddings for similarity analysis")
    print("   • Build recommendation systems based on entity connections")

print("\\n✅ Entity relationship network analysis complete!")
'''


def get_component():
    """Return the analysis component."""
    return EntityRelationshipNetwork
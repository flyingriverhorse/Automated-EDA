"""Dataset Shape Analysis Component.

Provides detailed analysis of dataset dimensions including rows, columns,
memory usage, and basic structural information.
"""


class DatasetShapeAnalysis:
    """Analyze dataset dimensions and basic structure."""
    
    @staticmethod
    def get_metadata():
        return {
            "name": "dataset_shape_analysis",
            "display_name": "Dataset Shape Analysis", 
            "description": "Inspect dataset dimensions (rows, columns) and basic structure",
            "category": "data_quality",
            "complexity": "basic",
            "tags": ["shape", "dimensions", "structure", "overview"],
            "estimated_runtime": "< 1 second",
            "icon": "📊"
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview=None):
        """Check if analysis can be performed on the data."""
        return True  # Works with any dataset
    
    @staticmethod
    def generate_code(data_preview=None):
        """Generate code for dataset shape analysis."""
        
        return '''
# ===== DATASET SHAPE ANALYSIS =====

import pandas as pd
import numpy as np

print("="*60)
print("📊 DATASET SHAPE AND STRUCTURE ANALYSIS")  
print("="*60)

# Basic shape information
print(f"\\n📏 Dataset Dimensions:")
print(f"   • Rows: {df.shape[0]:,}")
print(f"   • Columns: {df.shape[1]:,}")
print(f"   • Total cells: {df.shape[0] * df.shape[1]:,}")

# Memory usage analysis
print(f"\\n💾 Memory Usage:")
memory_usage = df.memory_usage(deep=True)
total_memory = memory_usage.sum()

if total_memory > 1024**3:  # GB
    print(f"   • Total memory: {total_memory / (1024**3):.2f} GB")
elif total_memory > 1024**2:  # MB  
    print(f"   • Total memory: {total_memory / (1024**2):.2f} MB")
else:  # KB
    print(f"   • Total memory: {total_memory / 1024:.2f} KB")

print(f"   • Average memory per row: {total_memory / df.shape[0]:.2f} bytes")

# Column information summary
print(f"\\n📋 Column Overview:")
print(f"   • Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"   • Text/Object columns: {len(df.select_dtypes(include=['object']).columns)}")
print(f"   • DateTime columns: {len(df.select_dtypes(include=['datetime64']).columns)}")
print(f"   • Boolean columns: {len(df.select_dtypes(include=['bool']).columns)}")

# Data sparsity
non_null_count = df.count().sum()
total_cells = df.shape[0] * df.shape[1]
completeness = (non_null_count / total_cells) * 100

print(f"\\n✅ Data Completeness:")
print(f"   • Non-null cells: {non_null_count:,} ({completeness:.1f}%)")
print(f"   • Null cells: {total_cells - non_null_count:,} ({100-completeness:.1f}%)")

# Quick preview of column names
print(f"\\n📝 Column Names Preview:")
if df.shape[1] <= 10:
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
else:
    for i, col in enumerate(df.columns[:8], 1):  # Show first 8 columns (increased from 5)
        print(f"   {i:2d}. {col}")
    print(f"   ... ({df.shape[1] - 10} more columns)")
    for i, col in enumerate(df.columns[-5:], df.shape[1]-4):
        print(f"   {i:2d}. {col}")

print("\\n" + "="*60)
print("✅ Dataset shape analysis complete!")
print("="*60)
'''


def get_component():
    """Return the analysis component."""
    return DatasetShapeAnalysis
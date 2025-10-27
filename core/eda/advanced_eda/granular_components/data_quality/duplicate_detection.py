"""Duplicate Detection Analysis Component.

Focused analysis component for detecting and analyzing duplicate records in the dataset.
"""
from typing import Dict, Any


class DuplicateDetectionAnalysis:
    """Focused component for duplicate detection analysis"""
    
    def __init__(self):
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this analysis component."""
        return {
            "name": "duplicate_detection",
            "display_name": "Duplicate Detection",
            "description": "Detect and analyze duplicate records in the dataset",
            "category": "data_quality",
            "complexity": "basic",
            "required_data_types": ["any"],
            "estimated_runtime": "5-10 seconds",
            "icon": "copy",
            "tags": ["duplicates", "data-quality", "validation"]
        }
    
    @staticmethod
    def validate_data_compatibility(data_preview: Dict[str, Any] = None) -> bool:
        """This analysis works with any dataset"""
        return True
    
    def generate_code(self, data_preview: Dict[str, Any] = None) -> str:
        """Generate focused duplicate detection analysis code"""
        return '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== DUPLICATE DETECTION ANALYSIS ===")
print("Focused analysis of duplicate records in the dataset")
print()

# Basic duplicate analysis
total_rows = len(df)
duplicate_rows = df.duplicated().sum()
duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0

print(f"📊 DUPLICATE SUMMARY")
print(f"   Total rows: {total_rows:,}")
print(f"   Duplicate rows: {duplicate_rows:,}")
print(f"   Duplicate percentage: {duplicate_percentage:.2f}%")
print()

if duplicate_rows > 0:
    print("🔍 DUPLICATE DETAILS")
    print("   First 5 duplicate rows:")
    duplicate_df = df[df.duplicated()]
    print(duplicate_df.head())
    print()
    
    # Show which rows are duplicates of which
    print("   Duplicate row indices:")
    duplicate_indices = df[df.duplicated()].index.tolist()
    print(f"   Indices: {duplicate_indices[:20]}{'...' if len(duplicate_indices) > 20 else ''}")
    print()
    
    # Keep options analysis
    print("📝 DUPLICATE HANDLING OPTIONS")
    print("   • df.drop_duplicates() - Remove all duplicates")
    print("   • df.drop_duplicates(keep='first') - Keep first occurrence")
    print("   • df.drop_duplicates(keep='last') - Keep last occurrence")
    print("   • df.drop_duplicates(keep=False) - Remove all duplicate occurrences")
    print()

# Semantic duplicate analysis (excluding ID columns)
id_keywords = ['id', 'index', 'key', 'uuid', 'guid', 'pk', 'primary']
non_id_cols = [col for col in df.columns if not any(keyword in col.lower() for keyword in id_keywords)]

if len(non_id_cols) < len(df.columns):
    semantic_dups = df[non_id_cols].duplicated().sum()
    semantic_percentage = (semantic_dups / total_rows * 100) if total_rows > 0 else 0
    
    print("🎯 SEMANTIC DUPLICATE ANALYSIS")
    print(f"   (Excluding potential ID columns: {[col for col in df.columns if col not in non_id_cols]})")
    print(f"   Semantic duplicates: {semantic_dups:,} ({semantic_percentage:.2f}%)")
    print()

# Column-wise duplicate analysis
print("📋 COLUMN-WISE DUPLICATE ANALYSIS")
for col in df.columns:
    col_total = len(df[col])
    col_unique = df[col].nunique()
    col_duplicates = col_total - col_unique
    col_dup_percent = (col_duplicates / col_total * 100) if col_total > 0 else 0
    
    print(f"   {col}: {col_duplicates:,} duplicates ({col_dup_percent:.1f}%)")

if duplicate_rows == 0:
    print("✅ NO DUPLICATES FOUND")
    print("   Your dataset appears to be free of duplicate records!")
else:
    print()
    print("⚠️  RECOMMENDATION")
    print("   Consider reviewing and handling duplicate records before analysis.")
    print("   Use appropriate deduplication strategy based on your use case.")

print("\\n" + "="*50)
'''
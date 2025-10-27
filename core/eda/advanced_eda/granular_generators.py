"""Granular Analysis Code Generators.

Replaces the old analysis_factory and analysis_base system with a simple,
direct approach using the granular components.
"""
from typing import Dict, List, Any, Optional, Tuple
import inspect
import logging
import re

from .granular_components import get_all_granular_components, get_components_by_category

logger = logging.getLogger(__name__)

GEOSPATIAL_LAT_HINTS = ("latitude", "lat", "lat_deg", "latitud")
GEOSPATIAL_LON_HINTS = ("longitude", "lon", "lng", "long", "lon_deg")


def _columns_contain_hint(columns: List[str], hints: Tuple[str, ...]) -> bool:
    if not columns:
        return False

    for raw_col in columns:
        column = str(raw_col).strip().lower()
        for hint in hints:
            pattern = rf"(?:^|[ _\-]){re.escape(hint)}(?:$|[ _\-])"
            if column == hint or re.search(pattern, column):
                return True
    return False


class GranularAnalysisCodeGenerators:
    """Simple code generator system for granular analysis components."""
    
    def __init__(self):
        self.components = get_all_granular_components()
        self.categories = get_components_by_category()
        
        # Create display names mapping
        self.display_names = {}
        for comp_id, comp_info in self.components.items():
            metadata = comp_info.get('metadata', {})
            display_name = metadata.get('display_name', comp_id.replace('_', ' ').title())
            self.display_names[comp_id] = display_name
    
    def get_available_analyses(self) -> List[str]:
        """Get list of all available analysis IDs."""
        return list(self.components.keys())
    
    def get_analysis_metadata(self, analysis_id: str) -> Dict[str, Any]:
        """Get metadata for a specific analysis."""
        if analysis_id not in self.components:
            return {}
        return self.components[analysis_id].get('metadata', {})
    
    def get_analysis_display_name(self, analysis_id: str) -> str:
        """Get display name for analysis."""
        return self.display_names.get(analysis_id, analysis_id.replace('_', ' ').title())
    
    def validate_analysis_compatibility(self, analysis_id: str, data_preview: Dict[str, Any] = None) -> bool:
        """Check if analysis is compatible with the data."""
        if analysis_id not in self.components:
            return False
        
        component_class = self.components[analysis_id]['component']

        try:
            validation_attr = inspect.getattr_static(component_class, 'validate_data_compatibility')
        except AttributeError:
            validation_attr = None

        try:
            if isinstance(validation_attr, staticmethod):
                return validation_attr.__func__(data_preview)
            if isinstance(validation_attr, classmethod):
                return validation_attr.__func__(component_class, data_preview)
            if callable(validation_attr):
                # Regular function defined on class; bind via instance to supply self
                instance = component_class()
                bound_method = getattr(instance, 'validate_data_compatibility', None)
                if callable(bound_method):
                    return bound_method(data_preview)

            # Fallback: instantiate and call if available
            instance = component_class()
            bound_method = getattr(instance, 'validate_data_compatibility', None)
            if callable(bound_method):
                return bound_method(data_preview)
        except Exception as e:
            logger.warning(f"Error validating compatibility for {analysis_id}: {e}")
            return True  # Default to compatible if validation fails
        
        return True  # Default to compatible if no validation method
    
    def generate_analysis_code(self, analysis_id: str, data_preview: Dict[str, Any] = None) -> str:
        """Generate code for a specific analysis."""
        if analysis_id not in self.components:
            return f"# Error: Unknown analysis '{analysis_id}'"
        
        component_class = self.components[analysis_id]['component']
        
        try:
            # Create an instance of the component class
            component_instance = component_class()
            if hasattr(component_instance, 'generate_code'):
                return component_instance.generate_code(data_preview)
            else:
                return f"# Error: Component '{analysis_id}' does not have generate_code method"
        except Exception as e:
            logger.error(f"Error generating code for {analysis_id}: {e}")
            return f"# Error generating code for {analysis_id}: {str(e)}"
    
    def generate_multiple_analyses_code(self, analysis_ids: List[str], data_preview: Dict[str, Any] = None) -> str:
        """Generate code for multiple analyses."""
        if not analysis_ids:
            return "# No analyses selected"
        
        code_parts = []
        
        # Add header
        code_parts.append("""
# ===== GRANULAR EDA ANALYSIS =====
# Generated code for selected analysis components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ensure df is available
if 'df' not in globals():
    raise NameError("DataFrame 'df' is not defined. Please load your data first.")

print("🚀 Starting Granular EDA Analysis...")
print("Original dataset shape:", df.shape)""")

        # Add column filtering if specific columns are selected
        if data_preview and 'columns' in data_preview:
            selected_columns = data_preview['columns']
            # Check if we have information about all columns to determine if filtering is needed
            all_columns = data_preview.get('all_columns') or data_preview.get('original_columns')
            
            # Only add filtering code if we know we're working with a subset
            if selected_columns and all_columns and len(selected_columns) < len(all_columns):
                # Only filter if we have a subset of columns selected
                code_parts.append(f'''
# Column filtering based on user selection
selected_columns = {selected_columns}
print(f"🎯 Filtering to {{len(selected_columns)}} selected columns")
print(f"   Selected: {{', '.join(selected_columns)}}")

# Check if all selected columns exist in the dataframe
available_columns = [col for col in selected_columns if col in df.columns]
if len(available_columns) != len(selected_columns):
    missing_columns = [col for col in selected_columns if col not in df.columns]
    print(f"⚠️  Warning: {{len(missing_columns)}} selected columns not found: {{missing_columns}}")

if available_columns:
    # Filter DataFrame to selected columns only
    df_filtered = df[available_columns].copy()
    df = df_filtered  # Use filtered dataframe for analysis
    print("Filtered dataset shape:", df.shape)
else:
    print("❌ No valid columns selected, using all columns")
''')
            elif selected_columns:
                # We have selected columns but don't know if it's a subset - add a check
                code_parts.append(f'''
# Column selection check
selected_columns = {selected_columns}
available_columns = [col for col in selected_columns if col in df.columns]

if len(available_columns) < df.shape[1] and available_columns:
    print(f"🎯 Filtering to {{len(available_columns)}} selected columns")
    print(f"   Selected: {{', '.join(available_columns)}}")
    df_filtered = df[available_columns].copy()
    df = df_filtered  # Use filtered dataframe for analysis
    print("Filtered dataset shape:", df.shape)
else:
    print("📊 Analyzing all columns in dataset")
''')

        code_parts.append('print("="*80)')
        
        # Add each analysis
        for i, analysis_id in enumerate(analysis_ids, 1):
            if analysis_id in self.components:
                display_name = self.get_analysis_display_name(analysis_id)
                
                # Use regular string formatting, not f-strings
                header = '''
print("\\n" + "="*80)
print("📊 ANALYSIS {}/{}: {}")
print("="*80)
'''.format(i, len(analysis_ids), display_name.upper())
                code_parts.append(header)
                
                analysis_code = self.generate_analysis_code(analysis_id, data_preview)
                code_parts.append(analysis_code)
                
                footer = '''
print("\\n✅ {} completed!")
'''.format(display_name)
                code_parts.append(footer)
        
        # Add footer
        code_parts.append("""
print("\\n" + "="*80)
print("🎉 All selected analyses completed successfully!")
print("="*80)
""")
        
        return "\n".join(code_parts)
    
    def get_grouped_analyses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get analyses grouped by category."""
        grouped = {}
        
        for category, components in self.categories.items():
            grouped[category] = []
            for comp_info in components:
                comp_id = comp_info['id']
                metadata = comp_info['metadata']
                
                grouped[category].append({
                    'id': comp_id,
                    'name': self.get_analysis_display_name(comp_id),
                    'description': metadata.get('description', ''),
                    'complexity': metadata.get('complexity', 'intermediate'),
                    'estimated_runtime': metadata.get('estimated_runtime', '1-5 seconds'),
                    'tags': metadata.get('tags', []),
                    'icon': metadata.get('icon', '📊')
                })
        
        return grouped
    
    def get_analysis_recommendations(self, data_preview: Dict[str, Any]) -> List[str]:
        """Get recommended analyses based on data characteristics."""
        if not data_preview:
            return list(self.components.keys())[:10]  # Return first 10 if no preview
        
        recommendations = []
        
        # Always recommend basic data quality checks
        recommendations.extend([
            'dataset_shape_analysis',
            'missing_value_analysis',
            'duplicate_detection'
        ])
        
        # Numeric data recommendations
        numeric_cols = data_preview.get('numeric_columns', [])
        if len(numeric_cols) > 0:
            recommendations.extend([
                'summary_statistics',
                'distribution_plots'
            ])
            
            if len(numeric_cols) >= 2:
                recommendations.append('correlation_analysis')
                
            if len(numeric_cols) >= 3:
                recommendations.extend([
                    'iqr_outlier_detection',
                    'pca_dimensionality_reduction'
                ])
        
        # Categorical data recommendations
        categorical_cols = data_preview.get('categorical_columns', []) + data_preview.get('object_columns', [])
        if len(categorical_cols) > 0:
            recommendations.extend([
                'categorical_frequency_analysis',
                'categorical_visualization'
            ])
            
            if len(categorical_cols) >= 2:
                recommendations.append('cross_tabulation_analysis')
        
        # Time series recommendations
        datetime_cols = data_preview.get('datetime_columns', [])
        if len(datetime_cols) > 0:
            recommendations.extend([
                'temporal_trend_analysis',
                'seasonality_detection'
            ])
        
        columns = data_preview.get('columns', [])
        if columns:
            has_latitude = _columns_contain_hint(columns, GEOSPATIAL_LAT_HINTS)
            has_longitude = _columns_contain_hint(columns, GEOSPATIAL_LON_HINTS)

            if has_latitude and has_longitude:
                recommendations.extend([
                    'coordinate_system_projection_check',
                    'spatial_data_quality_analysis',
                    'spatial_distribution_analysis',
                    'spatial_relationships_analysis',
                    'geospatial_proximity_analysis',
                ])

        # Data quality recommendations
        recommendations.extend([
            'data_range_validation',
            'data_types_validation'
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen and rec in self.components:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:15]  # Limit to 15 recommendations
    
    def get_component_info(self, analysis_id: str) -> Dict[str, Any]:
        """Get complete information about a component."""
        if analysis_id not in self.components:
            return {}
        
        comp_info = self.components[analysis_id]
        metadata = comp_info.get('metadata', {})
        
        return {
            'id': analysis_id,
            'name': self.get_analysis_display_name(analysis_id),
            'description': metadata.get('description', ''),
            'category': comp_info.get('category', 'Unknown'),
            'complexity': metadata.get('complexity', 'intermediate'),
            'estimated_runtime': metadata.get('estimated_runtime', '1-5 seconds'),
            'tags': metadata.get('tags', []),
            'icon': metadata.get('icon', '📊'),
            'component_class': comp_info['component'].__name__
        }


# Create global instance
granular_generators = GranularAnalysisCodeGenerators()
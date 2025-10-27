"""Granular Analysis Components Package.

This package contains highly granular analysis components that provide focused,
specific analysis options instead of monolithic, overwhelming results.

Each component is completely independent and can be used individually.
"""

# ===== DATA QUALITY COMPONENTS =====
from .data_quality.duplicate_detection import DuplicateDetectionAnalysis
from .data_quality.missing_value_analysis import MissingValueAnalysis  
from .data_quality.data_types_validation import DataTypesValidationAnalysis
from .data_quality.dataset_shape_analysis import DatasetShapeAnalysis
from .data_quality.data_range_validation import DataRangeValidationAnalysis

# ===== UNIVARIATE ANALYSIS COMPONENTS =====
# Numeric Analysis
from .numeric.summary_statistics import SummaryStatisticsAnalysis
from .numeric.skewness_analysis import SkewnessAnalysis
from .numeric.skewness_statistics import SkewnessStatisticsAnalysis
from .numeric.skewness_visualization import SkewnessVisualizationAnalysis
from .numeric.normality_test import NormalityTestAnalysis
from .numeric.distribution_plots import DistributionPlotsAnalysis
from .numeric.histogram_plots import HistogramPlotsAnalysis
from .numeric.box_plots import BoxPlotsAnalysis
from .numeric.violin_plots import ViolinPlotsAnalysis
from .numeric.kde_plots import KDEPlotsAnalysis
from .numeric.frequency_analysis import NumericFrequencyAnalysis

# Categorical Analysis
from .categorical.frequency_analysis import CategoricalFrequencyAnalysis
from .categorical.visualization import CategoricalVisualizationAnalysis
from .categorical.bar_charts import CategoricalBarChartsAnalysis
from .categorical.pie_charts import CategoricalPieChartsAnalysis
from .categorical.cardinality_profile import CategoricalCardinalityProfileAnalysis
from .categorical.rare_category_detection import RareCategoryDetectionAnalysis

# ===== BIVARIATE/MULTIVARIATE ANALYSIS COMPONENTS =====
from .bivariate.correlation_analysis import CorrelationAnalysis
from .bivariate.correlation_pearson import PearsonCorrelationAnalysis
from .bivariate.correlation_spearman import SpearmanCorrelationAnalysis
from .bivariate.scatter_plot_analysis import ScatterPlotAnalysis
from .bivariate.cross_tabulation_analysis import CrossTabulationAnalysis
from .bivariate.categorical_numeric_relationships import CategoricalNumericRelationshipAnalysis

# ===== OUTLIER DETECTION COMPONENTS =====
from .outliers.iqr_detection import IQROutlierDetection
from .outliers.zscore_detection import ZScoreOutlierDetection
from .outliers.visual_inspection import VisualOutlierInspection
from .outliers.distribution_visualization import OutlierDistributionVisualization
from .outliers.scatter_matrix import OutlierScatterMatrixVisualization

# ===== TIME-SERIES COMPONENTS =====
from .time_series.temporal_trends import TemporalTrendAnalysis
from .time_series.seasonality_detection import SeasonalityDetectionAnalysis
from .time_series.datetime_feature_extraction import DatetimeFeatureExtractionAnalysis

# ===== GEOSPATIAL ANALYSIS COMPONENTS =====
from .geospatial.coordinate_system_projection_check import CoordinateSystemProjectionCheck
from .geospatial.spatial_data_quality import SpatialDataQualityAnalysis
from .geospatial.spatial_distribution import SpatialDistributionAnalysis
from .geospatial.spatial_relationships import SpatialRelationshipsAnalysis
from .geospatial.proximity_analysis import GeospatialProximityAnalysis

# ===== TEXT ANALYSIS COMPONENTS =====
from .text.text_length_distribution import TextLengthDistributionAnalysis
from .text.text_token_frequency import TextTokenFrequencyAnalysis
from .text.text_vocabulary_summary import TextVocabularySummaryAnalysis
from .text.text_feature_engineering_profile import TextFeatureEngineeringProfileAnalysis
from .text.text_nlp_profile import TextNLPProfileAnalysis

# ===== RELATIONSHIP EXPLORATION COMPONENTS =====
from .relationships.multicollinearity_analysis import MulticollinearityAnalysis
from .relationships.pca_analysis import PCADimensionalityReduction
from .relationships.pca_scree_plot import PCAScreePlotAnalysis
from .relationships.pca_cumulative_variance import PCACumulativeVarianceAnalysis
from .relationships.pca_visualization import PCAVisualizationAnalysis
from .relationships.pca_biplot import PCABiplotAnalysis
from .relationships.pca_heatmap import PCAHeatmapAnalysis
from .relationships.cluster_tendency_analysis import ClusterTendencyAnalysis
from .relationships.cluster_segmentation_analysis import ClusterSegmentationAnalysis
from .relationships.network_analysis import NetworkAnalysis
from .relationships.entity_network import EntityRelationshipNetwork

# ===== TARGET VARIABLE COMPONENTS =====
from .target.target_variable_analysis import TargetVariableAnalysis

# ===== MARKETING ANALYSIS COMPONENTS =====
from .marketing.campaign_metrics_analysis import CampaignMetricsAnalysis
from .marketing.conversion_funnel_analysis import ConversionFunnelAnalysis
from .marketing.engagement_analysis import EngagementAnalysis
from .marketing.channel_performance_analysis import ChannelPerformanceAnalysis
from .marketing.audience_segmentation_analysis import AudienceSegmentationAnalysis
from .marketing.roi_analysis import ROIAnalysis
from .marketing.attribution_analysis import AttributionAnalysis
from .marketing.cohort_analysis import CohortAnalysis


# ===== COMPONENT REGISTRY =====
def get_all_granular_components():
    """Get all available granular analysis components."""
    return {
        # Data Quality & Structure
        'dataset_shape_analysis': {
            'component': DatasetShapeAnalysis,
            'category': 'Data Quality & Structure',
            'metadata': DatasetShapeAnalysis().get_metadata()
        },
        'data_range_validation': {
            'component': DataRangeValidationAnalysis,
            'category': 'Data Quality & Structure',
            'metadata': DataRangeValidationAnalysis().get_metadata()
        },
        'data_types_validation': {
            'component': DataTypesValidationAnalysis,
            'category': 'Data Quality & Structure', 
            'metadata': DataTypesValidationAnalysis().get_metadata() if hasattr(DataTypesValidationAnalysis, 'get_metadata') else {}
        },
        'missing_value_analysis': {
            'component': MissingValueAnalysis,
            'category': 'Data Quality & Structure',
            'metadata': MissingValueAnalysis().get_metadata() if hasattr(MissingValueAnalysis, 'get_metadata') else {}
        },
        'duplicate_detection': {
            'component': DuplicateDetectionAnalysis,
            'category': 'Data Quality & Structure',
            'metadata': DuplicateDetectionAnalysis().get_metadata() if hasattr(DuplicateDetectionAnalysis, 'get_metadata') else {}
        },
        
        # Univariate Analysis - Numeric
        'summary_statistics': {
            'component': SummaryStatisticsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'metadata': SummaryStatisticsAnalysis().get_metadata() if hasattr(SummaryStatisticsAnalysis, 'get_metadata') else {}
        },
        'distribution_plots': {
            'component': DistributionPlotsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'metadata': DistributionPlotsAnalysis().get_metadata() if hasattr(DistributionPlotsAnalysis, 'get_metadata') else {}
        },
        'numeric_frequency_analysis': {
            'component': NumericFrequencyAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'metadata': NumericFrequencyAnalysis().get_metadata()
        },
        'histogram_plots': {
            'component': HistogramPlotsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Distribution Plots',
            'metadata': HistogramPlotsAnalysis().get_metadata() if hasattr(HistogramPlotsAnalysis, 'get_metadata') else {}
        },
        'box_plots': {
            'component': BoxPlotsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Distribution Plots',
            'metadata': BoxPlotsAnalysis().get_metadata() if hasattr(BoxPlotsAnalysis, 'get_metadata') else {}
        },
        'violin_plots': {
            'component': ViolinPlotsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Distribution Plots',
            'metadata': ViolinPlotsAnalysis().get_metadata() if hasattr(ViolinPlotsAnalysis, 'get_metadata') else {}
        },
        'kde_plots': {
            'component': KDEPlotsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Distribution Plots',
            'metadata': KDEPlotsAnalysis().get_metadata() if hasattr(KDEPlotsAnalysis, 'get_metadata') else {}
        },
        'skewness_analysis': {
            'component': SkewnessAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'metadata': SkewnessAnalysis().get_metadata() if hasattr(SkewnessAnalysis, 'get_metadata') else {}
        },
        'skewness_statistics': {
            'component': SkewnessStatisticsAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Skewness Analysis',
            'metadata': SkewnessStatisticsAnalysis().get_metadata() if hasattr(SkewnessStatisticsAnalysis, 'get_metadata') else {}
        },
        'skewness_visualization': {
            'component': SkewnessVisualizationAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'subcategory': 'Skewness Analysis',
            'metadata': SkewnessVisualizationAnalysis().get_metadata() if hasattr(SkewnessVisualizationAnalysis, 'get_metadata') else {}
        },
        'normality_test': {
            'component': NormalityTestAnalysis,
            'category': 'Univariate Analysis (Numeric)',
            'metadata': NormalityTestAnalysis().get_metadata() if hasattr(NormalityTestAnalysis, 'get_metadata') else {}
        },
        
        # Univariate Analysis - Categorical
        'categorical_frequency_analysis': {
            'component': CategoricalFrequencyAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'metadata': CategoricalFrequencyAnalysis().get_metadata()
        },
        'categorical_visualization': {
            'component': CategoricalVisualizationAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'metadata': CategoricalVisualizationAnalysis().get_metadata()
        },
        'categorical_bar_charts': {
            'component': CategoricalBarChartsAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'subcategory': 'Visualization',
            'metadata': CategoricalBarChartsAnalysis().get_metadata()
        },
        'categorical_pie_charts': {
            'component': CategoricalPieChartsAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'subcategory': 'Visualization',
            'metadata': CategoricalPieChartsAnalysis().get_metadata()
        },
        'categorical_cardinality_profile': {
            'component': CategoricalCardinalityProfileAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'subcategory': 'Category Health',
            'metadata': CategoricalCardinalityProfileAnalysis().get_metadata()
        },
        'rare_category_detection': {
            'component': RareCategoryDetectionAnalysis,
            'category': 'Univariate Analysis (Categorical)',
            'subcategory': 'Category Health',
            'metadata': RareCategoryDetectionAnalysis().get_metadata()
        },
        
        # Bivariate/Multivariate Analysis
        'correlation_analysis': {
            'component': CorrelationAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'metadata': CorrelationAnalysis().get_metadata()
        },
        'pearson_correlation': {
            'component': PearsonCorrelationAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'subcategory': 'Correlation Analysis',
            'metadata': PearsonCorrelationAnalysis().get_metadata()
        },
        'spearman_correlation': {
            'component': SpearmanCorrelationAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'subcategory': 'Correlation Analysis',
            'metadata': SpearmanCorrelationAnalysis().get_metadata()
        },
        'scatter_plot_analysis': {
            'component': ScatterPlotAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'metadata': ScatterPlotAnalysis().get_metadata()
        },
        'cross_tabulation_analysis': {
            'component': CrossTabulationAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'metadata': CrossTabulationAnalysis().get_metadata()
        },
        'categorical_numeric_relationships': {
            'component': CategoricalNumericRelationshipAnalysis,
            'category': 'Bivariate/Multivariate Analysis',
            'metadata': CategoricalNumericRelationshipAnalysis().get_metadata()
        },
        
        # Outlier & Anomaly Detection
        'iqr_outlier_detection': {
            'component': IQROutlierDetection,
            'category': 'Outlier & Anomaly Detection',
            'metadata': IQROutlierDetection().get_metadata()
        },
        'zscore_outlier_detection': {
            'component': ZScoreOutlierDetection,
            'category': 'Outlier & Anomaly Detection',
            'metadata': ZScoreOutlierDetection().get_metadata()
        },
        'visual_outlier_inspection': {
            'component': VisualOutlierInspection,
            'category': 'Outlier & Anomaly Detection',
            'metadata': VisualOutlierInspection().get_metadata()
        },
        'outlier_distribution_visualization': {
            'component': OutlierDistributionVisualization,
            'category': 'Outlier & Anomaly Detection',
            'subcategory': 'Visual Inspection',
            'metadata': OutlierDistributionVisualization().get_metadata() if hasattr(OutlierDistributionVisualization, 'get_metadata') else {}
        },
        'outlier_scatter_matrix': {
            'component': OutlierScatterMatrixVisualization,
            'category': 'Outlier & Anomaly Detection',
            'subcategory': 'Visual Inspection',
            'metadata': OutlierScatterMatrixVisualization().get_metadata() if hasattr(OutlierScatterMatrixVisualization, 'get_metadata') else {}
        },
        
        # Time-Series Exploration
        'temporal_trend_analysis': {
            'component': TemporalTrendAnalysis,
            'category': 'Time-Series Exploration',
            'metadata': TemporalTrendAnalysis().get_metadata()
        },
        'seasonality_detection': {
            'component': SeasonalityDetectionAnalysis,
            'category': 'Time-Series Exploration',
            'metadata': SeasonalityDetectionAnalysis().get_metadata()
        },
        'datetime_feature_extraction': {
            'component': DatetimeFeatureExtractionAnalysis,
            'category': 'Time-Series Exploration',
            'metadata': DatetimeFeatureExtractionAnalysis().get_metadata()
        },

        # Geospatial Analysis
        'coordinate_system_projection_check': {
            'component': CoordinateSystemProjectionCheck,
            'category': 'Geospatial Analysis',
            'metadata': CoordinateSystemProjectionCheck().get_metadata()
        },
        'spatial_data_quality_analysis': {
            'component': SpatialDataQualityAnalysis,
            'category': 'Geospatial Analysis',
            'metadata': SpatialDataQualityAnalysis().get_metadata()
        },
        'spatial_distribution_analysis': {
            'component': SpatialDistributionAnalysis,
            'category': 'Geospatial Analysis',
            'metadata': SpatialDistributionAnalysis().get_metadata()
        },
        'spatial_relationships_analysis': {
            'component': SpatialRelationshipsAnalysis,
            'category': 'Geospatial Analysis',
            'metadata': SpatialRelationshipsAnalysis().get_metadata()
        },
        'geospatial_proximity_analysis': {
            'component': GeospatialProximityAnalysis,
            'category': 'Geospatial Analysis',
            'metadata': GeospatialProximityAnalysis().get_metadata()
        },

        # Text Analysis
        'text_length_distribution': {
            'component': TextLengthDistributionAnalysis,
            'category': 'Text Analysis',
            'metadata': TextLengthDistributionAnalysis().get_metadata()
        },
        'text_token_frequency': {
            'component': TextTokenFrequencyAnalysis,
            'category': 'Text Analysis',
            'metadata': TextTokenFrequencyAnalysis().get_metadata()
        },
        'text_vocabulary_summary': {
            'component': TextVocabularySummaryAnalysis,
            'category': 'Text Analysis',
            'metadata': TextVocabularySummaryAnalysis().get_metadata()
        },
        'text_feature_engineering_profile': {
            'component': TextFeatureEngineeringProfileAnalysis,
            'category': 'Text Analysis',
            'metadata': TextFeatureEngineeringProfileAnalysis().get_metadata()
        },
        'text_nlp_profile': {
            'component': TextNLPProfileAnalysis,
            'category': 'Text Analysis',
            'metadata': TextNLPProfileAnalysis().get_metadata()
        },
        
        # Relationship Exploration
        'multicollinearity_analysis': {
            'component': MulticollinearityAnalysis,
            'category': 'Relationship Exploration',
            'metadata': MulticollinearityAnalysis().get_metadata()
        },
        'pca_dimensionality_reduction': {
            'component': PCADimensionalityReduction,
            'category': 'Relationship Exploration',
            'metadata': PCADimensionalityReduction().get_metadata()
        },
        'pca_scree_plot': {
            'component': PCAScreePlotAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'PCA Analysis',
            'metadata': PCAScreePlotAnalysis().get_metadata() if hasattr(PCAScreePlotAnalysis, 'get_metadata') else {}
        },
        'pca_cumulative_variance': {
            'component': PCACumulativeVarianceAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'PCA Analysis',
            'metadata': PCACumulativeVarianceAnalysis().get_metadata() if hasattr(PCACumulativeVarianceAnalysis, 'get_metadata') else {}
        },
        'pca_visualization': {
            'component': PCAVisualizationAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'PCA Analysis',
            'metadata': PCAVisualizationAnalysis().get_metadata() if hasattr(PCAVisualizationAnalysis, 'get_metadata') else {}
        },
        'pca_biplot': {
            'component': PCABiplotAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'PCA Analysis',
            'metadata': PCABiplotAnalysis().get_metadata() if hasattr(PCABiplotAnalysis, 'get_metadata') else {}
        },
        'pca_heatmap': {
            'component': PCAHeatmapAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'PCA Analysis',
            'metadata': PCAHeatmapAnalysis.get_metadata()
        },
        'cluster_tendency_analysis': {
            'component': ClusterTendencyAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'Clustering',
            'metadata': ClusterTendencyAnalysis().get_metadata(),
        },
        'cluster_segmentation_analysis': {
            'component': ClusterSegmentationAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'Clustering',
            'metadata': ClusterSegmentationAnalysis().get_metadata(),
        },
        'network_analysis': {
            'component': NetworkAnalysis,
            'category': 'Relationship Exploration',
            'subcategory': 'Network Analysis',
            'metadata': NetworkAnalysis().get_metadata()
        },
        'entity_relationship_network': {
            'component': EntityRelationshipNetwork,
            'category': 'Relationship Exploration', 
            'subcategory': 'Network Analysis',
            'metadata': EntityRelationshipNetwork().get_metadata()
        },

        # Target Variable Analysis
        'target_variable_analysis': {
            'component': TargetVariableAnalysis,
            'category': 'Target & Outcome Analysis',
            'metadata': TargetVariableAnalysis().get_metadata()
        },
        
        # Marketing Analysis
        'campaign_metrics_analysis': {
            'component': CampaignMetricsAnalysis,
            'category': 'Marketing Analysis',
            'metadata': CampaignMetricsAnalysis().get_metadata()
        },
        'conversion_funnel_analysis': {
            'component': ConversionFunnelAnalysis,
            'category': 'Marketing Analysis', 
            'metadata': ConversionFunnelAnalysis().get_metadata()
        },
        'engagement_analysis': {
            'component': EngagementAnalysis,
            'category': 'Marketing Analysis',
            'metadata': EngagementAnalysis().get_metadata()
        },
        'channel_performance_analysis': {
            'component': ChannelPerformanceAnalysis,
            'category': 'Marketing Analysis',
            'metadata': ChannelPerformanceAnalysis().get_metadata()
        },
        'audience_segmentation_analysis': {
            'component': AudienceSegmentationAnalysis,
            'category': 'Marketing Analysis',
            'metadata': AudienceSegmentationAnalysis().get_metadata()
        },
        'roi_analysis': {
            'component': ROIAnalysis,
            'category': 'Marketing Analysis',
            'metadata': ROIAnalysis().get_metadata()
        },
        'attribution_analysis': {
            'component': AttributionAnalysis,
            'category': 'Marketing Analysis',
            'metadata': AttributionAnalysis().get_metadata()
        },
        'cohort_analysis': {
            'component': CohortAnalysis,
            'category': 'Marketing Analysis',
            'metadata': CohortAnalysis().get_metadata()
        }
    }


def get_components_by_category():
    """Get components grouped by category."""
    components = get_all_granular_components()
    categories = {}
    
    for comp_id, comp_info in components.items():
        category = comp_info['category']
        if category not in categories:
            categories[category] = []
        categories[category].append({
            'id': comp_id,
            'component': comp_info['component'],
            'metadata': comp_info['metadata']
        })
    
    return categories


__all__ = [
    # Data Quality
    'DatasetShapeAnalysis',
    'DataRangeValidationAnalysis', 
    'DuplicateDetectionAnalysis', 
    'MissingValueAnalysis', 
    'DataTypesValidationAnalysis',
    
    # Numeric Analysis  
    'SummaryStatisticsAnalysis',
    'SkewnessAnalysis',
    'SkewnessStatisticsAnalysis',
    'SkewnessVisualizationAnalysis', 
    'NormalityTestAnalysis',
    'DistributionPlotsAnalysis',
    'HistogramPlotsAnalysis',
    'BoxPlotsAnalysis',
    'ViolinPlotsAnalysis',
    'KDEPlotsAnalysis',
    
    # Categorical Analysis
    'CategoricalFrequencyAnalysis',
    'CategoricalVisualizationAnalysis',
    'CategoricalBarChartsAnalysis',
    'CategoricalPieChartsAnalysis',
    'CategoricalCardinalityProfileAnalysis',
    'RareCategoryDetectionAnalysis',
    
    # Bivariate/Multivariate
    'CorrelationAnalysis',
    'PearsonCorrelationAnalysis',
    'SpearmanCorrelationAnalysis',
    'ScatterPlotAnalysis',
    'CrossTabulationAnalysis',
    'CategoricalNumericRelationshipAnalysis',
    
    # Outliers
    'IQROutlierDetection',
    'ZScoreOutlierDetection',
    'VisualOutlierInspection',
    'OutlierDistributionVisualization',
    'OutlierScatterMatrixVisualization',
    
    # Time-Series
    'TemporalTrendAnalysis',
    'SeasonalityDetectionAnalysis',
    'DatetimeFeatureExtractionAnalysis',

    # Geospatial Analysis
    'CoordinateSystemProjectionCheck',
    'SpatialDataQualityAnalysis',
    'SpatialDistributionAnalysis',
    'SpatialRelationshipsAnalysis',
    'GeospatialProximityAnalysis',

    # Text Analysis
    'TextLengthDistributionAnalysis',
    'TextTokenFrequencyAnalysis',
    'TextVocabularySummaryAnalysis',
    'TextFeatureEngineeringProfileAnalysis',
    
    # Relationships
    'MulticollinearityAnalysis',
    'PCADimensionalityReduction',
    'PCAScreePlotAnalysis',
    'PCACumulativeVarianceAnalysis',
    'PCAVisualizationAnalysis',
    'PCABiplotAnalysis',
    'PCAHeatmapAnalysis',
    'ClusterTendencyAnalysis',
    'ClusterSegmentationAnalysis',
    'NetworkAnalysis',
    'EntityRelationshipNetwork',
    'TargetVariableAnalysis',
    
    # Marketing
    'CampaignMetricsAnalysis',
    'ConversionFunnelAnalysis', 
    'EngagementAnalysis',
    'ChannelPerformanceAnalysis',
    'AudienceSegmentationAnalysis',
    'ROIAnalysis',
    'AttributionAnalysis',
    'CohortAnalysis',
    
    # Functions
    'get_all_granular_components',
    'get_components_by_category'
]

"""Outlier Detection Package Init.

Imports all outlier detection components.
"""

from .iqr_detection import IQROutlierDetection
from .zscore_detection import ZScoreOutlierDetection  
from .visual_inspection import VisualOutlierInspection
from .distribution_visualization import OutlierDistributionVisualization
from .scatter_matrix import OutlierScatterMatrixVisualization

__all__ = [
    'IQROutlierDetection',
    'ZScoreOutlierDetection',
    'VisualOutlierInspection',
    'OutlierDistributionVisualization',
    'OutlierScatterMatrixVisualization'
]
"""
STA-UKAN: Subseasonal Temperature Forecast Refinement via 
Multisource Atmospheric Factor Fusion and Terrain-Aware Token-KAN
"""

__version__ = '1.0.0'
__author__ = 'STA-UKAN Team'

from .models import STA_UKAN, create_sta_ukan_model
from .data import AtmosphericDataset, DataNormalizer, create_dataloader
from .utils import Trainer, evaluate_model

__all__ = [
    'STA_UKAN',
    'create_sta_ukan_model',
    'AtmosphericDataset',
    'DataNormalizer',
    'create_dataloader',
    'Trainer',
    'evaluate_model',
]

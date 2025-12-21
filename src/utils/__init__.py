"""
Utility functions module.
"""

from .trainer import Trainer
from .metrics import (
    rmse, mae, mape, r2_score, bias, skill_score, evaluate_model
)

# Optional visualization imports
try:
    from .visualization import (
        plot_training_history,
        plot_predictions,
        plot_forecast_horizon_metrics,
        plot_error_distribution,
        plot_attention_weights
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    plot_training_history = None
    plot_predictions = None
    plot_forecast_horizon_metrics = None
    plot_error_distribution = None
    plot_attention_weights = None

__all__ = [
    'Trainer',
    'rmse',
    'mae',
    'mape',
    'r2_score',
    'bias',
    'skill_score',
    'evaluate_model',
]

if _HAS_VISUALIZATION:
    __all__.extend([
        'plot_training_history',
        'plot_predictions',
        'plot_forecast_horizon_metrics',
        'plot_error_distribution',
        'plot_attention_weights',
    ])

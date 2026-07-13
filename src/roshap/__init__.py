"""roshap — Robust SHAP feature importance.

Bootstrap SHAP + zero-inflated density estimation + robust (RoSHAP) ranking.
"""
from .api import RoshapResult, explain, explain_shap_values
from .bootstrap import (
    aggregate_shap_by_feature,
    boot_1_repeat_inference,
    boot_multi_repeat_inference_keep_all,
    boot_multi_repeat_inference_keep_feature,
)
from .distributions import (
    estimate_feature_level_mixture,
    estimate_feature_level_mixture_fast,
    estimate_feature_level_mixture_preagg,
    estimate_sample_feature_distribution,
    fit_zero_inflated_kde,
    fit_zero_inflated_normal,
    zero_inflated_kde_pdf,
    zero_inflated_pdf,
)
from .models import (
    CatBoostWrapper,
    KerasWrapper,
    LightGBMWrapper,
    ModelWrapper,
    PyTorchWrapper,
    SklearnWrapper,
    XGBoostWrapper,
    create_model_wrapper,
)
from .ranking import add_roshap_stat, rank_features
from . import plotting
from .plotting import (
    plot_feature_distribution,
    plot_feature_distribution_raw,
    plot_ridge,
    plot_sample_feature_distribution,
    plot_sample_top_features,
    plot_top_features,
    plot_top_features_density,
)

__version__ = "0.1.0"

__all__ = [
    "explain",
    "explain_shap_values",
    "RoshapResult",
    # models
    "ModelWrapper",
    "XGBoostWrapper",
    "SklearnWrapper",
    "LightGBMWrapper",
    "CatBoostWrapper",
    "PyTorchWrapper",
    "KerasWrapper",
    "create_model_wrapper",
    # bootstrap
    "boot_1_repeat_inference",
    "boot_multi_repeat_inference_keep_all",
    "boot_multi_repeat_inference_keep_feature",
    "aggregate_shap_by_feature",
    # distributions
    "fit_zero_inflated_kde",
    "fit_zero_inflated_normal",
    "zero_inflated_pdf",
    "zero_inflated_kde_pdf",
    "estimate_sample_feature_distribution",
    "estimate_feature_level_mixture",
    "estimate_feature_level_mixture_preagg",
    "estimate_feature_level_mixture_fast",
    # ranking
    "add_roshap_stat",
    "rank_features",
    # plotting
    "plotting",
    "plot_ridge",
    "plot_feature_distribution",
    "plot_feature_distribution_raw",
    "plot_sample_feature_distribution",
    "plot_sample_top_features",
    "plot_top_features",
    "plot_top_features_density",
    "__version__",
]

"""High-level roshap API: explain(), explain_shap_values(), RoshapResult."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._utils import encode_labels, infer_task, spawn_bootstrap_seeds, validate_X_y
from .bootstrap import (
    _normalize_task,
    aggregate_shap_by_feature,
    boot_multi_repeat_inference_keep_all,
    boot_multi_repeat_inference_keep_feature,
)
from .distributions import (
    estimate_feature_level_mixture_preagg,
    estimate_sample_feature_distribution,
)
from .models import ModelWrapper, create_model_wrapper
from .ranking import add_roshap_stat, rank_features
from . import plotting

_DEFAULT_BOOSTER_PARAMS = {
    "xgboost": {
        "binary": {"objective": "binary:logistic", "eval_metric": "logloss"},
        "multiclass": {"objective": "multi:softprob", "eval_metric": "mlogloss"},
        "regression": {"objective": "reg:squarederror"},
    },
    "lightgbm": {
        "binary": {"objective": "binary", "verbose": -1},
        "multiclass": {"objective": "multiclass", "verbose": -1},
        "regression": {"objective": "regression", "verbose": -1},
    },
    "catboost": {
        "binary": {"loss_function": "Logloss"},
        "multiclass": {"loss_function": "MultiClass"},
        "regression": {"loss_function": "RMSE"},
    },
}

_SEED_KEY = {"xgboost": "seed", "lightgbm": "seed", "catboost": "random_seed"}


def _build_wrapper(model, model_params, task, n_classes, num_boost_round, random_state):
    """Resolve the model argument into a fitted-ready ModelWrapper."""
    if isinstance(model, ModelWrapper):
        return model
    if not isinstance(model, str):
        raise TypeError(
            "model must be a ModelWrapper instance or one of the strings "
            f"{sorted(_DEFAULT_BOOSTER_PARAMS)}; got {type(model)}."
        )

    name = model.lower()
    if name not in _DEFAULT_BOOSTER_PARAMS:
        raise ValueError(
            f"Unknown model string '{model}'. Use one of {sorted(_DEFAULT_BOOSTER_PARAMS)}, "
            "or build any other model with create_model_wrapper(...) / a custom "
            "ModelWrapper subclass and pass the instance as model=."
        )

    params = dict(_DEFAULT_BOOSTER_PARAMS[name][task])
    if task == "multiclass":
        params["num_class"] = n_classes
    if model_params:
        params.update(model_params)
    params.setdefault(_SEED_KEY[name], random_state)

    return create_model_wrapper(name, params=params, num_boost_round=num_boost_round)


@dataclass
class RoshapResult:
    """Results of a roshap analysis.

    Attributes
    ----------
    boot_results : list of pd.DataFrame
        Pre-aggregated bootstrap SHAP results, one frame per bootstrap draw
        (columns: feature[, class_id], perm_round, sum_shap, sum_abs_shap,
        mean_abs_shap, n_samples, bootstrap_id, boot_random_state).
    raw_results : list of pd.DataFrame or None
        Raw per-sample SHAP rows (only kept with keep="all" or the
        precomputed-SHAP path).
    feature_stats : pd.DataFrame
        Per-feature zero-inflated density fits and summary statistics.
    ranking_ : pd.DataFrame
        feature_stats with SNR and roshap_stat columns added.
    task : str
        "binary", "multiclass", or "regression".
    feature_names : list of str
    classes_ : np.ndarray or None
        Original class labels when y was label-encoded.
    params : dict
        Echo of the configuration used to produce this result.
    """

    boot_results: list
    feature_stats: pd.DataFrame
    ranking_: pd.DataFrame
    task: str
    feature_names: list
    raw_results: list | None = None
    classes_: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    _sample_stats_cache: pd.DataFrame | None = field(default=None, repr=False)

    # ----- tables -------------------------------------------------------

    def top_features(self, k=15, by="roshap_stat", class_id=None):
        """Top-k features with their attribution statistics, best first."""
        return rank_features(self.ranking_, k=k, by=by, class_id=class_id)

    def ranking(self, by="roshap_stat", class_id=None):
        """Full feature ranking table sorted by the RoSHAP statistic."""
        return rank_features(self.ranking_, k=None, by=by, class_id=class_id)

    # ----- feature-level plots -------------------------------------------

    def plot_ridge(self, top_k=15, by="roshap_stat", agg_col="sum_abs_shap",
                   class_id=None, show=True, **kwargs):
        """Ridge plot of attribution distributions for the top-k features."""
        features = self.top_features(k=top_k, by=by, class_id=class_id)["feature"].astype(str).tolist()
        return plotting.plot_ridge(
            self.boot_results, features, agg_col=agg_col, class_id=class_id,
            show=show, **kwargs,
        )

    def plot_feature(self, feature, class_id=None, show=True, **kwargs):
        """Fitted density + histogram of one feature's bootstrap attributions."""
        return plotting.plot_feature_distribution(
            self.feature_stats, feature, self.boot_results,
            class_id=class_id, show=show, **kwargs,
        )

    def plot_top_features(self, top_k=15, score_col="roshap_stat",
                          show_metric="sd_estimated", show=True, **kwargs):
        """Bar chart of the top-k features with error bars."""
        return plotting.plot_top_features(
            self.ranking_, top_k=top_k, score_col=score_col,
            show_metric=show_metric, show=show, **kwargs,
        )

    def plot_top_features_density(self, top_k=15, score_col="roshap_stat",
                                  show=True, **kwargs):
        """Overlaid zero-inflated densities of the top-k features."""
        return plotting.plot_top_features_density(
            self.ranking_, top_k=top_k, score_col=score_col, show=show, **kwargs,
        )

    # ----- sample-level evaluation ---------------------------------------

    def _require_raw(self):
        if self.raw_results is None:
            raise ValueError(
                "Sample-level evaluation needs raw per-sample SHAP rows. "
                "Re-run explain(..., keep='all') or use explain_shap_values()."
            )

    def sample_feature_distributions(self, bandwidth=0.2, kernel="gaussian",
                                     zero_tol=0.0, approx="kde"):
        """Per-(sample, feature) zero-inflated density fits from raw SHAP rows."""
        self._require_raw()
        if self._sample_stats_cache is None:
            self._sample_stats_cache = estimate_sample_feature_distribution(
                self.raw_results, bandwidth=bandwidth, kernel=kernel,
                zero_tol=zero_tol, approx=approx,
            )
        return self._sample_stats_cache

    def plot_sample(self, sample_id, top_k=15, class_id=None, show=True, **kwargs):
        """Overlay the top-k feature attribution densities for one sample."""
        mix_df = self.sample_feature_distributions()
        return plotting.plot_sample_top_features(
            mix_df, sample_id, class_id=class_id, top_k=top_k, show=show, **kwargs,
        )


def explain(
    X,
    y,
    model="xgboost",
    model_params=None,
    task="auto",
    n_bootstrap=500,
    b_model=1,
    inner_variance="permutation",
    zero_tol=1e-8,
    approx="kde",
    bandwidth=0.2,
    kernel="gaussian",
    keep="feature",
    random_state=42,
    n_jobs=-1,
    positive_only=False,
    smote=False,
    smote_k_neighbors=5,
    show_progress=True,
    num_boost_round=100,
):
    """Bootstrap-SHAP feature importance with robust (RoSHAP) ranking.

    Parameters
    ----------
    X : pd.DataFrame or array-like of shape (n_samples, n_features)
    y : pd.Series or array-like of shape (n_samples,)
        Classification labels or regression targets.
    model : str or ModelWrapper
        "xgboost", "lightgbm", or "catboost" for a built-in booster with
        task-appropriate defaults, or any ModelWrapper instance (see
        create_model_wrapper for sklearn / PyTorch / Keras models).
    model_params : dict, optional
        Booster parameters merged over the task defaults (string models only).
    task : {"auto", "binary", "multiclass", "regression"}
    n_bootstrap : int
        Number of bootstrap resamples.
    b_model : int
        Models trained per bootstrap resample (inner repeats).
    inner_variance : {"permutation", "seed"}
        Source of inner-model variation across the b_model repeats.
    zero_tol : float
        Threshold below which an aggregated attribution counts as zero.
    approx : {"kde", "normal"}
        Continuous density estimator for the zero-inflated mixture.
    bandwidth, kernel
        KDE settings (ignored for approx="normal").
    keep : {"feature", "all"}
        "all" additionally keeps raw per-sample SHAP rows, enabling
        sample-level evaluation (more memory).
    random_state : int
        Single seed driving bootstrap resampling and model seeds.
    n_jobs : int
        Parallel workers for the bootstrap loop (-1 = all cores).
    positive_only : bool or "both"
        Restrict aggregation to positive-class OOB samples (binary tasks).
    smote : bool
        Apply SMOTE to each bootstrap training set (requires roshap[smote]).
    num_boost_round : int
        Boosting rounds for string booster models.

    Returns
    -------
    RoshapResult
    """
    if keep not in {"feature", "all"}:
        raise ValueError("keep must be 'feature' or 'all'")

    X, y = validate_X_y(X, y)
    task = infer_task(y) if task == "auto" else _normalize_task(task)
    y, classes_ = encode_labels(y, task)
    n_classes = int(y.nunique()) if task != "regression" else None

    wrapper = _build_wrapper(model, model_params, task, n_classes, num_boost_round, random_state)
    bootstrap_random_states = spawn_bootstrap_seeds(random_state, n_bootstrap)

    common = dict(
        X=X, y=y, task=task,
        n_bootstrap=n_bootstrap, b_model=b_model, zero_tol=0,
        inner_variance=inner_variance,
        bootstrap_random_states=bootstrap_random_states,
        n_jobs=n_jobs, show_progress=show_progress,
        model_wrapper=wrapper,
        smote=smote, smote_k_neighbors=smote_k_neighbors,
    )

    raw_results = None
    if keep == "all":
        raw_results = boot_multi_repeat_inference_keep_all(**common)
        boot_results = []
        for i, raw_df in enumerate(raw_results):
            agg = aggregate_shap_by_feature(raw_df, positive_only=positive_only, y=y)
            agg["bootstrap_id"] = i
            agg["boot_random_state"] = bootstrap_random_states[i]
            boot_results.append(agg)
    else:
        boot_results = boot_multi_repeat_inference_keep_feature(
            **common, positive_only=positive_only,
        )

    feature_stats = estimate_feature_level_mixture_preagg(
        boot_results, agg_col="sum_abs_shap",
        bandwidth=bandwidth, kernel=kernel, zero_tol=zero_tol,
        support="positive", approx=approx,
    )
    ranking_ = add_roshap_stat(feature_stats)

    return RoshapResult(
        boot_results=boot_results,
        raw_results=raw_results,
        feature_stats=feature_stats,
        ranking_=ranking_,
        task=task,
        feature_names=list(X.columns),
        classes_=classes_,
        params={
            "model": model if isinstance(model, str) else type(model).__name__,
            "model_params": model_params,
            "n_bootstrap": n_bootstrap,
            "b_model": b_model,
            "inner_variance": inner_variance,
            "zero_tol": zero_tol,
            "approx": approx,
            "bandwidth": bandwidth,
            "kernel": kernel,
            "keep": keep,
            "random_state": random_state,
            "positive_only": positive_only,
            "smote": smote,
            "num_boost_round": num_boost_round,
        },
    )


def _shap_array_to_long(arr, feature_names, sample_ids):
    """Convert one (n_samples, n_features) or (n_samples, n_classes, n_features)
    SHAP array into a raw long DataFrame (sample_id, [class_id,] feature,
    shap_value, perm_round)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        n, d = arr.shape
        if len(feature_names) != d:
            raise ValueError(f"feature_names has {len(feature_names)} entries but SHAP array has {d} features.")
        return pd.DataFrame({
            "sample_id": np.repeat(sample_ids, d),
            "feature": np.tile(feature_names, n),
            "shap_value": arr.reshape(-1),
            "perm_round": 0,
        })
    if arr.ndim == 3:
        n, c, d = arr.shape
        if len(feature_names) != d:
            raise ValueError(f"feature_names has {len(feature_names)} entries but SHAP array has {d} features.")
        return pd.DataFrame({
            "sample_id": np.repeat(sample_ids, c * d),
            "class_id": np.tile(np.repeat(np.arange(c), d), n),
            "feature": np.tile(feature_names, n * c),
            "shap_value": arr.reshape(-1),
            "perm_round": 0,
        })
    raise ValueError(
        f"SHAP arrays must be 2D (n_samples, n_features) or 3D "
        f"(n_samples, n_classes, n_features); got shape {arr.shape}."
    )


def _normalize_shap_input(shap_values, feature_names, sample_ids):
    """Normalize the accepted shap_values forms into a list of raw long frames.

    Returns (raw_frames, feature_names, samples_as_draws) where
    samples_as_draws is True for the single-array form, in which each sample
    becomes one pseudo-bootstrap draw.
    """
    # Long DataFrame form
    if isinstance(shap_values, pd.DataFrame):
        df = shap_values.copy()
        required = {"sample_id", "feature", "shap_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Long-format shap_values is missing columns: {sorted(missing)}")
        if "perm_round" not in df.columns:
            df["perm_round"] = 0
        if "bootstrap_id" not in df.columns:
            df["bootstrap_id"] = 0
        feature_names = df["feature"].astype(str).unique().tolist()
        frames = [g.drop(columns="bootstrap_id") for _, g in df.groupby("bootstrap_id", sort=True)]
        return frames, feature_names, False

    # List of replicate arrays
    if isinstance(shap_values, (list, tuple)):
        if len(shap_values) == 0:
            raise ValueError("shap_values is an empty list.")
        arrs = [np.asarray(a, dtype=float) for a in shap_values]
        d = arrs[0].shape[-1]
        if feature_names is None:
            feature_names = [f"f{j}" for j in range(d)]
        frames = []
        for arr in arrs:
            n = arr.shape[0]
            ids = np.arange(n) if sample_ids is None else np.asarray(sample_ids)
            if len(ids) != n:
                raise ValueError(f"sample_ids has {len(ids)} entries but SHAP array has {n} samples.")
            frames.append(_shap_array_to_long(arr, feature_names, ids))
        return frames, list(feature_names), False

    # Single array: samples become pseudo-bootstrap draws
    arr = np.asarray(shap_values, dtype=float)
    d = arr.shape[-1]
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(d)]
    n = arr.shape[0]
    ids = np.arange(n) if sample_ids is None else np.asarray(sample_ids)
    if len(ids) != n:
        raise ValueError(f"sample_ids has {len(ids)} entries but SHAP array has {n} samples.")
    frames = [
        _shap_array_to_long(arr[i:i + 1], feature_names, ids[i:i + 1])
        for i in range(n)
    ]
    return frames, list(feature_names), True


def explain_shap_values(
    shap_values,
    feature_names=None,
    sample_ids=None,
    task="binary",
    zero_tol=1e-8,
    approx="kde",
    bandwidth=0.2,
    kernel="gaussian",
):
    """Run the roshap distribution-estimation + ranking pipeline on
    precomputed SHAP values (e.g. from a transformer or image model).

    Accepted forms of *shap_values*:

    1. Single 2D array ``(n_samples, n_features)`` — one SHAP matrix. Each
       sample becomes one attribution draw, so the feature-level densities
       describe variation across samples.
    2. Single 3D array ``(n_samples, n_classes, n_features)`` — multiclass
       variant of (1).
    3. List of 2D/3D arrays — one array per bootstrap/replicate (e.g. SHAP
       from B retrained models); densities describe variation across
       replicates of aggregated |SHAP|.
    4. Long DataFrame with columns ``sample_id, feature, shap_value`` and
       optional ``class_id, bootstrap_id, perm_round``.

    Sample-level evaluation (``.sample_feature_distributions()``,
    ``.plot_sample()``) is available on the result for every input form.

    Returns
    -------
    RoshapResult
    """
    frames, feature_names, samples_as_draws = _normalize_shap_input(
        shap_values, feature_names, sample_ids
    )

    boot_results = []
    for i, raw_df in enumerate(frames):
        agg = aggregate_shap_by_feature(raw_df)
        agg["bootstrap_id"] = i
        agg["boot_random_state"] = np.nan
        boot_results.append(agg)

    raw_results = [df.assign(bootstrap_id=i) for i, df in enumerate(frames)]

    feature_stats = estimate_feature_level_mixture_preagg(
        boot_results, agg_col="sum_abs_shap",
        bandwidth=bandwidth, kernel=kernel, zero_tol=zero_tol,
        support="positive", approx=approx,
    )
    ranking_ = add_roshap_stat(feature_stats)

    return RoshapResult(
        boot_results=boot_results,
        raw_results=raw_results,
        feature_stats=feature_stats,
        ranking_=ranking_,
        task=_normalize_task(task),
        feature_names=list(feature_names),
        classes_=None,
        params={
            "source": "precomputed_shap",
            "samples_as_draws": samples_as_draws,
            "n_draws": len(boot_results),
            "zero_tol": zero_tol,
            "approx": approx,
            "bandwidth": bandwidth,
            "kernel": kernel,
        },
    )

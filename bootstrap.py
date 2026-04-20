from contextlib import contextmanager
import copy
import os
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

try:
    from archive.model_wrapper import ModelWrapper
except ImportError:
    ModelWrapper = None

os.environ.setdefault("LOKY_PICKLER", "cloudpickle")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalize_task(task):
    if task is None:
        raise ValueError("task must be provided.")
    task_norm = str(task).strip().lower().replace("_", "-")
    aliases = {
        "binary":                   "binary",
        "binary-classification":    "binary",
        "multiclass":               "multiclass",
        "multi-class":              "multiclass",
        "multiclass-classification":"multiclass",
        "regression":               "regression",
    }
    if task_norm not in aliases:
        raise ValueError("Unsupported task. Use one of: 'binary', 'multiclass', 'regression'.")
    return aliases[task_norm]


def _normalize_inner_variance(inner_variance):
    if inner_variance not in {"seed", "permutation"}:
        raise ValueError("inner_variance must be 'seed' or 'permutation'.")
    return inner_variance


# ---------------------------------------------------------------------------
# SHAP extraction
# ---------------------------------------------------------------------------

def _extract_shap_feature_contribs(shap_values, task):
    """
    Strip the bias column from XGBoost pred_contribs output.

    Returns shape:
      binary/regression : (n_samples, n_features)
      multiclass        : (n_samples, n_classes, n_features)
    """
    if shap_values.ndim == 2:
        return shap_values[:, :-1]
    if shap_values.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape {shap_values.shape}. Expected 2D or 3D.")
    if task == "multiclass":
        return shap_values[:, :, :-1]
    if shap_values.shape[1] != 1:
        raise ValueError(f"Unexpected group dimension {shap_values.shape[1]} for task '{task}'.")
    return shap_values[:, 0, :-1]


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def boot_1_repeat_inference(
    X,
    y,
    task,
    r_model,
    zero_tol,
    inner_variance="permutation",
    params=None,
    boot_random_state=42,
    num_boost_round=100,
    model_wrapper=None,
):
    # zero_tol is intentionally unused in raw-output mode, retained for API compatibility.
    _ = zero_tol

    task_norm = _normalize_task(task)
    inner_variance = _normalize_inner_variance(inner_variance)

    if model_wrapper is None and params is None:
        raise ValueError("Either model_wrapper or params must be provided.")

    # --- Bootstrap split ---
    stratify_target = y if task_norm in {"binary", "multiclass"} else None
    X_boot_train, y_boot_train = resample(
        X,
        y,
        replace=True,
        n_samples=len(X),
        random_state=boot_random_state,
        stratify=stratify_target,
    )
    oob_mask = ~X.index.isin(X_boot_train.index)
    X_boot_test = X.loc[oob_mask]
    y_boot_test = y.loc[oob_mask]

    feature_names = X.columns.to_numpy()
    n_feat = X.shape[1]

    base_cols = ["sample_id", "feature", "shap_value", "perm_round"]
    if task_norm == "multiclass":
        base_cols.insert(1, "class_id")

    if X_boot_train.shape[0] == 0 or X_boot_test.shape[0] == 0:
        return pd.DataFrame(columns=base_cols)

    sample_ids = X_boot_test.index.to_numpy()

    if task_norm == "multiclass":
        n_classes = len(np.unique(y))
        if model_wrapper is None and params is not None:
            n_classes = int(params.get("num_class", n_classes))
        if n_classes <= 1:
            raise ValueError("For multiclass task, need at least 2 classes.")
        class_ids = np.arange(n_classes, dtype=int)

    round_frames = []

    for j in range(r_model):
        # Prepare train/test views for this iteration.
        if inner_variance == "permutation":
            rng = np.random.default_rng(boot_random_state * r_model + j)
            perm = rng.permutation(n_feat)
            inv_perm = np.argsort(perm)
            X_train_j = X_boot_train.iloc[:, perm]
            X_test_j = X_boot_test.iloc[:, perm]
        else:
            X_train_j = X_boot_train
            X_test_j = X_boot_test

        # Train model and compute SHAP.
        if model_wrapper is not None:
            model = copy.deepcopy(model_wrapper)
            if inner_variance == "seed":
                if hasattr(model, "set_seed"):
                    model.set_seed(j)
                elif hasattr(model, "params") and isinstance(model.params, dict):
                    model.params["seed"] = j
                elif hasattr(model, "model_params") and isinstance(model.model_params, dict):
                    model.model_params["random_state"] = j
            model.fit(X_train_j, y_boot_train)
            shap_feat = model.compute_shap(X_test_j, task=task_norm)
        else:
            params_local = params.copy()
            if inner_variance == "seed":
                params_local["seed"] = j
            dtrain = xgb.DMatrix(X_train_j, label=y_boot_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test_j, label=y_boot_test, enable_categorical=True)
            model = xgb.train(
                params_local,
                dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=False,
            )
            shap_feat = _extract_shap_feature_contribs(
                model.predict(dtest, pred_contribs=True, strict_shape=True), task_norm
            )

        # Restore original feature order in permutation mode.
        if inner_variance == "permutation":
            if task_norm == "multiclass":
                shap_feat = shap_feat[:, :, inv_perm]
            else:
                shap_feat = shap_feat[:, inv_perm]

        # Build per-round raw rows.
        if task_norm == "multiclass":
            n_test = shap_feat.shape[0]
            round_df = pd.DataFrame({
                "sample_id": np.repeat(sample_ids, n_classes * n_feat),
                "class_id": np.tile(np.repeat(class_ids, n_feat), n_test),
                "feature": np.tile(feature_names, n_test * n_classes),
                "shap_value": shap_feat.reshape(-1),
                "perm_round": j,
            })
        else:
            n_test = shap_feat.shape[0]
            round_df = pd.DataFrame({
                "sample_id": np.repeat(sample_ids, n_feat),
                "feature": np.tile(feature_names, n_test),
                "shap_value": shap_feat.reshape(-1),
                "perm_round": j,
            })

        round_frames.append(round_df)

    if not round_frames:
        return pd.DataFrame(columns=base_cols)

    return pd.concat(round_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Parallel bootstrap
# ---------------------------------------------------------------------------

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


def _boot_1_repeat_df_job(
    bootstrap_id, rs,
    X, y, task, b_model, zero_tol,
    params, num_boost_round, xgb_nthread,
    inner_variance, model_wrapper,
):
    if model_wrapper is None and params is not None:
        params_local = params.copy()
        if xgb_nthread is not None:
            params_local["nthread"] = xgb_nthread
    else:
        params_local = params

    result = boot_1_repeat_inference(
        X=X, y=y, task=task,
        r_model=b_model, zero_tol=zero_tol,
        inner_variance=inner_variance,
        params=params_local,
        boot_random_state=rs,
        num_boost_round=num_boost_round,
        model_wrapper=model_wrapper,
    )

    meta = {"bootstrap_id": bootstrap_id, "boot_random_state": rs}

    return result.assign(**meta)


def boot_multi_repeat_inference_keep_all(
    X,
    y,
    task,
    n_bootstrap,
    b_model,
    zero_tol,
    params=None,
    inner_variance="seed",
    bootstrap_random_states=None,
    num_boost_round=150,
    n_jobs=-1,
    backend="loky",
    xgb_nthread=1,
    show_progress=True,
    tqdm_desc="Bootstrap repeats",
    pre_dispatch="2*n_jobs",
    model_wrapper=None,
):
    """
    Run bootstrap resampling n_bootstrap times in parallel.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    task : str
        "binary", "multiclass", or "regression".
    n_bootstrap : int
        Number of bootstrap iterations.
    b_model : int
        Number of models per bootstrap iteration.
    zero_tol : float
        Kept for backward compatibility. Not used in raw-output mode.
    inner_variance : str
        "seed" or "permutation".
    params : dict, optional
        XGBoost parameters. Ignored when model_wrapper is provided.
    bootstrap_random_states : list, optional
        RNG seeds per bootstrap. Defaults to range(n_bootstrap).
    num_boost_round : int
        XGBoost boosting rounds. Ignored when model_wrapper is provided.
    n_jobs : int
        Parallel workers (-1 = all cores).
    backend : str
        joblib backend.
    xgb_nthread : int
        Threads per XGBoost model when parallelising across bootstraps.
    show_progress : bool
    tqdm_desc : str
    pre_dispatch : str
    model_wrapper : ModelWrapper, optional

    Returns
    -------
    list of pd.DataFrame
                Each DataFrame contains raw SHAP rows with columns:
                - binary/regression: sample_id, feature, shap_value, perm_round,
                    bootstrap_id, boot_random_state
                - multiclass: sample_id, class_id, feature, shap_value, perm_round,
                    bootstrap_id, boot_random_state
    """
    if model_wrapper is None and params is None:
        raise ValueError("Either model_wrapper or params must be provided.")

    _normalize_inner_variance(inner_variance)

    if bootstrap_random_states is None:
        bootstrap_random_states = list(range(n_bootstrap))
    elif len(bootstrap_random_states) != n_bootstrap:
        raise ValueError("len(bootstrap_random_states) must equal n_bootstrap.")

    task_norm = _normalize_task(task)
    parallel  = Parallel(n_jobs=n_jobs, backend=backend, pre_dispatch=pre_dispatch, verbose=0)

    jobs = (
        delayed(_boot_1_repeat_df_job)(
            bootstrap_id=i, rs=rs,
            X=X, y=y, task=task_norm,
            b_model=b_model, zero_tol=zero_tol,
            params=params, num_boost_round=num_boost_round,
            xgb_nthread=xgb_nthread,
            inner_variance=inner_variance,
            model_wrapper=model_wrapper,
        )
        for i, rs in enumerate(bootstrap_random_states)
    )

    if show_progress:
        with tqdm_joblib(tqdm(total=n_bootstrap, desc=tqdm_desc)):
            return parallel(jobs)
    return parallel(jobs)

# ...existing code...

def _boot_1_repeat_feature_agg_job(
    bootstrap_id, rs,
    X, y, task, b_model, zero_tol,
    params, num_boost_round, xgb_nthread,
    inner_variance, model_wrapper,
):
    """Like _boot_1_repeat_df_job but aggregates |SHAP| over samples immediately."""
    if model_wrapper is None and params is not None:
        params_local = params.copy()
        if xgb_nthread is not None:
            params_local["nthread"] = xgb_nthread
    else:
        params_local = params

    result = boot_1_repeat_inference(
        X=X, y=y, task=task,
        r_model=b_model, zero_tol=zero_tol,
        inner_variance=inner_variance,
        params=params_local,
        boot_random_state=rs,
        num_boost_round=num_boost_round,
        model_wrapper=model_wrapper,
    )

    # Aggregate |SHAP| over all samples per (feature, perm_round)
    group_cols = ["feature", "perm_round"]
    if "class_id" in result.columns:
        group_cols.insert(1, "class_id")

    agg = (
        result.groupby(group_cols, as_index=False)
        .agg(
            sum_abs_shap=("shap_value", lambda x: x.abs().sum()),
            mean_abs_shap=("shap_value", lambda x: x.abs().mean()),
            n_samples=("shap_value", "count"),
        )
    )
    agg["bootstrap_id"] = bootstrap_id
    agg["boot_random_state"] = rs
    return agg


def boot_multi_repeat_inference_keep_feature(
    X,
    y,
    task,
    n_bootstrap,
    b_model,
    zero_tol,
    params=None,
    inner_variance="seed",
    bootstrap_random_states=None,
    num_boost_round=150,
    n_jobs=-1,
    backend="loky",
    xgb_nthread=1,
    show_progress=True,
    tqdm_desc="Bootstrap repeats (feature-agg)",
    pre_dispatch="2*n_jobs",
    model_wrapper=None,
):
    """
    Like boot_multi_repeat_inference_keep_all but aggregates |SHAP| over
    all OOB samples per feature per perm_round, dropping sample_id.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame has columns:
        - binary/regression: feature, perm_round, sum_abs_shap, mean_abs_shap,
                             n_samples, bootstrap_id, boot_random_state
        - multiclass: feature, class_id, perm_round, sum_abs_shap, mean_abs_shap,
                      n_samples, bootstrap_id, boot_random_state
    """
    if model_wrapper is None and params is None:
        raise ValueError("Either model_wrapper or params must be provided.")

    _normalize_inner_variance(inner_variance)

    if bootstrap_random_states is None:
        bootstrap_random_states = list(range(n_bootstrap))
    elif len(bootstrap_random_states) != n_bootstrap:
        raise ValueError("len(bootstrap_random_states) must equal n_bootstrap.")

    task_norm = _normalize_task(task)
    parallel = Parallel(n_jobs=n_jobs, backend=backend, pre_dispatch=pre_dispatch, verbose=0)

    jobs = (
        delayed(_boot_1_repeat_feature_agg_job)(
            bootstrap_id=i, rs=rs,
            X=X, y=y, task=task_norm,
            b_model=b_model, zero_tol=zero_tol,
            params=params, num_boost_round=num_boost_round,
            xgb_nthread=xgb_nthread,
            inner_variance=inner_variance,
            model_wrapper=model_wrapper,
        )
        for i, rs in enumerate(bootstrap_random_states)
    )

    if show_progress:
        with tqdm_joblib(tqdm(total=n_bootstrap, desc=tqdm_desc)):
            return parallel(jobs)
    return parallel(jobs)

# ---------------------------------------------------------------------------
# Seed importance (standalone diagnostic)
# ---------------------------------------------------------------------------

def seed_importance(
    X,
    y,
    b_model,
    params,
    split_seed=123,
    num_boost_round=150,
    return_format="dataframe",
):
    """
    Train b_model XGBoost models with different seeds on a fixed train/test
    split and compare SHAP-based feature importance across runs.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    b_model : int
    params : dict
        XGBoost parameters (seed is overridden per run).
    split_seed : int
    num_boost_round : int
    return_format : str
        "dataframe" — one row per (run, feature).
        "stats"     — aggregated mean/sd per feature.

    Returns
    -------
    pd.DataFrame
    """
    if return_format not in {"dataframe", "stats"}:
        raise ValueError("return_format must be 'dataframe' or 'stats'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed, stratify=y
    )
    dtrain        = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest         = xgb.DMatrix(X_test,  label=y_test,  enable_categorical=True)
    feature_names = list(X.columns)

    all_runs = []
    for i in range(b_model):
        model = xgb.train(
            {**params, "seed": i}, dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        acc           = accuracy_score(y_test, (model.predict(dtest) > 0.5).astype(int))
        mean_abs_shap = np.abs(model.predict(dtest, pred_contribs=True)[:, :-1]).mean(axis=0)

        run_df = (
            pd.DataFrame({
                "run":           i,
                "feature":       feature_names,
                "mean_abs_shap": mean_abs_shap,
                "test_accuracy": acc,
            })
            .assign(rank=lambda d: d["mean_abs_shap"].rank(ascending=False, method="average"))
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        run_df["rank_order"] = np.arange(1, len(run_df) + 1)
        all_runs.append(run_df)

    importance_df = pd.concat(all_runs, ignore_index=True)

    if return_format == "dataframe":
        return importance_df

    return (
        importance_df
        .groupby("feature", as_index=False)
        .agg(
            mean_shap_mean=("mean_abs_shap", "mean"),
            mean_shap_sd  =("mean_abs_shap", "std"),
            mean_rank     =("rank",          "mean"),
            sd_rank       =("rank",          "std"),
        )
        .sort_values("mean_shap_mean", ascending=False)
        .reset_index(drop=True)
    )
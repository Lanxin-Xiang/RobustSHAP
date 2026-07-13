import numpy as np
import pandas as pd
import pytest

from roshap import (
    aggregate_shap_by_feature,
    boot_1_repeat_inference,
    boot_multi_repeat_inference_keep_all,
    boot_multi_repeat_inference_keep_feature,
)
from roshap.bootstrap import _boot_1_repeat_feature_agg_job

RAW_COLS = ["sample_id", "feature", "shap_value", "perm_round"]
AGG_COLS = ["feature", "perm_round", "sum_shap", "sum_abs_shap", "mean_abs_shap",
            "n_samples", "bootstrap_id", "boot_random_state"]


def test_boot_1_repeat_schema_and_oob(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    out = boot_1_repeat_inference(
        X, y, task="binary", r_model=1, zero_tol=0,
        boot_random_state=3, model_wrapper=xgb_wrapper_binary,
    )
    assert list(out.columns) == RAW_COLS
    # OOB sample ids must come from X's index
    assert set(out["sample_id"]).issubset(set(X.index))
    # OOB must be disjoint from the bootstrap resample
    from sklearn.utils import resample
    X_boot, _ = resample(X, y, replace=True, n_samples=len(X), random_state=3, stratify=y)
    assert set(out["sample_id"]).isdisjoint(set(X_boot.index))


def test_boot_1_repeat_multiclass_has_class_id(multiclass_data):
    from roshap import XGBoostWrapper

    X, y = multiclass_data
    w = XGBoostWrapper(
        {"objective": "multi:softprob", "num_class": 3, "seed": 0}, num_boost_round=5
    )
    out = boot_1_repeat_inference(
        X, y, task="multiclass", r_model=1, zero_tol=0,
        boot_random_state=1, model_wrapper=w,
    )
    assert "class_id" in out.columns
    assert set(out["class_id"]) == {0, 1, 2}


def test_keep_feature_schema(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    boot = boot_multi_repeat_inference_keep_feature(
        X, y, task="binary", n_bootstrap=3, b_model=1, zero_tol=0,
        inner_variance="permutation", model_wrapper=xgb_wrapper_binary,
        n_jobs=1, show_progress=False,
    )
    assert len(boot) == 3
    for df in boot:
        assert list(df.columns) == AGG_COLS
        assert set(df["feature"]) == set(X.columns)


def test_reproducible_with_same_seeds(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    kwargs = dict(
        X=X, y=y, task="binary", n_bootstrap=3, b_model=1, zero_tol=0,
        inner_variance="permutation", model_wrapper=xgb_wrapper_binary,
        n_jobs=1, show_progress=False,
    )
    a = boot_multi_repeat_inference_keep_feature(bootstrap_random_states=[7, 8, 9], **kwargs)
    b = boot_multi_repeat_inference_keep_feature(bootstrap_random_states=[7, 8, 9], **kwargs)
    c = boot_multi_repeat_inference_keep_feature(bootstrap_random_states=[10, 11, 12], **kwargs)
    for df_a, df_b in zip(a, b):
        pd.testing.assert_frame_equal(df_a, df_b)
    assert not all(
        df_a["sum_abs_shap"].equals(df_c["sum_abs_shap"]) for df_a, df_c in zip(a, c)
    )


def test_aggregate_matches_agg_job(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    raw = boot_1_repeat_inference(
        X, y, task="binary", r_model=1, zero_tol=0,
        boot_random_state=5, model_wrapper=xgb_wrapper_binary,
    )
    agg = aggregate_shap_by_feature(raw)
    agg["bootstrap_id"] = 0
    agg["boot_random_state"] = 5

    job = _boot_1_repeat_feature_agg_job(
        bootstrap_id=0, rs=5, X=X, y=y, task="binary", b_model=1, zero_tol=0,
        params=None, num_boost_round=10, xgb_nthread=1,
        inner_variance="permutation", model_wrapper=xgb_wrapper_binary,
    )
    pd.testing.assert_frame_equal(agg, job)


def test_positive_only_both_adds_label(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    boot = boot_multi_repeat_inference_keep_feature(
        X, y, task="binary", n_bootstrap=2, b_model=1, zero_tol=0,
        inner_variance="permutation", model_wrapper=xgb_wrapper_binary,
        n_jobs=1, show_progress=False, positive_only="both",
    )
    for df in boot:
        assert "label" in df.columns
        assert set(df["label"]) <= {0, 1}


def test_keep_all_schema(binary_data, xgb_wrapper_binary):
    X, y = binary_data
    raw = boot_multi_repeat_inference_keep_all(
        X, y, task="binary", n_bootstrap=2, b_model=1, zero_tol=0,
        inner_variance="permutation", model_wrapper=xgb_wrapper_binary,
        n_jobs=1, show_progress=False,
    )
    assert len(raw) == 2
    for df in raw:
        assert list(df.columns) == RAW_COLS + ["bootstrap_id", "boot_random_state"]


def test_regression_bootstrap(regression_data):
    from roshap import XGBoostWrapper

    X, y = regression_data
    w = XGBoostWrapper({"objective": "reg:squarederror", "seed": 0}, num_boost_round=5)
    boot = boot_multi_repeat_inference_keep_feature(
        X, y, task="regression", n_bootstrap=2, b_model=1, zero_tol=0,
        inner_variance="permutation", model_wrapper=w,
        n_jobs=1, show_progress=False,
    )
    for df in boot:
        assert not df.empty
        assert np.isfinite(df["sum_abs_shap"]).all()

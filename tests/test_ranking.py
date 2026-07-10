import numpy as np
import pandas as pd
import pytest

from roshap import add_roshap_stat, rank_features


@pytest.fixture
def feature_stats():
    return pd.DataFrame({
        "feature": ["a", "b", "c"],
        "median": [10.0, 4.0, 8.0],
        "sd_estimated": [2.0, 1.0, 0.0],
        "p_nonzero": [1.0, 0.5, 1.0],
        "nonzero_median": [10.0, 4.0, 8.0],
    })


def test_stat_formula(feature_stats):
    out = add_roshap_stat(feature_stats)
    # a: 1.0 * 10 * (10/2) = 50 ; b: 0.5 * 4 * (4/1) = 8
    assert out.loc[out["feature"] == "a", "roshap_stat"].iloc[0] == pytest.approx(50.0)
    assert out.loc[out["feature"] == "b", "roshap_stat"].iloc[0] == pytest.approx(8.0)


def test_zero_sd_guard(feature_stats):
    out = add_roshap_stat(feature_stats)
    row = out.loc[out["feature"] == "c"].iloc[0]
    assert row["SNR"] == 0.0
    assert row["roshap_stat"] == 0.0


def test_rank_features_ordering(feature_stats):
    ranked = rank_features(feature_stats, k=2)
    assert ranked["feature"].tolist() == ["a", "b"]
    assert ranked.index.tolist() == [1, 2]
    assert "importance" in ranked.columns


def test_rank_features_class_filter():
    df = pd.DataFrame({
        "feature": ["a", "a", "b", "b"],
        "class_id": [0, 1, 0, 1],
        "median": [1.0, 8.0, 4.0, 2.0],
        "sd_estimated": [1.0, 1.0, 1.0, 1.0],
        "p_nonzero": [1.0, 1.0, 1.0, 1.0],
        "nonzero_median": [1.0, 8.0, 4.0, 2.0],
    })
    per_class = rank_features(df, class_id=1)
    assert per_class["feature"].tolist() == ["a", "b"]
    averaged = rank_features(df)  # mean across classes: a=(1+64)/2=32.5, b=(16+4)/2=10
    assert averaged["feature"].tolist() == ["a", "b"]
    assert averaged["importance"].iloc[0] == pytest.approx(32.5)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        add_roshap_stat(pd.DataFrame({"feature": ["a"], "median": [1.0]}))

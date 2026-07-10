import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from roshap import (
    estimate_feature_level_mixture_preagg,
    plot_feature_distribution,
    plot_ridge,
    plot_sample_feature_distribution,
    plot_sample_top_features,
    plot_top_features,
    plot_top_features_density,
)
from roshap.distributions import estimate_sample_feature_distribution


@pytest.fixture
def preagg_boot(rng):
    frames = []
    for b in range(10):
        frames.append(pd.DataFrame({
            "feature": ["a", "b", "c"],
            "perm_round": 0,
            "sum_shap": rng.normal(size=3),
            "sum_abs_shap": np.abs(rng.normal([5.0, 3.0, 1.0], 0.8)),
            "mean_abs_shap": np.abs(rng.normal(0.5, 0.1, size=3)),
            "n_samples": 10,
            "bootstrap_id": b,
            "boot_random_state": b,
        }))
    return frames


@pytest.fixture
def feature_stats(preagg_boot):
    return estimate_feature_level_mixture_preagg(preagg_boot, zero_tol=1e-8)


@pytest.fixture
def raw_boot(rng):
    frames = []
    for b in range(8):
        rows = []
        for sid in range(5):
            for feat in ("a", "b"):
                rows.append({
                    "sample_id": sid,
                    "feature": feat,
                    "shap_value": rng.normal(),
                    "perm_round": 0,
                    "bootstrap_id": b,
                    "boot_random_state": b,
                })
        frames.append(pd.DataFrame(rows))
    return frames


def test_plot_ridge_returns_figure(preagg_boot):
    fig = plot_ridge(preagg_boot, ["a", "b"], show=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_feature_distribution(feature_stats, preagg_boot):
    fig = plot_feature_distribution(feature_stats, "a", preagg_boot, show=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize("approx", ["kde", "normal"])
def test_plot_feature_distribution_both_approx(preagg_boot, approx):
    stats = estimate_feature_level_mixture_preagg(preagg_boot, zero_tol=1e-8, approx=approx)
    fig = plot_feature_distribution(stats, "a", preagg_boot, show=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_top_features(feature_stats):
    fig, top_df = plot_top_features(feature_stats, top_k=3, show=False)
    assert isinstance(fig, Figure)
    assert len(top_df) == 3
    plt.close(fig)


def test_plot_top_features_density(feature_stats):
    fig, top_df = plot_top_features_density(feature_stats, top_k=2, show=False)
    assert isinstance(fig, Figure)
    assert len(top_df) == 2
    plt.close(fig)


def test_sample_level_plots(raw_boot):
    mix_df = estimate_sample_feature_distribution(raw_boot)
    fig = plot_sample_feature_distribution(
        mix_df, feature="a", sample_id=0, boot_results=raw_boot, show=False
    )
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig, top_df = plot_sample_top_features(mix_df, sample_id=0, top_k=2, xlim=(-3, 3), show=False)
    assert isinstance(fig, Figure)
    assert len(top_df) == 2
    plt.close(fig)


def test_plot_ridge_missing_feature_raises(preagg_boot):
    with pytest.raises(ValueError, match="No data found"):
        plot_ridge(preagg_boot, ["nope"], show=False)

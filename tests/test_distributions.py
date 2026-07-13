import numpy as np
import pandas as pd
import pytest

from roshap import (
    estimate_feature_level_mixture_preagg,
    fit_zero_inflated_kde,
    fit_zero_inflated_normal,
    zero_inflated_pdf,
)

REQUIRED_STAT_COLS = {
    "feature", "pi_zero", "p_nonzero", "median", "std",
    "nonzero_median", "sd_estimated", "mean_abs_estimated", "kde_model",
}


def _half_zero_values(rng, n=100, loc=3.0, scale=0.5):
    nz = rng.normal(loc, scale, size=n // 2)
    return np.concatenate([np.zeros(n // 2), np.abs(nz)])


def test_kde_pi_zero(rng):
    vals = _half_zero_values(rng)
    model = fit_zero_inflated_kde(vals, support="positive")
    assert model["dist"] == "kde"
    assert model["pi_zero"] == pytest.approx(0.5)
    assert model["kde"] is not None


def test_kde_pdf_integrates_to_continuous_mass(rng):
    vals = _half_zero_values(rng)
    model = fit_zero_inflated_kde(vals, bandwidth=0.2, support="positive")
    x = np.linspace(1e-6, 10, 5000)
    y = zero_inflated_pdf(x, model)
    mass = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    assert mass == pytest.approx(1.0 - model["pi_zero"], abs=0.05)


def test_normal_fit_recovers_moments(rng):
    nz = rng.normal(2.0, 0.7, size=4000)
    model = fit_zero_inflated_normal(nz, support="real")
    assert model["dist"] == "normal"
    assert model["pi_zero"] == pytest.approx(0.0)
    assert model["mu"] == pytest.approx(2.0, abs=0.05)
    assert model["sigma"] == pytest.approx(0.7, abs=0.05)


def test_normal_positive_support_is_lognormal(rng):
    vals = np.exp(rng.normal(1.0, 0.3, size=4000))
    model = fit_zero_inflated_normal(vals, support="positive")
    assert model["mu"] == pytest.approx(1.0, abs=0.05)
    assert model["sigma"] == pytest.approx(0.3, abs=0.05)
    # induced density must be zero at/below zero and integrate to 1 - pi_zero
    x = np.linspace(1e-6, 30, 8000)
    y = zero_inflated_pdf(x, model)
    mass = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    assert mass == pytest.approx(1.0, abs=0.02)
    assert zero_inflated_pdf(np.array([-1.0]), model)[0] == 0.0


def test_normal_pdf_zero_inflated(rng):
    vals = _half_zero_values(rng)
    model = fit_zero_inflated_normal(vals, support="positive")
    x = np.linspace(1e-6, 10, 5000)
    y = zero_inflated_pdf(x, model)
    mass = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    assert mass == pytest.approx(0.5, abs=0.02)


def _fake_boot_results(rng, n_boot=20, features=("a", "b")):
    frames = []
    for b in range(n_boot):
        frames.append(pd.DataFrame({
            "feature": list(features),
            "perm_round": 0,
            "sum_shap": rng.normal(size=len(features)),
            "sum_abs_shap": np.abs(rng.normal(5.0, 1.0, size=len(features))),
            "mean_abs_shap": np.abs(rng.normal(0.5, 0.1, size=len(features))),
            "n_samples": 10,
            "bootstrap_id": b,
            "boot_random_state": b,
        }))
    return frames


@pytest.mark.parametrize("approx", ["kde", "normal"])
def test_preagg_returns_required_columns(rng, approx):
    boot = _fake_boot_results(rng)
    stats = estimate_feature_level_mixture_preagg(boot, zero_tol=1e-8, approx=approx)
    assert REQUIRED_STAT_COLS.issubset(stats.columns)
    assert len(stats) == 2
    assert np.isfinite(stats["sd_estimated"]).all()
    assert (stats["p_nonzero"] > 0.9).all()
    for model in stats["kde_model"]:
        assert model["dist"] == approx


def test_preagg_kde_median_matches_empirical(rng):
    boot = _fake_boot_results(rng)
    stats = estimate_feature_level_mixture_preagg(boot, zero_tol=1e-8, approx="kde")
    concat = pd.concat(boot, ignore_index=True)
    for _, row in stats.iterrows():
        vals = concat.loc[concat["feature"] == row["feature"], "sum_abs_shap"]
        assert row["median"] == pytest.approx(float(np.median(vals)))
        assert row["nonzero_median"] == pytest.approx(float(np.median(vals)))


def test_bad_approx_raises(rng):
    boot = _fake_boot_results(rng, n_boot=3)
    with pytest.raises(ValueError, match="approx"):
        estimate_feature_level_mixture_preagg(boot, approx="cauchy")

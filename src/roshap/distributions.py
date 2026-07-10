"""Zero-inflated distribution estimation for bootstrap SHAP aggregates.

Each feature's bootstrap SHAP-magnitude distribution is modelled as a
zero-inflated mixture:

    P(X = 0) = pi_zero
    f(X | X != 0) = continuous density (KDE or Normal)

With ``support="positive"`` the continuous part is fit on log(nonzero) values,
so the induced density has support on x > 0 only (KDE -> log-KDE,
Normal -> lognormal).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from sklearn.neighbors import KernelDensity

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _concat_and_prepare(boot_results, group_cols_for_cat):
    """Concat list of DataFrames and convert object cols to categorical."""
    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()
    x = pd.concat(boot_results, ignore_index=True)
    for col in group_cols_for_cat:
        if col is not None and col in x.columns and x[col].dtype == "object":
            x[col] = x[col].astype("category")
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_zero_inflated_kde(
    values,
    bandwidth=0.2,
    kernel="gaussian",
    zero_tol=0.0,
    support="real",
):
    """Fit a zero-inflated KDE model on a 1D array-like of values.

    The model is:
      P(X = 0) = pi_zero
      f(X | X != 0) estimated via sklearn.neighbors.KernelDensity

        support:
            - "real": KDE is fit directly on the nonzero values
            - "positive": KDE is fit on log(nonzero values) so the induced density
                has support on x > 0 only
    """
    if support not in {"real", "positive"}:
        raise ValueError("support must be 'real' or 'positive'")

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    n_total = int(arr.size)
    if n_total == 0:
        return {
            "dist": "kde",
            "pi_zero": np.nan,
            "n_total": 0,
            "n_zero": 0,
            "n_nonzero": 0,
            "zero_tol": float(zero_tol),
            "bandwidth": float(bandwidth),
            "kernel": kernel,
            "support": support,
            "nonzero_min": np.nan,
            "nonzero_max": np.nan,
            "kde": None,
        }

    if support == "positive":
        is_zero = arr <= zero_tol
    else:
        is_zero = np.abs(arr) <= zero_tol

    n_zero = int(is_zero.sum())
    n_nonzero = int(n_total - n_zero)
    pi_zero = n_zero / n_total

    kde = None
    nonzero_min = np.nan
    nonzero_max = np.nan
    nonzero_empirical_median = np.nan
    if n_nonzero > 0:
        nz = arr[~is_zero]
        nonzero_min = float(np.min(nz))
        nonzero_max = float(np.max(nz))
        nonzero_empirical_median = float(np.median(nz))
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        if support == "positive":
            kde.fit(np.log(nz).reshape(-1, 1))
        else:
            kde.fit(nz.reshape(-1, 1))

    return {
        "dist": "kde",
        "pi_zero": float(pi_zero),
        "n_total": n_total,
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "zero_tol": float(zero_tol),
        "bandwidth": float(bandwidth),
        "kernel": kernel,
        "support": support,
        "nonzero_min": nonzero_min,
        "nonzero_max": nonzero_max,
        "nonzero_empirical_median": nonzero_empirical_median,
        "kde": kde,
    }


def fit_zero_inflated_normal(
    values,
    zero_tol=0.0,
    support="real",
):
    """Fit a zero-inflated Normal model on a 1D array-like of values.

    Analogue of :func:`fit_zero_inflated_kde` where the continuous part is a
    parametric Normal instead of a KDE:

        support:
            - "real": Normal(mu, sigma) fit on the nonzero values
            - "positive": Normal fit on log(nonzero values), i.e. a lognormal
                density on x > 0

    Returns the same dict structure as :func:`fit_zero_inflated_kde` with
    ``dist="normal"``, ``mu``/``sigma`` parameters, and ``kde=None``.
    """
    if support not in {"real", "positive"}:
        raise ValueError("support must be 'real' or 'positive'")

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    n_total = int(arr.size)
    if n_total == 0:
        return {
            "dist": "normal",
            "pi_zero": np.nan,
            "n_total": 0,
            "n_zero": 0,
            "n_nonzero": 0,
            "zero_tol": float(zero_tol),
            "bandwidth": np.nan,
            "kernel": None,
            "support": support,
            "nonzero_min": np.nan,
            "nonzero_max": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "kde": None,
        }

    if support == "positive":
        is_zero = arr <= zero_tol
    else:
        is_zero = np.abs(arr) <= zero_tol

    n_zero = int(is_zero.sum())
    n_nonzero = int(n_total - n_zero)
    pi_zero = n_zero / n_total

    mu = np.nan
    sigma = np.nan
    nonzero_min = np.nan
    nonzero_max = np.nan
    nonzero_empirical_median = np.nan
    if n_nonzero > 0:
        nz = arr[~is_zero]
        nonzero_min = float(np.min(nz))
        nonzero_max = float(np.max(nz))
        nonzero_empirical_median = float(np.median(nz))
        fit_vals = np.log(nz) if support == "positive" else nz
        mu = float(np.mean(fit_vals))
        sigma = float(np.std(fit_vals, ddof=1)) if n_nonzero > 1 else 0.0
        sigma = max(sigma, 1e-12)

    return {
        "dist": "normal",
        "pi_zero": float(pi_zero),
        "n_total": n_total,
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "zero_tol": float(zero_tol),
        "bandwidth": np.nan,
        "kernel": None,
        "support": support,
        "nonzero_min": nonzero_min,
        "nonzero_max": nonzero_max,
        "nonzero_empirical_median": nonzero_empirical_median,
        "mu": mu,
        "sigma": sigma,
        "kde": None,
    }


def _fit_zero_inflated(values, bandwidth, kernel, zero_tol, support, approx):
    """Dispatch to the KDE or Normal fitter based on *approx*."""
    if approx == "kde":
        return fit_zero_inflated_kde(
            values, bandwidth=bandwidth, kernel=kernel, zero_tol=zero_tol, support=support
        )
    if approx == "normal":
        return fit_zero_inflated_normal(values, zero_tol=zero_tol, support=support)
    raise ValueError("approx must be 'kde' or 'normal'")


def zero_inflated_kde_pdf(x, model, include_zero_spike=False):
    """Evaluate density-like values for a fitted zero-inflated KDE model.

    If include_zero_spike is False, returns the continuous part:
        (1 - pi_zero) * f_kde(x)
    If include_zero_spike is True, values with |x| <= zero_tol get np.inf to
    indicate a point mass at zero in the full mixed distribution.
    """
    x_arr = np.asarray(x, dtype=float)
    out = np.full(x_arr.shape, np.nan, dtype=float)

    if model is None or np.isnan(model.get("pi_zero", np.nan)):
        return out

    pi_zero = float(model["pi_zero"])
    kde = model.get("kde")
    zero_tol = float(model.get("zero_tol", 0.0))
    support = model.get("support", "real")

    if support not in {"real", "positive"}:
        raise ValueError("model support must be 'real' or 'positive'")

    if include_zero_spike:
        if support == "positive":
            out[x_arr <= zero_tol] = np.inf
        else:
            out[np.abs(x_arr) <= zero_tol] = np.inf

    if kde is None:
        # All mass is at zero, or no non-zero data to fit KDE.
        if support == "positive":
            mask_nonzero = x_arr > zero_tol
        else:
            mask_nonzero = np.abs(x_arr) > zero_tol
        out[mask_nonzero] = 0.0
        return out

    if support == "positive":
        cont = np.zeros_like(x_arr, dtype=float)
        mask_pos = x_arr > zero_tol
        if np.any(mask_pos):
            z = np.log(x_arr[mask_pos]).reshape(-1, 1)
            cont[mask_pos] = (1.0 - pi_zero) * np.exp(kde.score_samples(z)) / x_arr[mask_pos]
    else:
        x2 = x_arr.reshape(-1, 1)
        cont = (1.0 - pi_zero) * np.exp(kde.score_samples(x2))

    # Keep the explicit point-mass marker where requested.
    if include_zero_spike:
        if support == "positive":
            mask_nonzero = x_arr > zero_tol
        else:
            mask_nonzero = np.abs(x_arr) > zero_tol
        out[mask_nonzero] = cont[mask_nonzero]
    else:
        out = cont

    return out


def _zero_inflated_normal_pdf(x, model, include_zero_spike=False):
    """Evaluate density-like values for a fitted zero-inflated Normal model."""
    x_arr = np.asarray(x, dtype=float)
    out = np.full(x_arr.shape, np.nan, dtype=float)

    if model is None or np.isnan(model.get("pi_zero", np.nan)):
        return out

    pi_zero = float(model["pi_zero"])
    zero_tol = float(model.get("zero_tol", 0.0))
    support = model.get("support", "real")
    mu = float(model.get("mu", np.nan))
    sigma = float(model.get("sigma", np.nan))

    if support not in {"real", "positive"}:
        raise ValueError("model support must be 'real' or 'positive'")

    if include_zero_spike:
        if support == "positive":
            out[x_arr <= zero_tol] = np.inf
        else:
            out[np.abs(x_arr) <= zero_tol] = np.inf

    if support == "positive":
        mask_nonzero = x_arr > zero_tol
    else:
        mask_nonzero = np.abs(x_arr) > zero_tol

    if not (np.isfinite(mu) and np.isfinite(sigma)):
        # No nonzero data was available to fit the Normal component.
        out[mask_nonzero] = 0.0
        return out

    if support == "positive":
        cont = np.zeros_like(x_arr, dtype=float)
        mask_pos = x_arr > zero_tol
        if np.any(mask_pos):
            cont[mask_pos] = (1.0 - pi_zero) * lognorm.pdf(
                x_arr[mask_pos], s=sigma, scale=np.exp(mu)
            )
    else:
        cont = (1.0 - pi_zero) * norm.pdf(x_arr, loc=mu, scale=sigma)

    if include_zero_spike:
        out[mask_nonzero] = cont[mask_nonzero]
    else:
        out = cont

    return out


def zero_inflated_pdf(x, model, include_zero_spike=False):
    """Evaluate the continuous density of any fitted zero-inflated model.

    Dispatches on ``model["dist"]`` ("kde" or "normal"); models without a
    ``dist`` key are treated as KDE for backward compatibility.
    """
    if isinstance(model, dict) and model.get("dist", "kde") == "normal":
        return _zero_inflated_normal_pdf(x, model, include_zero_spike=include_zero_spike)
    return zero_inflated_kde_pdf(x, model, include_zero_spike=include_zero_spike)


def _model_grid_width(model):
    """Half-width term used to pad the integration grid past the data range."""
    if model.get("dist", "kde") == "normal":
        mu = float(model.get("mu", np.nan))
        sigma = float(model.get("sigma", np.nan))
        if not (np.isfinite(mu) and np.isfinite(sigma)):
            return 0.2
        if model.get("support", "real") == "positive":
            # Standard deviation of the induced lognormal on the original scale.
            spread = float(
                np.sqrt(max(np.exp(sigma**2) - 1.0, 0.0)) * np.exp(mu + sigma**2 / 2.0)
            )
        else:
            spread = sigma
        return max(spread, 1e-3)
    return float(model.get("bandwidth", 0.2))


def _summarize_zero_inflated_model(model, n_grid=400):
    """Compute original-scale summary metrics for a zero-inflated model."""
    if not isinstance(model, dict):
        return {
            "mean_abs_estimated": 0.0,
            "mean_estimated": 0.0,
            "var_estimated": 0.0,
            "sd_estimated": 0.0,
            "p_nonzero": 0.0,
            "peak_density": 0.0,
            "nonzero_median": 0.0,
            "nonzero_median_kde": 0.0,
        }

    pi_zero = float(model.get("pi_zero", np.nan))
    p_nonzero = 0.0 if np.isnan(pi_zero) else max(0.0, 1.0 - pi_zero)

    support = model.get("support", "real")
    xmin = float(model.get("nonzero_min", np.nan))
    xmax = float(model.get("nonzero_max", np.nan))
    bw = _model_grid_width(model)

    if support == "positive":
        hi = xmax + 4.0 * bw if np.isfinite(xmax) else 1.0
        hi = max(hi, (xmin if np.isfinite(xmin) else 0.0) + 1e-3)
        x_grid = np.linspace(0.0, hi, n_grid)
    else:
        lo = (xmin - 4.0 * bw) if np.isfinite(xmin) else -1.0
        hi = (xmax + 4.0 * bw) if np.isfinite(xmax) else 1.0
        if hi <= lo:
            lo, hi = -1.0, 1.0
        x_grid = np.linspace(lo, hi, n_grid)

    y = zero_inflated_pdf(x_grid, model, include_zero_spike=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return {
            "mean_abs_estimated": 0.0,
            "mean_estimated": 0.0,
            "var_estimated": 0.0,
            "sd_estimated": 0.0,
            "p_nonzero": p_nonzero,
            "peak_density": 0.0,
            "nonzero_median": 0.0,
            "nonzero_median_kde": 0.0,
        }

    mean_abs = max(float(_trapz(np.abs(x_grid) * y, x_grid)), 0.0)
    mean_val = float(_trapz(x_grid * y, x_grid))
    second_moment = float(_trapz((x_grid ** 2) * y, x_grid))
    var_val = max(second_moment - mean_val ** 2, 0.0)
    sd_val = float(np.sqrt(var_val))
    peak = float(np.max(y)) if y.size else 0.0

    cont_mass = float(_trapz(y, x_grid))
    if cont_mass > 0:
        y_cond = y / cont_mass
        dx = np.diff(x_grid)
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (y_cond[:-1] + y_cond[1:]) * dx)))
        cdf = np.clip(cdf, 0.0, 1.0)
        nonzero_median_kde = float(np.interp(0.5, cdf, x_grid))
    else:
        nonzero_median_kde = 0.0

    empirical_median = model.get("nonzero_empirical_median", np.nan)
    if np.isfinite(empirical_median):
        nonzero_median = float(empirical_median)
    else:
        nonzero_median = 0.0

    return {
        "mean_abs_estimated": mean_abs,
        "mean_estimated": mean_val,
        "var_estimated": var_val,
        "sd_estimated": sd_val,
        "p_nonzero": p_nonzero,
        "peak_density": peak,
        "nonzero_median": nonzero_median,
        "nonzero_median_kde": nonzero_median_kde,
    }


def estimate_sample_feature_distribution(
        boot_results,
        sample_cols=None,
        group_cols=None,
        value_col="shap_value",
        bandwidth=0.2,
        kernel="gaussian",
        zero_tol=0.0,
        approx="kde",
    ):
    """Fit a zero-inflated model for each sample-feature(-class) group.

    Expects raw bootstrap rows (e.g. from boot_multi_repeat_inference_keep_all)
    with at least sample_id/feature[/class_id] and shap_value columns.
    """
    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()

    first = boot_results[0] if boot_results else pd.DataFrame()

    if group_cols is None:
        group_cols = (
            ("sample_id", "class_id", "feature")
            if "class_id" in first.columns
            else ("sample_id", "feature")
        )

    x = _concat_and_prepare(boot_results, group_cols)
    if x.empty:
        return pd.DataFrame()

    required = set(group_cols) | {value_col}
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sample_cols = tuple(sample_cols) if sample_cols is not None else tuple()
    extra_cols = [c for c in sample_cols if c in x.columns and c not in group_cols]

    rows = []
    grouped = x.groupby(list(group_cols), as_index=False, sort=False, observed=True)
    for key, g in grouped:
        key_vals = key if isinstance(key, tuple) else (key,)
        base = {c: v for c, v in zip(group_cols, key_vals)}
        for c in extra_cols:
            base[c] = g[c].iloc[0]

        model = _fit_zero_inflated(
            values=g[value_col].to_numpy(),
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support="real",
            approx=approx,
        )

        rows.append({
            **base,
            "n_total": model["n_total"],
            "n_zero": model["n_zero"],
            "n_nonzero": model["n_nonzero"],
            "pi_zero": model["pi_zero"],
            "bandwidth": model["bandwidth"],
            "kernel": model["kernel"],
            "zero_tol": model["zero_tol"],
            "kde_model": model,
        })

    return pd.DataFrame(rows)


def estimate_feature_level_mixture(
    boot_results,
    group_cols=None,
    value_col="shap_value",
    bandwidth=0.2,
    kernel="gaussian",
    zero_tol=0.0,
    approx="kde",
):
    """Fit a zero-inflated model on aggregated |SHAP| per (bootstrap, perm_round, feature).

    For each (bootstrap_id, perm_round, feature) combination (and optionally class_id),
    this function:
      1. Sums the absolute values of SHAP across all samples
      2. Fits a zero-inflated model to the resulting aggregated values

    Returns a DataFrame with one row per (feature, [class_id])
    containing statistics and the fitted model dict (in the "kde_model" column,
    regardless of the chosen approximation).
    """
    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()

    first = boot_results[0] if boot_results else pd.DataFrame()

    # Determine grouping columns based on data structure
    if group_cols is None:
        has_class = "class_id" in first.columns
        group_cols = (
            ("bootstrap_id", "perm_round", "class_id", "feature")
            if has_class
            else ("bootstrap_id", "perm_round", "feature")
        )

    # Concatenate all bootstrap results
    x = _concat_and_prepare(boot_results, group_cols)
    if x.empty:
        return pd.DataFrame()

    # Check required columns
    required = set(group_cols) | {value_col}
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = x.copy()
    x["_abs_shap"] = x[value_col].abs()

    aggregated = (
        x.groupby(list(group_cols), as_index=False, sort=False, observed=True)
        .agg(
            agg_shap=("_abs_shap", "sum"),
            n_samples=(value_col, "count"),
        )
    )

    # Determine feature group columns (for fitting the model per feature)
    has_class = "class_id" in group_cols
    if has_class:
        feat_group_cols = ["feature", "class_id"]
    else:
        feat_group_cols = ["feature"]

    # Fit zero-inflated model for each feature (aggregating across bootstrap/perm_round)
    rows = []
    feat_groups = aggregated.groupby(feat_group_cols, as_index=False, sort=False, observed=True)

    for _, feat_df in feat_groups:
        # Extract feature and class_id from the grouped data
        base = {col: feat_df[col].iloc[0] for col in feat_group_cols}

        # Fit zero-inflated model on the aggregated SHAP values for this feature
        agg_values = feat_df["agg_shap"].to_numpy()

        model = _fit_zero_inflated(
            values=agg_values,
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support="positive",
            approx=approx,
        )

        summary = _summarize_zero_inflated_model(model)

        rows.append({
            **base,
            "n_bootstrap_rounds": len(feat_df),
            "n_total": model["n_total"],
            "n_zero": model["n_zero"],
            "n_nonzero": model["n_nonzero"],
            "pi_zero": model["pi_zero"],
            "bandwidth": model["bandwidth"],
            "kernel": model["kernel"],
            "zero_tol": model["zero_tol"],
            "kde_model": model,
            **summary,
        })

    return pd.DataFrame(rows)


def estimate_feature_level_mixture_preagg(
    boot_results,
    agg_col="sum_abs_shap",
    bandwidth=0.2,
    kernel="gaussian",
    zero_tol=0.0,
    support="positive",
    approx="kde",
):
    """Fit a zero-inflated model per feature from pre-aggregated bootstrap results.

    Use this when boot_results come from ``boot_multi_repeat_inference_keep_feature``.
    If the requested *agg_col* is ``"mean_abs_shap"`` and unavailable, the function
    automatically falls back to ``sum_abs_shap``.

    Parameters
    ----------
    boot_results : list of pd.DataFrame
        Each DataFrame must contain ``feature``, ``perm_round``, ``bootstrap_id``,
        ``sum_abs_shap`` (or the column named by *agg_col*), and optionally
        ``class_id``.
    agg_col : str
        Column holding the pre-aggregated SHAP value (default ``"sum_abs_shap"``).
    bandwidth, kernel, zero_tol
        Passed to the underlying fitter (bandwidth/kernel apply to KDE only).
    support : {"positive", "real"}
        Support of the continuous component. Use "positive" for nonnegative
        aggregates like |SHAP| sums, and "real" for signed sums.
    approx : {"kde", "normal"}
        Continuous-density estimator: zero-inflated KDE (default) or a
        zero-inflated Normal approximation (lognormal when support="positive").
    """
    if support not in {"positive", "real"}:
        raise ValueError("support must be 'positive' or 'real'")

    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()

    first = boot_results[0]
    has_class = "class_id" in first.columns

    aggregated = pd.concat(boot_results, ignore_index=True)

    if agg_col not in aggregated.columns:
        if agg_col == "mean_abs_shap" and "sum_abs_shap" in aggregated.columns:
            agg_col = "sum_abs_shap"
        else:
            raise ValueError(
                f"Column '{agg_col}' not found in boot_results. "
                f"Available columns include: {sorted(aggregated.columns.tolist())}"
            )

    if "feature" in aggregated.columns and aggregated["feature"].dtype == "object":
        aggregated["feature"] = aggregated["feature"].astype("category")
    if has_class and aggregated["class_id"].dtype == "object":
        aggregated["class_id"] = aggregated["class_id"].astype("category")

    feat_group_cols = ["feature", "class_id"] if has_class else ["feature"]

    rows = []
    for key, feat_df in aggregated.groupby(feat_group_cols, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        base = {c: v for c, v in zip(feat_group_cols, key)}

        agg_values = feat_df[agg_col].to_numpy()

        model = _fit_zero_inflated(
            values=agg_values,
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support=support,
            approx=approx,
        )

        summary = _summarize_zero_inflated_model(model)

        rows.append({
            **base,
            "n_bootstrap_rounds": len(feat_df),
            "n_total": model["n_total"],
            "n_zero": model["n_zero"],
            "n_nonzero": model["n_nonzero"],
            "pi_zero": model["pi_zero"],
            "bandwidth": model["bandwidth"],
            "kernel": model["kernel"],
            "zero_tol": model["zero_tol"],
            "kde_model": model,
            "median": float(np.median(agg_values)),
            "std": float(np.std(agg_values, ddof=1)),
            **summary,
        })

    return pd.DataFrame(rows)


def estimate_feature_level_mixture_fast(
    boot_results,
    group_cols=None,
    value_col="shap_value",
    bandwidth=0.2,
    kernel="gaussian",
    zero_tol=0.0,
    support="positive",
    approx="kde",
):
    """Lower-memory variant of estimate_feature_level_mixture (chunked aggregation)."""
    if support not in {"positive", "real"}:
        raise ValueError("support must be 'positive' or 'real'")

    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()

    first = boot_results[0]
    if group_cols is None:
        has_class = "class_id" in first.columns
        group_cols = (
            ("bootstrap_id", "perm_round", "class_id", "feature")
            if has_class
            else ("bootstrap_id", "perm_round", "feature")
        )

    pieces = []
    needed = list(set(group_cols) | {value_col})

    for df in boot_results:
        cols = [c for c in needed if c in df.columns]
        tmp = df[cols].copy()

        if tmp.empty:
            continue

        if "feature" in tmp.columns and tmp["feature"].dtype == "object":
            tmp["feature"] = tmp["feature"].astype("category")
        if "class_id" in tmp.columns and tmp["class_id"].dtype == "object":
            tmp["class_id"] = tmp["class_id"].astype("category")

        tmp["_abs_shap"] = tmp[value_col].abs()

        agg = (
            tmp.groupby(list(group_cols), as_index=False, sort=False, observed=True)
               .agg(
                   agg_shap=("_abs_shap", "sum"),
                   n_samples=(value_col, "count"),
               )
        )
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame()

    aggregated = pd.concat(pieces, ignore_index=True)

    feat_group_cols = ["feature", "class_id"] if "class_id" in group_cols else ["feature"]

    rows = []
    for key, feat_df in aggregated.groupby(feat_group_cols, sort=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        base = {c: v for c, v in zip(feat_group_cols, key)}

        agg_values = feat_df["agg_shap"].to_numpy()

        model = _fit_zero_inflated(
            values=agg_values,
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support=support,
            approx=approx,
        )

        summary = _summarize_zero_inflated_model(model)

        rows.append({
            **base,
            "n_bootstrap_rounds": len(feat_df),
            "n_total": model["n_total"],
            "n_zero": model["n_zero"],
            "n_nonzero": model["n_nonzero"],
            "pi_zero": model["pi_zero"],
            "bandwidth": model["bandwidth"],
            "kernel": model["kernel"],
            "zero_tol": model["zero_tol"],
            "kde_model": model,
            **summary,
        })

    return pd.DataFrame(rows)

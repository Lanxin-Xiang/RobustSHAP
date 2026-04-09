from scipy.stats import gamma
from sklearn.neighbors import KernelDensity

import numpy as np
import pandas as pd


def _concat_and_prepare(boot_results, group_cols_for_cat):
    """Concat list of DataFrames and convert object cols to categorical."""
    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        return pd.DataFrame()
    x = pd.concat(boot_results, ignore_index=True, copy=False)
    for col in group_cols_for_cat:
        if col is not None and col in x.columns and x[col].dtype == "object":
            x[col] = x[col].astype("category")
    return x


def _compute_nz_sufficient_stats(x, var_col):
    """
    Add nz_sum and nz_sum_sq columns to x using numpy.
    These are the sufficient statistics for the nonzero normal component.
    """
    nz_mean = x["nonzero_mean"].to_numpy(dtype=float, na_value=0.0)
    nz_count = x["nonzero_count"].to_numpy(dtype=float)
    var_vals = x[var_col].to_numpy(dtype=float, na_value=0.0)

    if var_col.endswith("_sample"):
        within_ss = np.where(nz_count > 1, (nz_count - 1) * var_vals, 0.0)
    else:
        within_ss = nz_count * var_vals

    x["nz_sum"] = nz_count * nz_mean
    x["nz_sum_sq"] = within_ss + nz_count * (nz_mean ** 2)
    return x


def _moments_from_sufficient_stats(total_eval, total_zero, total_nz, nz_sum, nz_sum_sq):
    """
    Vectorised computation of (pi_zero, mu_nz, sigma2_nz).
    All inputs are numpy arrays. Returns NaN where undefined.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_zero = np.where(total_eval > 0, total_zero / total_eval, np.nan)
        mu_nz = np.where(total_nz > 0, nz_sum / total_nz, np.nan)
        sigma2_nz = np.where(
            total_nz > 0,
            nz_sum_sq / total_nz - np.where(total_nz > 0, mu_nz, 0.0) ** 2,
            np.nan,
        )
    sigma2_nz = np.clip(sigma2_nz, 0, None)
    return pi_zero, mu_nz, sigma2_nz


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
    if n_nonzero > 0:
        nz = arr[~is_zero]
        nonzero_min = float(np.min(nz))
        nonzero_max = float(np.max(nz))
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        if support == "positive":
            kde.fit(np.log(nz).reshape(-1, 1))
        else:
            kde.fit(nz.reshape(-1, 1))

    return {
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
        "kde": kde,
    }


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


def estimate_sample_feature_distribution(
        boot_results,
        sample_cols=None,
        group_cols=None,
        value_col="shap_value",
        bandwidth=0.2,
        kernel="gaussian",
        zero_tol=0.0,
    ):
    """Fit zero-inflated KDE for each sample-feature(-class) group.

    Expects raw bootstrap rows produced by bootstrap_v3.py with at least
    sample_id/feature[/class_id] and shap_value columns.
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

        model = fit_zero_inflated_kde(
            values=g[value_col].to_numpy(),
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support="real",
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
):
    """Fit zero-inflated KDE on aggregated |SHAP| per (bootstrap, perm_round, feature).

    For each (bootstrap_id, perm_round, feature) combination (and optionally class_id),
    this function:
      1. Sums the absolute values of SHAP across all samples
      2. Fits a zero-inflated KDE model to the resulting aggregated values

    Returns a DataFrame with one row per (feature, [class_id])
    containing statistics and the fitted KDE model.
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

    # Aggregate: group by (bootstrap_id, perm_round, feature, [class_id])
    # and sum |SHAP| across all samples in each group
    aggregated = x.groupby(list(group_cols), as_index=False, sort=False, observed=True).agg(
        agg_shap=("shap_value", lambda v: np.sum(np.abs(v))),
        n_samples=(value_col, "count"),
    )

    # Determine feature group columns (for fitting KDE per feature)
    has_class = "class_id" in group_cols
    if has_class:
        feat_group_cols = ["feature", "class_id"]
    else:
        feat_group_cols = ["feature"]

    # Fit zero-inflated KDE for each feature (aggregating across bootstrap/perm_round)
    rows = []
    feat_groups = aggregated.groupby(feat_group_cols, as_index=False, sort=False, observed=True)

    for _, feat_df in feat_groups:
        # Extract feature and class_id from the grouped data
        base = {col: feat_df[col].iloc[0] for col in feat_group_cols}

        # Fit zero-inflated KDE on the aggregated SHAP values for this feature
        agg_values = feat_df["agg_shap"].to_numpy()
        
        model = fit_zero_inflated_kde(
            values=agg_values,
            bandwidth=bandwidth,
            kernel=kernel,
            zero_tol=zero_tol,
            support="positive",
        )

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
        })

    return pd.DataFrame(rows)


def estimate_feature_level_sq_from_bootstrap_draws(
    boot_results,
    sample_col="sample_id",
    feature_col="feature",
    class_col=None,
    bootstrap_col="bootstrap_id",
    use_sample_var=True,
    n_total_samples=None,
    zero_threshold=1e-12,
    return_boot_draws=False,
):
    """
    Estimate the distribution of
        A_j = (1/N) * sum_i X_ij^2
    by constructing one draw A_j^(b) per bootstrap b.

    This fixes the p_nonzero issue by estimating P(A_j > 0)
    from the empirical bootstrap distribution of A_j itself.
    """
    if not isinstance(boot_results, list):
        raise TypeError(f"boot_results must be a list of DataFrames, got {type(boot_results)}")
    if len(boot_results) == 0:
        raise ValueError("boot_results is an empty list.")

    x = pd.concat(boot_results, ignore_index=True, copy=False)

    if class_col is None and "class_id" in x.columns:
        class_col = "class_id"

    required = {
        bootstrap_col, sample_col, feature_col,
        "n_evaluated", "zero_count", "nonzero_count",
        "nonzero_mean",
        "nonzero_var_sample" if use_sample_var else "nonzero_var",
    }
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    var_col = "nonzero_var_sample" if use_sample_var else "nonzero_var"

    # reconstruct sufficient stats within each row
    nz_mean = x["nonzero_mean"].to_numpy(dtype=float, na_value=0.0)
    nz_count = x["nonzero_count"].to_numpy(dtype=float)
    var_vals = x[var_col].to_numpy(dtype=float, na_value=0.0)

    if use_sample_var:
        within_ss = np.where(nz_count > 1, (nz_count - 1) * var_vals, 0.0)
    else:
        within_ss = nz_count * var_vals

    x = x.copy()
    x["nz_sum"] = nz_count * nz_mean
    x["nz_sum_sq"] = within_ss + nz_count * (nz_mean ** 2)

    # aggregate within bootstrap × sample × feature (and class if needed)
    group_cols = [bootstrap_col, sample_col, feature_col]
    if class_col is not None:
        group_cols.insert(2, class_col)

    sf = x.groupby(group_cols, as_index=False, sort=False, observed=True).agg(
        total_evaluated=("n_evaluated", "sum"),
        total_zero=("zero_count", "sum"),
        total_nonzero=("nonzero_count", "sum"),
        nz_sum=("nz_sum", "sum"),
        nz_sum_sq=("nz_sum_sq", "sum"),
    )

    total_eval = sf["total_evaluated"].to_numpy(dtype=float)
    total_zero = sf["total_zero"].to_numpy(dtype=float)
    total_nz = sf["total_nonzero"].to_numpy(dtype=float)
    sum_val = sf["nz_sum"].to_numpy(dtype=float)
    sum_sq = sf["nz_sum_sq"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        pi_zero = np.where(total_eval > 0, total_zero / total_eval, np.nan)
        mu_nz = np.where(total_nz > 0, sum_val / total_nz, np.nan)
        sigma2_nz = np.where(total_nz > 0, sum_sq / total_nz - np.where(total_nz > 0, mu_nz, 0.0) ** 2, np.nan)

    sigma2_nz = np.clip(sigma2_nz, 0, None)

    pi = np.nan_to_num(pi_zero, nan=1.0)
    mu = np.nan_to_num(mu_nz, nan=0.0)
    sigma2 = np.nan_to_num(sigma2_nz, nan=0.0)

    # sample-level E[X_ij^2 | bootstrap b]
    sf["mean_x2_ij"] = (1.0 - pi) * (mu**2 + sigma2)

    # total N in A_j = (1/N) sum_i X_ij^2
    N = x[sample_col].nunique() if n_total_samples is None else int(n_total_samples)

    # one draw A_j^(b) per bootstrap
    feat_group_cols = [bootstrap_col, feature_col]
    if class_col is not None:
        feat_group_cols.insert(1, class_col)

    boot_draws = sf.groupby(feat_group_cols, as_index=False, sort=False, observed=True).agg(
        sum_x2=("mean_x2_ij", "sum"),
        n_samples_used=(sample_col, "nunique"),
    )
    boot_draws["A_draw"] = boot_draws["sum_x2"] / N

    # summarize empirical distribution of A_j^(b)
    out_group_cols = [feature_col]
    if class_col is not None:
        out_group_cols.insert(0, class_col)

    def summarize_one_group(g):
        a = g["A_draw"].to_numpy(dtype=float)
        p_all_zero = np.mean(a <= zero_threshold)
        p_nonzero = 1.0 - p_all_zero

        pos = a[a > zero_threshold]

        mean_A = np.mean(a)
        var_A = np.var(a, ddof=1) if len(a) > 1 else 0.0
        sd_A = np.sqrt(var_A)

        if len(pos) == 0:
            pos_mean = 0.0
            pos_var = 0.0
            pos_sd = 0.0
            gamma_shape = np.nan
            gamma_scale = np.nan
            pos_median = 0.0
        else:
            pos_mean = np.mean(pos)
            pos_var = np.var(pos, ddof=1) if len(pos) > 1 else 0.0
            pos_sd = np.sqrt(pos_var)

            if pos_var > 0:
                gamma_shape = pos_mean**2 / pos_var
                gamma_scale = pos_var / pos_mean
                pos_median = gamma.ppf(0.5, a=gamma_shape, scale=gamma_scale)
            else:
                gamma_shape = np.inf
                gamma_scale = 0.0
                pos_median = pos_mean

        return pd.Series({
            "n_bootstrap_used": len(a),
            "N": N,
            "p_all_zero": p_all_zero,
            "p_nonzero": p_nonzero,
            "mean_A": mean_A,
            "var_A": var_A,
            "sd_A": sd_A,
            "pos_mean_A": pos_mean,
            "pos_var_A": pos_var,
            "pos_sd_A": pos_sd,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
            "pos_median_A": pos_median,
        })

    feat = boot_draws.groupby(out_group_cols, as_index=False, sort=False, observed=True).apply(summarize_one_group)
    feat = feat.reset_index(drop=True)

    if return_boot_draws:
        return feat, boot_draws
    return feat
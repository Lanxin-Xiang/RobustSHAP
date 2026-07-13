"""Distribution plots for bootstrap SHAP results.

All functions return the matplotlib Figure (functions that also compute a
top-features table return ``(fig, top_df)``) and accept ``show=True`` to
control whether ``plt.show()`` is called.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .distributions import zero_inflated_pdf, _model_grid_width


# ---------------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------------

def _get_density(model, xvals):
    """Continuous part (1 - pi_zero) * f(x) of a fitted zero-inflated model."""
    xvals = np.asarray(xvals, dtype=float)
    if not isinstance(model, dict):
        return np.zeros_like(xvals)
    y = zero_inflated_pdf(xvals, model, include_zero_spike=False)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _integration_grid(model, n_grid=400):
    if not isinstance(model, dict):
        return np.linspace(0.0, 1.0, n_grid)

    support = model.get("support", "real")
    xmax = float(model.get("nonzero_max", np.nan))
    xmin = float(model.get("nonzero_min", np.nan))
    bw = _model_grid_width(model)

    if support == "positive":
        hi = xmax + 4.0 * bw if np.isfinite(xmax) else 1.0
        hi = max(hi, (xmin if np.isfinite(xmin) else 0.0) + 1e-3)
        return np.linspace(0.0, hi, n_grid)

    lo = (xmin - 4.0 * bw) if np.isfinite(xmin) else -1.0
    hi = (xmax + 4.0 * bw) if np.isfinite(xmax) else 1.0
    if hi <= lo:
        lo, hi = -1.0, 1.0
    return np.linspace(lo, hi, n_grid)


def _model_summary_stats(model):
    if not isinstance(model, dict):
        return {
            "mean_abs_estimated": 0.0,
            "mean_estimated": 0.0,
            "var_estimated": 0.0,
            "sd_estimated": 0.0,
            "p_nonzero": 0.0,
            "peak_density": 0.0,
            "nonzero_median": 0.0,
        }

    pi_zero = float(model.get("pi_zero", np.nan))
    p_nonzero = 0.0 if np.isnan(pi_zero) else max(0.0, 1.0 - pi_zero)
    x_grid = _integration_grid(model)
    y = _get_density(model, x_grid)
    if y.size == 0:
        return {
            "mean_abs_estimated": 0.0,
            "mean_estimated": 0.0,
            "var_estimated": 0.0,
            "sd_estimated": 0.0,
            "p_nonzero": p_nonzero,
            "peak_density": 0.0,
            "nonzero_median": 0.0,
        }

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
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
        nonzero_median = float(np.interp(0.5, cdf, x_grid))
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
    }


def _finish(fig, show):
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Sample-level plots
# ---------------------------------------------------------------------------

def plot_sample_feature_distribution(
    mix_df,
    feature,
    sample_id=None,
    class_id=None,
    boot_results=None,
    shap_values=None,
    value_col="shap_value",
    show_zero_spike=True,
    spike_width=None,
    xlim=None,
    bins=50,
    n_grid=1000,
    title=None,
    show=True,
):
    """Plot estimated zero-inflated density and histogram for one group.

    Works in two modes based on columns available in ``mix_df``:
      - sample-feature level: selects by (sample_id, feature[, class_id])
      - feature-level aggregation: selects by (feature[, class_id])

    Overlays:
      1) histogram of non-zero SHAP values scaled by (1 - pi_zero),
      2) estimated continuous density from the fitted model,
      3) optional point mass spike at 0.
    """
    is_sample_level = "sample_id" in mix_df.columns

    if feature is None:
        raise ValueError("feature must be provided")

    if "feature" not in mix_df.columns:
        raise ValueError("mix_df must contain a 'feature' column")

    if is_sample_level:
        if sample_id is None:
            raise ValueError("sample_id is required when mix_df contains sample_id")
        sel = (mix_df["sample_id"] == sample_id) & (mix_df["feature"] == feature)
    else:
        sel = mix_df["feature"] == feature

    if class_id is not None:
        if "class_id" not in mix_df.columns:
            raise ValueError("class_id provided but class_id column not present in mix_df")
        sel = sel & (mix_df["class_id"] == class_id)

    row_df = mix_df.loc[sel]
    if row_df.empty:
        if is_sample_level:
            raise ValueError(
                f"No estimated density row for sample_id={sample_id}, feature={feature}, class_id={class_id}"
            )
        raise ValueError(
            f"No estimated density row for feature={feature}, class_id={class_id}"
        )
    if len(row_df) > 1:
        row_df = row_df.head(1)

    row = row_df.iloc[0]
    model = row.get("kde_model", None)
    pi_zero = float(row.get("pi_zero", np.nan))
    support = "real"
    if isinstance(model, dict):
        pi_zero = float(model.get("pi_zero", pi_zero))
        zero_tol = float(model.get("zero_tol", 0.0))
        support = model.get("support", "real")
    else:
        zero_tol = 0.0

    # Sample-feature estimates are always on real support.
    # Positive support is reserved for feature-level aggregated mixtures.
    if is_sample_level:
        support = "real"

    if shap_values is None:
        if boot_results is None:
            raise ValueError("Provide shap_values directly or provide boot_results to filter raw SHAP values.")

        if isinstance(boot_results, list):
            if len(boot_results) == 0:
                raise ValueError("boot_results is an empty list")
            raw_df = pd.concat(boot_results, ignore_index=True)
        elif isinstance(boot_results, pd.DataFrame):
            raw_df = boot_results
        else:
            raise TypeError(f"boot_results must be list[DataFrame] or DataFrame, got {type(boot_results)}")

        if value_col not in raw_df.columns:
            raise ValueError(f"{value_col} column not found in boot_results")

        raw_sel = raw_df["feature"] == feature
        if class_id is not None:
            if "class_id" not in raw_df.columns:
                raise ValueError("class_id provided but class_id column not present in boot_results")
            raw_sel = raw_sel & (raw_df["class_id"] == class_id)

        if is_sample_level:
            if "sample_id" not in raw_df.columns:
                raise ValueError("mix_df is sample-level but boot_results has no sample_id column")
            raw_sel = raw_sel & (raw_df["sample_id"] == sample_id)
            vals_all = raw_df.loc[raw_sel, value_col].to_numpy(dtype=float)
        else:
            sub_df = raw_df.loc[raw_sel].copy()
            if sub_df.empty:
                vals_all = np.array([], dtype=float)
            elif {"bootstrap_id", "perm_round"}.issubset(sub_df.columns):
                # Match estimate_feature_level_mixture: aggregate |SHAP| per (bootstrap, perm_round, feature[, class]).
                group_cols = ["bootstrap_id", "perm_round", "feature"]
                if "class_id" in sub_df.columns:
                    group_cols.append("class_id")
                agg = (
                    sub_df.groupby(group_cols, as_index=False, sort=False, observed=True)[value_col]
                    .apply(lambda v: np.sum(np.abs(v)))
                )
                vals_all = agg.to_numpy(dtype=float)
            else:
                # Fallback when bootstrap/round columns are unavailable.
                vals_all = np.abs(sub_df[value_col].to_numpy(dtype=float))
    else:
        vals_all = np.asarray(shap_values, dtype=float).reshape(-1)

    vals_all = vals_all[np.isfinite(vals_all)]
    if vals_all.size == 0:
        raise ValueError("No finite SHAP values available for the selected group")

    if support == "positive":
        is_zero = vals_all <= zero_tol
    else:
        is_zero = np.abs(vals_all) <= zero_tol
    vals_nonzero = vals_all[~is_zero]

    if xlim is None:
        if vals_nonzero.size > 1:
            lo = float(np.nanpercentile(vals_nonzero, 1))
            hi = float(np.nanpercentile(vals_nonzero, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = (0.0, 1.0) if support == "positive" else (-1.0, 1.0)
            else:
                pad = 0.1 * (hi - lo)
                lo -= pad
                hi += pad
                if support == "positive":
                    lo = max(lo, 0.0)
                    hi = max(hi, lo + 1e-8)
                else:
                    lo = min(lo, -1e-8)
                    hi = max(hi, 1e-8)
        else:
            lo, hi = (0.0, 1.0) if support == "positive" else (-1.0, 1.0)
    else:
        lo, hi = xlim

    x = np.linspace(lo, hi, n_grid)
    y = _get_density(model, x)

    if vals_nonzero.size > 0:
        bin_counts, bin_edges = np.histogram(vals_nonzero, bins=bins, range=(lo, hi))
        bin_widths = np.diff(bin_edges)
        # Normalize using total nonzero count so density is correct even when xlim clips the data
        hist_density = bin_counts / (vals_nonzero.size * bin_widths)
        scale = 1.0 - (float(np.mean(is_zero)) if not np.isfinite(pi_zero) else pi_zero)
        hist_density = hist_density * max(scale, 0.0)
    else:
        hist_density = np.zeros(bins, dtype=float)
        bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(
        bin_edges[:-1],
        hist_density,
        width=np.diff(bin_edges),
        align="edge",
        alpha=0.5,
        edgecolor="black",
        label="Histogram (non-zero, scaled)",
    )
    ax1.plot(x, y, linewidth=2.0, label="Estimated continuous density")

    if show_zero_spike and np.isfinite(pi_zero) and pi_zero > 0:
        if spike_width is None:
            spike_width = (hi - lo) / bins * 0.35
        ax1.bar(
            -spike_width / 2,
            pi_zero,
            width=spike_width,
            align="edge",
            alpha=0.8,
            edgecolor="red",
            linewidth=1.5,
            label=f"Point mass at 0 (area={pi_zero:.3f})",
        )

    ax1.set_xlabel("SHAP value")
    ax1.set_ylabel("Density")
    if title is None:
        if is_sample_level:
            title = f"Zero-inflated density | sample={sample_id}, feature={feature}"
        else:
            title = f"Zero-inflated density | feature={feature}"
        if class_id is not None:
            title += f", class={class_id}"
    ax1.set_title(title)
    ax1.legend(fontsize=8)

    ax2 = ax1.twinx()
    if np.isfinite(pi_zero):
        ax2.vlines(0, 0, pi_zero, linestyles="--")
    ax2.set_ylabel("P(X=0)")

    return _finish(fig, show)


def plot_sample_top_features(
    mix_df,
    sample_id,
    class_id=None,
    top_k=15,
    rank_by="mean_abs_estimated",   # or "p_nonzero"
    n_grid=600,
    xlim=None,
    alpha=0.8,
    linewidth=2,
    show=True,
):
    """Overlay top feature densities from zero-inflated estimates for one sample.

    Expected columns in mix_df:
      - sample_id, feature
      - optional class_id
      - pi_zero
      - kde_model (dict from fit_zero_inflated_kde / fit_zero_inflated_normal)

    Returns
    -------
    (fig, top_df)
    """
    if class_id is not None:
        df = mix_df[(mix_df["sample_id"] == sample_id) & (mix_df["class_id"] == class_id)].copy()
    else:
        df = mix_df[mix_df["sample_id"] == sample_id].copy()
    if df.empty:
        raise ValueError(f"No rows found for sample_id={sample_id} and class_id={class_id}")

    required_cols = ["feature", "pi_zero", "kde_model"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for density overlay: {missing}")

    if xlim is None:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = xlim

    x = np.linspace(lo, hi, n_grid)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    mean_abs_est = []
    p_nonzero = []
    peak_density = []
    for _, row in df.iterrows():
        y = _get_density(row["kde_model"], x)
        mean_abs_est.append(max(float(_trapz(np.abs(x) * y, x)), 0.0))
        p_nonzero.append(1.0 - float(row["pi_zero"]))
        peak_density.append(float(np.max(y)) if y.size else 0.0)

    df["mean_abs_estimated"] = mean_abs_est
    df["p_nonzero"] = p_nonzero
    df["peak_density"] = peak_density

    if rank_by not in {"mean_abs_estimated", "p_nonzero", "peak_density"}:
        raise ValueError("rank_by must be 'mean_abs_estimated', 'p_nonzero', or 'peak_density'")

    df["rank_score"] = df[rank_by]

    top_df = df.sort_values("rank_score", ascending=False).head(top_k).copy()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for _, row in top_df.iterrows():
        y = _get_density(row["kde_model"], x)
        ax1.plot(x, y, alpha=alpha, linewidth=linewidth, label=row["feature"])

    ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("SHAP value")
    ax1.set_ylabel("Continuous density")
    if class_id is not None:
        ax1.set_title(f"Top {len(top_df)} feature densities for sample {sample_id}, class {class_id}")
    else:
        ax1.set_title(f"Top {len(top_df)} feature densities for sample {sample_id}")
    ax1.legend(fontsize=8, ncol=2)

    # second axis: show point masses at zero (only for features with pi > 0)
    pi_vals = top_df["pi_zero"].to_numpy(dtype=float)
    has_mass_at_zero = pi_vals > 0
    if np.any(has_mass_at_zero):
        ax2 = ax1.twinx()
        ax2.scatter(
            np.zeros(np.sum(has_mass_at_zero)),
            pi_vals[has_mass_at_zero],
            marker="o",
        )
        ax2.set_ylabel("P(X=0)")

    return _finish(fig, show), top_df


# ---------------------------------------------------------------------------
# Feature-level plots
# ---------------------------------------------------------------------------

def _plot_feature_distribution_common(
    row, vals_all, agg_label,
    show_zero_spike, spike_width, xlim, bins, n_grid, title,
    feature, class_id, hist_style, show,
):
    """Shared rendering for the feature-level histogram + density plots."""
    model = row.get("kde_model", None)
    pi_zero = float(row.get("pi_zero", np.nan))
    if isinstance(model, dict):
        pi_zero = float(model.get("pi_zero", pi_zero))
        zero_tol = float(model.get("zero_tol", 0.0))
        support = model.get("support", "real")
    else:
        zero_tol = 0.0
        support = "real"

    vals_all = vals_all[np.isfinite(vals_all)]
    if vals_all.size == 0:
        raise ValueError("No finite aggregated values available for the selected feature")

    is_zero = vals_all <= zero_tol if support == "positive" else np.abs(vals_all) <= zero_tol
    vals_nonzero = vals_all[~is_zero]

    # --- x-axis range ---
    if xlim is None:
        if vals_nonzero.size > 1:
            lo = float(np.nanpercentile(vals_nonzero, 1))
            hi = float(np.nanpercentile(vals_nonzero, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = 0.0, 1.0
            else:
                pad = 0.1 * (hi - lo)
                lo = max(lo - pad, 0.0) if support == "positive" else lo - pad
                hi += pad
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = xlim

    x = np.linspace(lo, hi, n_grid)
    y = _get_density(model, x)

    # --- histogram scaled by (1 - pi_zero) ---
    if vals_nonzero.size > 0:
        bin_counts, bin_edges = np.histogram(vals_nonzero, bins=bins, range=(lo, hi))
        bin_widths = np.diff(bin_edges)
        # Normalize using total nonzero count so density is correct even when xlim clips the data
        hist_density = bin_counts / (vals_nonzero.size * bin_widths)
        scale = 1.0 - (float(np.mean(is_zero)) if not np.isfinite(pi_zero) else pi_zero)
        hist_density = hist_density * max(scale, 0.0)
    else:
        hist_density = np.zeros(bins, dtype=float)
        bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    if hist_style == "grey":
        hist_kwargs = dict(alpha=0.3, color="grey", edgecolor="none")
        spike_kwargs = dict(alpha=0.8, color="red", edgecolor="none")
        vline_kwargs = dict(linestyles="--", colors="red")
    else:
        hist_kwargs = dict(alpha=0.5, edgecolor="black")
        spike_kwargs = dict(alpha=0.8, edgecolor="red", linewidth=1.5)
        vline_kwargs = dict(linestyles="--")

    ax1.bar(
        bin_edges[:-1],
        hist_density,
        width=np.diff(bin_edges),
        align="edge",
        label="Histogram (non-zero, scaled)",
        **hist_kwargs,
    )
    ax1.plot(x, y, linewidth=2.0, label="Estimated continuous density")

    if show_zero_spike and np.isfinite(pi_zero) and pi_zero > 0:
        if spike_width is None:
            spike_width = (hi - lo) / bins * 0.35
        ax1.bar(
            -spike_width / 2,
            pi_zero,
            width=spike_width,
            align="edge",
            label=f"Point mass at 0 (area={pi_zero:.3f})",
            **spike_kwargs,
        )

    ax1.set_xlabel(agg_label)
    ax1.set_ylabel("Density")
    if title is None:
        title = f"Zero-inflated density | feature={feature}"
        if class_id is not None:
            title += f", class={class_id}"
    ax1.set_title(title)
    ax1.legend(fontsize=8)

    ax2 = ax1.twinx()
    if np.isfinite(pi_zero):
        ax2.vlines(0, 0, pi_zero, **vline_kwargs)
    ax2.set_ylabel("P(X=0)")

    return _finish(fig, show)


def _select_feature_row(feature_level_df, feature, class_id):
    if class_id is not None:
        sel = (feature_level_df["feature"] == feature) & (feature_level_df["class_id"] == class_id)
    else:
        sel = feature_level_df["feature"] == feature

    row_df = feature_level_df.loc[sel]
    if row_df.empty:
        raise ValueError(f"No row for feature={feature}, class_id={class_id}")
    return row_df.iloc[0]


def plot_feature_distribution_raw(
    feature_level_df,
    feature,
    class_id=None,
    boot_results=None,
    agg_values=None,
    value_col="shap_value",
    show_zero_spike=True,
    spike_width=None,
    xlim=None,
    bins=50,
    n_grid=1000,
    title=None,
    show=True,
):
    """Feature-level density + histogram plot, recomputing aggregates from raw rows.

    Selects one row from feature_level_df by (feature[, class_id]) and overlays:
      1) histogram of non-zero aggregated |SHAP| values scaled by (1 - pi_zero),
      2) estimated continuous density from the fitted model,
      3) optional point mass spike at 0.

    Raw aggregated values come from boot_results (list[DataFrame] or DataFrame) where
    agg_shap is recomputed as sum(|shap_value|) per (bootstrap_id, perm_round).
    Alternatively pass agg_values directly as a 1-D array.
    """
    row = _select_feature_row(feature_level_df, feature, class_id)

    # --- gather raw aggregated values for histogram ---
    if agg_values is None:
        if boot_results is None:
            raise ValueError("Provide agg_values directly or provide boot_results to recompute.")

        def _extract_vals_from_df(raw_df):
            if value_col not in raw_df.columns:
                raise ValueError(f"{value_col} column not found in boot_results")

            feat_sel = raw_df["feature"] == feature
            if class_id is not None:
                if "class_id" not in raw_df.columns:
                    raise ValueError("class_id provided but class_id column not present in boot_results")
                feat_sel = feat_sel & (raw_df["class_id"] == class_id)

            sub = raw_df.loc[feat_sel]
            if sub.empty:
                return np.array([], dtype=float)

            group_cols = [c for c in ("bootstrap_id", "perm_round") if c in sub.columns]
            if class_id is not None and "class_id" in sub.columns:
                group_cols.append("class_id")

            if group_cols:
                # faster/lighter than groupby(...).apply(lambda v: np.sum(np.abs(v)))
                abs_vals = np.abs(sub[value_col].to_numpy(dtype=float))
                key_df = sub[group_cols].copy()
                key_df["_abs"] = abs_vals
                agg_series = key_df.groupby(group_cols, sort=False, observed=True)["_abs"].sum()
                return agg_series.to_numpy(dtype=float)
            else:
                return np.abs(sub[value_col].to_numpy(dtype=float))

        if isinstance(boot_results, list):
            if len(boot_results) == 0:
                raise ValueError("boot_results is an empty list")

            # no pd.concat; process each DataFrame separately
            vals_chunks = []
            for df_i in boot_results:
                vals_i = _extract_vals_from_df(df_i)
                if vals_i.size:
                    vals_chunks.append(vals_i)

            vals_all = np.concatenate(vals_chunks) if vals_chunks else np.array([], dtype=float)

        elif isinstance(boot_results, pd.DataFrame):
            vals_all = _extract_vals_from_df(boot_results)

        else:
            raise TypeError(f"boot_results must be list[DataFrame] or DataFrame, got {type(boot_results)}")
    else:
        vals_all = np.asarray(agg_values, dtype=float).reshape(-1)

    return _plot_feature_distribution_common(
        row, vals_all, "Aggregated |SHAP| per round",
        show_zero_spike, spike_width, xlim, bins, n_grid, title,
        feature, class_id, hist_style="default", show=show,
    )


def plot_feature_distribution(
    feature_level_df,
    feature,
    boot_results,
    class_id=None,
    agg_col="sum_abs_shap",
    show_zero_spike=True,
    spike_width=None,
    xlim=None,
    bins=50,
    n_grid=1000,
    title=None,
    show=True,
):
    """Feature-level density + histogram plot for pre-aggregated boot_results.

    Use when boot_results come from ``boot_multi_repeat_inference_keep_feature``,
    where each row already contains ``sum_abs_shap`` (or the column named by
    *agg_col*) — no per-sample re-aggregation is performed.
    """
    row = _select_feature_row(feature_level_df, feature, class_id)

    # --- gather pre-aggregated values for histogram ---
    if isinstance(boot_results, list):
        if len(boot_results) == 0:
            raise ValueError("boot_results is an empty list")
        raw_df = pd.concat(boot_results, ignore_index=True)
    elif isinstance(boot_results, pd.DataFrame):
        raw_df = boot_results
    else:
        raise TypeError(f"boot_results must be list[DataFrame] or DataFrame, got {type(boot_results)}")

    if agg_col not in raw_df.columns:
        raise ValueError(f"agg_col '{agg_col}' not found in boot_results columns: {list(raw_df.columns)}")

    feat_sel = raw_df["feature"] == feature
    if class_id is not None:
        if "class_id" not in raw_df.columns:
            raise ValueError("class_id provided but class_id column not present in boot_results")
        feat_sel = feat_sel & (raw_df["class_id"] == class_id)

    vals_all = raw_df.loc[feat_sel, agg_col].to_numpy(dtype=float)
    if vals_all[np.isfinite(vals_all)].size == 0:
        raise ValueError(f"No finite values for feature={feature} in boot_results['{agg_col}']")

    return _plot_feature_distribution_common(
        row, vals_all, f"Aggregated |SHAP| per round ({agg_col})",
        show_zero_spike, spike_width, xlim, bins, n_grid, title,
        feature, class_id, hist_style="grey", show=show,
    )


def plot_top_features(
    feature_level_df,
    top_k=15,
    score_col="mean_abs_estimated",
    show_metric="sd_estimated",
    show=True,
):
    """Horizontal bar chart of top features with error bars.

    Computes expected absolute value and P(nonzero) from the fitted models.

    Parameters
    ----------
    feature_level_df : DataFrame
        Output from estimate_feature_level_mixture[_preagg] with columns:
        feature, [class_id], pi_zero, kde_model, ...
    top_k : int
        Number of top features to display
    score_col : str
        Metric to rank features: "mean_abs_estimated", "p_nonzero",
        "peak_density", "n_nonzero", "nonzero_median", or "roshap_stat" if present
    show_metric : str
        Metric to show as error/secondary bar: "sd_estimated", "var_estimated",
        "p_nonzero", "pi_zero", "n_bootstrap_rounds"

    Returns
    -------
    (fig, top_df)
    """
    df = feature_level_df.copy()

    if "feature" not in df.columns:
        raise ValueError("'feature' column not found in feature_level_df")
    if "kde_model" not in df.columns:
        raise ValueError("'kde_model' column not found in feature_level_df")

    stats_df = pd.DataFrame([_model_summary_stats(row.get("kde_model")) for _, row in df.iterrows()])
    if not stats_df.empty:
        overlap_cols = [c for c in stats_df.columns if c in df.columns]
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
    df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

    if score_col not in df.columns:
        raise ValueError(
            "Metric '"
            f"{score_col}"
            "' not computed. Choose from: mean_abs_estimated, mean_estimated, "
            "var_estimated, sd_estimated, p_nonzero, peak_density, nonzero_median, n_nonzero"
        )

    # Top features sorted
    top_df = (
        df.sort_values(score_col, ascending=False)
          .head(top_k)
          .iloc[::-1]
          .copy()
        .reset_index(drop=True)
    )

    # Prepare data for barh plot
    x = top_df[score_col].to_numpy()

    if show_metric == "sd_estimated":
        xerr = top_df["sd_estimated"].to_numpy()
    elif show_metric == "var_estimated":
        xerr = np.sqrt(np.maximum(top_df["var_estimated"].to_numpy(), 0.0))
    elif show_metric == "p_nonzero":
        xerr = np.sqrt(np.maximum(top_df["p_nonzero"].to_numpy() * (1 - top_df["p_nonzero"].to_numpy()), 0))
    elif show_metric == "pi_zero":
        xerr = np.sqrt(np.maximum(top_df["pi_zero"].to_numpy() * (1 - top_df["pi_zero"].to_numpy()), 0))
    elif show_metric == "n_bootstrap_rounds":
        xerr = np.sqrt(np.maximum(top_df.get("n_bootstrap_rounds", 1).to_numpy(), 1))
    else:
        xerr = None

    fig, ax = plt.subplots(figsize=(10, 7))
    if xerr is not None:
        ax.barh(top_df["feature"].astype(str), x, xerr=xerr, alpha=0.7)
    else:
        ax.barh(top_df["feature"].astype(str), x, alpha=0.7)

    if xerr is not None:
        ax.set_xlabel(f"{score_col} with {show_metric}")
    else:
        ax.set_xlabel(f"{score_col}")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(top_df)} features by {score_col}")

    return _finish(fig, show), top_df


def plot_top_features_density(
    feature_level_df,
    top_k=15,
    score_col="mean_abs_estimated",
    feature_col="feature",
    x_min=0.0,
    x_max=None,
    n_grid=800,
    log_x=False,
    show=True,
):
    """Plot zero-inflated mixtures for top features.

    Visualizes:
        A_j ~ pi_zero * delta_0 + (1 - pi_zero) * f(a)

    Left panel: point mass at zero (pi_zero)
    Right panel: continuous density on the full support

    Parameters
    ----------
    feature_level_df : DataFrame
        Output from estimate_feature_level_mixture[_preagg] with columns:
        feature, [class_id], pi_zero, kde_model, ...
    top_k : int
        Number of top features
    score_col : str
        Metric to rank features: "mean_abs_estimated", "mean_estimated", "var_estimated",
        "sd_estimated", "p_nonzero", "peak_density", "nonzero_median"
    feature_col : str
        Name of feature column
    x_min, x_max : float
        X-axis range for density plot
    n_grid : int
        Number of grid points for density evaluation
    log_x : bool
        Whether to use log scale for x-axis

    Returns
    -------
    (fig, top_df)
    """
    df = feature_level_df.copy()

    if feature_col not in df.columns:
        raise ValueError(f"'{feature_col}' column not found")
    if "kde_model" not in df.columns:
        raise ValueError("'kde_model' column not found")
    if "pi_zero" not in df.columns:
        raise ValueError("'pi_zero' column not found")

    stats_df = pd.DataFrame([_model_summary_stats(row.get("kde_model")) for _, row in df.iterrows()])
    if not stats_df.empty:
        overlap_cols = [c for c in stats_df.columns if c in df.columns]
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
    df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

    # Validate score_col
    if score_col not in df.columns:
        raise ValueError(
            "Metric '"
            f"{score_col}"
            "' not available. Choose from: mean_abs_estimated, mean_estimated, "
            "var_estimated, sd_estimated, p_nonzero, peak_density, nonzero_median"
        )

    # Get top features
    top_df = (
        df.sort_values(score_col, ascending=False)
        .head(top_k)
        .copy()
        .reset_index(drop=True)
    )

    # Determine x-axis range
    if x_max is None:
        xmax_candidates = []
        for _, row in top_df.iterrows():
            model = row.get("kde_model")
            if isinstance(model, dict):
                xmax = model.get("nonzero_max", np.nan)
                bw = _model_grid_width(model)
                if np.isfinite(xmax):
                    xmax_candidates.append(float(xmax) + 4.0 * bw)
        x_max = max(xmax_candidates) if xmax_candidates else 1.0
        x_max = max(x_max, x_min + 1e-5)

    x = np.linspace(x_min, x_max, n_grid)

    fig, (ax0, ax1) = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.2, 3.8]}
    )

    y_positions = np.arange(len(top_df))

    # Left panel: point mass at zero
    ax0.barh(
        y_positions,
        top_df["pi_zero"].to_numpy(),
        alpha=0.7,
        color="steelblue"
    )
    ax0.set_yticks(y_positions)
    ax0.set_yticklabels(top_df[feature_col].astype(str).tolist())
    ax0.invert_yaxis()
    ax0.set_xlim(0, 1)
    ax0.set_xlabel(r"$\pi_0$ = P(A=0)")
    ax0.set_title("Point mass at 0")

    # Right panel: continuous density
    for i, row in top_df.iterrows():
        model = row.get("kde_model")
        y = _get_density(model, x)

        label = (
            f"{row[feature_col]} | "
            f"π₀={row['pi_zero']:.2f}, "
            f"q={row['p_nonzero']:.2f}"
        )

        if np.any(y > 0):
            ax1.plot(x, y, label=label, linewidth=1.5)
        else:
            ax1.plot([], [], label=label)

    if log_x and np.all(x > 0):
        ax1.set_xscale("log")

    ax1.set_xlabel("Aggregated |SHAP| per round")
    ax1.set_ylabel("Continuous density (scaled by 1 - π₀)")
    ax1.set_title(f"Top {len(top_df)} zero-inflated mixtures")
    ax1.legend(fontsize=8, ncol=1, loc="best")
    ax1.grid(True, alpha=0.3)

    return _finish(fig, show), top_df


def plot_ridge(
    boot_results,
    features,
    agg_col="sum_abs_shap",
    class_id=None,
    x_min=None,
    x_max=None,
    n_grid=500,
    spacing=0.65,
    scale=0.55,
    figsize=None,
    show=True,
):
    """Ridge (joy) plot of per-feature bootstrap attribution distributions.

    One Gaussian-KDE silhouette per feature, top feature at the top.

    Parameters
    ----------
    boot_results : list of pd.DataFrame or pd.DataFrame
        Pre-aggregated bootstrap results (from
        ``boot_multi_repeat_inference_keep_feature``) containing *agg_col*.
    features : sequence of str
        Features to plot, in rank order (first = plotted on top).
    agg_col : str
        Column holding the per-bootstrap aggregated attribution.
    class_id : int, optional
        For multiclass results, restrict to one class.
    x_min, x_max : float, optional
        X-axis range. Defaults: 0 (or the data minimum when values are
        signed) and the 99.5th percentile of the plotted values.
    """
    if isinstance(boot_results, list):
        if len(boot_results) == 0:
            raise ValueError("boot_results is an empty list")
        raw_df = pd.concat(boot_results, ignore_index=True)
    elif isinstance(boot_results, pd.DataFrame):
        raw_df = boot_results
    else:
        raise TypeError(f"boot_results must be list[DataFrame] or DataFrame, got {type(boot_results)}")

    if agg_col not in raw_df.columns:
        raise ValueError(f"agg_col '{agg_col}' not found in boot_results columns: {list(raw_df.columns)}")

    features = [str(f) for f in features]
    if not features:
        raise ValueError("features must be a non-empty sequence")

    part = raw_df.copy()
    part["feature"] = part["feature"].astype(str)
    if class_id is not None:
        if "class_id" not in part.columns:
            raise ValueError("class_id provided but class_id column not present in boot_results")
        part = part[part["class_id"] == class_id]
    part = part[part["feature"].isin(features)]
    part = part.dropna(subset=[agg_col])

    plot_data = [
        part.loc[part["feature"] == f, agg_col].to_numpy(dtype=float)
        for f in features
    ]

    nonempty = [v for v in plot_data if v.size]
    if not nonempty:
        raise ValueError(f"No data found for the requested features in boot_results['{agg_col}']")
    all_vals = np.concatenate(nonempty)

    if x_min is None:
        data_min = float(np.nanmin(all_vals))
        x_min = 0.0 if data_min >= 0 else float(np.nanpercentile(all_vals, 0.5))
    if x_max is None:
        x_max = float(np.nanpercentile(all_vals, 99.5))
    if x_max <= x_min:
        x_max = x_min + 1e-8

    x_grid = np.linspace(x_min, x_max, n_grid)

    if figsize is None:
        figsize = (9, max(6, 0.45 * len(features)))
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.45, right=0.98, top=0.98, bottom=0.10)

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDDDD", linewidth=1.0, zorder=0)

    for i, feature in enumerate(features):
        vals = plot_data[i]
        y_base = (len(features) - 1 - i) * spacing

        if len(vals) > 2 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            dens = kde(x_grid)
            dens = dens / dens.max() * scale
        else:
            dens = np.zeros_like(x_grid)

        ax.fill_between(
            x_grid,
            y_base,
            y_base + dens,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.8,
            alpha=1.0,
            zorder=10 + i,
        )

        ax.plot(
            x_grid,
            y_base + dens,
            color="black",
            linewidth=0.8,
            zorder=11 + i,
        )

        ax.hlines(
            y_base,
            x_min,
            x_max,
            color="black",
            linewidth=0.5,
            zorder=12 + i,
        )

    ax.set_yticks([(len(features) - 1 - i) * spacing for i in range(len(features))])
    ax.set_yticklabels(features, fontsize=13)

    ax.set_xlabel("Feature Attribution", fontsize=18)
    ax.set_ylabel("")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.15, (len(features) - 1) * spacing + scale + 0.15)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", labelsize=13, length=0)
    ax.tick_params(axis="y", labelsize=15, length=0)

    return _finish(fig, show)

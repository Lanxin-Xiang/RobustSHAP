import pandas as pd
import numpy as np
from math import erf
from math import lgamma


import matplotlib.pyplot as plt


def plot_zero_inflated_kde_with_hist(
    mix_df,
    sample_id,
    feature,
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
):
    """Plot estimated zero-inflated KDE density and histogram for one group.

    Selects one row from mix_df by (sample_id, feature[, class_id]) and overlays:
      1) histogram of non-zero SHAP values scaled by (1 - pi_zero),
      2) estimated continuous density from kde_model,
      3) optional point mass spike at 0.
    """
    if class_id is not None:
        sel = (
            (mix_df["sample_id"] == sample_id)
            & (mix_df["feature"] == feature)
            & (mix_df["class_id"] == class_id)
        )
    else:
        sel = (mix_df["sample_id"] == sample_id) & (mix_df["feature"] == feature)

    row_df = mix_df.loc[sel]
    if row_df.empty:
        raise ValueError(
            f"No estimated density row for sample_id={sample_id}, feature={feature}, class_id={class_id}"
        )
    if len(row_df) > 1:
        row_df = row_df.head(1)

    row = row_df.iloc[0]
    model = row.get("kde_model", None)
    pi_zero = float(row.get("pi_zero", np.nan))
    if isinstance(model, dict):
        pi_zero = float(model.get("pi_zero", pi_zero))
        zero_tol = float(model.get("zero_tol", 0.0))
        kde = model.get("kde", None)
    else:
        zero_tol = 0.0
        kde = None

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

        raw_sel = (raw_df["sample_id"] == sample_id) & (raw_df["feature"] == feature)
        if class_id is not None:
            if "class_id" not in raw_df.columns:
                raise ValueError("class_id provided but class_id column not present in boot_results")
            raw_sel = raw_sel & (raw_df["class_id"] == class_id)

        if value_col not in raw_df.columns:
            raise ValueError(f"{value_col} column not found in boot_results")

        vals_all = raw_df.loc[raw_sel, value_col].to_numpy(dtype=float)
    else:
        vals_all = np.asarray(shap_values, dtype=float).reshape(-1)

    vals_all = vals_all[np.isfinite(vals_all)]
    if vals_all.size == 0:
        raise ValueError("No finite SHAP values available for the selected group")

    is_zero = np.abs(vals_all) <= zero_tol
    vals_nonzero = vals_all[~is_zero]

    if xlim is None:
        if vals_nonzero.size > 1:
            lo = float(np.nanpercentile(vals_nonzero, 1))
            hi = float(np.nanpercentile(vals_nonzero, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1.0, 1.0
            else:
                pad = 0.1 * (hi - lo)
                lo -= pad
                hi += pad
                lo = min(lo, -1e-8)
                hi = max(hi, 1e-8)
        else:
            lo, hi = -1.0, 1.0
    else:
        lo, hi = xlim

    x = np.linspace(lo, hi, n_grid)

    if kde is not None and np.isfinite(pi_zero):
        y = (1.0 - pi_zero) * np.exp(kde.score_samples(x.reshape(-1, 1)))
    else:
        y = np.zeros_like(x, dtype=float)

    if vals_nonzero.size > 0:
        hist_density, bin_edges = np.histogram(vals_nonzero, bins=bins, range=(lo, hi), density=True)
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
        title = f"Zero-inflated KDE | sample={sample_id}, feature={feature}"
        if class_id is not None:
            title += f", class={class_id}"
    ax1.set_title(title)
    ax1.legend(fontsize=8)

    ax2 = ax1.twinx()
    if np.isfinite(pi_zero):
        ax2.vlines(0, 0, pi_zero, linestyles="--")
    ax2.set_ylabel("P(X=0)")

    plt.tight_layout()
    plt.show()


def plot_sample_top_features_overlay(
    mix_df,
    sample_id,
    class_id=None,
    top_k=15,
    rank_by="mean_abs_estimated",   # or "p_nonzero"
    n_grid=600,
    xlim=None,
    alpha=0.8,
    linewidth=2,
):
    """Overlay top feature densities from zero-inflated KDE estimates.

    Expected columns in mix_df:
      - sample_id, feature
      - optional class_id
      - pi_zero
      - kde_model (dict from fit_zero_inflated_kde)
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
        raise ValueError(f"Missing required columns for KDE overlay: {missing}")

    def _get_density(model, xvals):
        if not isinstance(model, dict):
            return np.zeros_like(xvals, dtype=float)

        pi_zero = float(model.get("pi_zero", np.nan))
        kde = model.get("kde")
        if np.isnan(pi_zero) or kde is None:
            return np.zeros_like(xvals, dtype=float)

        return (1.0 - pi_zero) * np.exp(kde.score_samples(np.asarray(xvals).reshape(-1, 1)))

    if xlim is None:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = xlim

    x = np.linspace(lo, hi, n_grid)

    mean_abs_est = []
    p_nonzero = []
    peak_density = []
    for _, row in df.iterrows():
        y = _get_density(row["kde_model"], x)
        mean_abs_est.append(np.trapz(np.abs(x) * y, x))
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
        ax1.set_title(f"Top {len(top_df)} feature KDE mixtures for sample {sample_id}, class {class_id}")
    else:
        ax1.set_title(f"Top {len(top_df)} feature KDE mixtures for sample {sample_id}")
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

    plt.tight_layout()
    plt.show()

    return top_df


def plot_top_feature_with_error(
    feature_level_df,
    top_k=15,
    score_col="mean_abs_estimated",
    show_metric="sd_estimated",
):
    """Plot top features from zero-inflated KDE mixture at feature level.

    Computes expected absolute value and P(nonzero) from KDE models.
    
    Parameters
    ----------
    feature_level_df : DataFrame
        Output from estimate_feature_level_mixture with columns:
        feature, [class_id], pi_zero, kde_model, ...
    top_k : int
        Number of top features to display
    score_col : str
        Metric to rank features: "mean_abs_estimated", "p_nonzero", "peak_density", "n_nonzero", "nonzero_median"
    show_metric : str
        Metric to show as error/secondary bar: "sd_estimated", "var_estimated",
        "p_nonzero", "pi_zero", "n_bootstrap_rounds"
    """
    df = feature_level_df.copy()

    if "feature" not in df.columns:
        raise ValueError("'feature' column not found in feature_level_df")
    if "kde_model" not in df.columns:
        raise ValueError("'kde_model' column not found in feature_level_df")

    # Compute metrics from KDE models
    def _get_density(model, xvals):
        if not isinstance(model, dict):
            return np.zeros_like(xvals, dtype=float)

        support = model.get("support", "real")
        pi_zero = float(model.get("pi_zero", np.nan))
        kde = model.get("kde")
        if np.isnan(pi_zero) or kde is None:
            return np.zeros_like(xvals, dtype=float)

        xvals = np.asarray(xvals, dtype=float)
        if support == "positive":
            out = np.zeros_like(xvals, dtype=float)
            mask = xvals > float(model.get("zero_tol", 0.0))
            if np.any(mask):
                z = np.log(xvals[mask]).reshape(-1, 1)
                out[mask] = (1.0 - pi_zero) * np.exp(kde.score_samples(z)) / xvals[mask]
            return out

        return (1.0 - pi_zero) * np.exp(kde.score_samples(xvals.reshape(-1, 1)))

    def _integration_grid(model):
        if not isinstance(model, dict):
            return np.linspace(0.0, 1.0, 400)

        support = model.get("support", "real")
        xmax = float(model.get("nonzero_max", np.nan))
        xmin = float(model.get("nonzero_min", np.nan))
        bw = float(model.get("bandwidth", 0.2))

        if support == "positive":
            hi = xmax + 4.0 * bw if np.isfinite(xmax) else 1.0
            hi = max(hi, (xmin if np.isfinite(xmin) else 0.0) + 1e-3)
            return np.linspace(0.0, hi, 400)

        lo = (xmin - 4.0 * bw) if np.isfinite(xmin) else -1.0
        hi = (xmax + 4.0 * bw) if np.isfinite(xmax) else 1.0
        if hi <= lo:
            lo, hi = -1.0, 1.0
        return np.linspace(lo, hi, 400)

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

        mean_abs = float(np.trapz(np.abs(x_grid) * y, x_grid))
        mean_val = float(np.trapz(x_grid * y, x_grid))
        second_moment = float(np.trapz((x_grid ** 2) * y, x_grid))
        var_val = max(second_moment - mean_val ** 2, 0.0)
        sd_val = float(np.sqrt(var_val))
        peak = float(np.max(y)) if y.size else 0.0

        cont_mass = float(np.trapz(y, x_grid))
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

    stats_df = pd.DataFrame([_model_summary_stats(row.get("kde_model")) for _, row in df.iterrows()])
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

    plt.tight_layout()
    plt.show()

    return top_df


def plot_top_feature_density(
    feature_level_df,
    top_k=15,
    score_col="mean_abs_estimated",
    feature_col="feature",
    x_min=0.0,
    x_max=None,
    n_grid=800,
    log_x=False,
):
    """Plot zero-inflated KDE mixtures for top features from estimate_feature_level_mixture.

    Visualizes:
        A_j ~ pi_zero * delta_0 + (1 - pi_zero) * KDE(a)

    Left panel: point mass at zero (pi_zero)
    Right panel: continuous KDE density on the full support

    Parameters
    ----------
    feature_level_df : DataFrame
        Output from estimate_feature_level_mixture with columns:
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
    """
    df = feature_level_df.copy()

    if feature_col not in df.columns:
        raise ValueError(f"'{feature_col}' column not found")
    if "kde_model" not in df.columns:
        raise ValueError("'kde_model' column not found")
    if "pi_zero" not in df.columns:
        raise ValueError("'pi_zero' column not found")

    def _get_density(model, xvals):
        if not isinstance(model, dict):
            return np.zeros_like(xvals, dtype=float)

        support = model.get("support", "real")
        pi_zero = float(model.get("pi_zero", np.nan))
        kde = model.get("kde")
        if np.isnan(pi_zero) or kde is None:
            return np.zeros_like(xvals, dtype=float)

        xvals = np.asarray(xvals, dtype=float)
        if support == "positive":
            out = np.zeros_like(xvals, dtype=float)
            mask = xvals > float(model.get("zero_tol", 0.0))
            if np.any(mask):
                z = np.log(xvals[mask]).reshape(-1, 1)
                out[mask] = (1.0 - pi_zero) * np.exp(kde.score_samples(z)) / xvals[mask]
            return out

        return (1.0 - pi_zero) * np.exp(kde.score_samples(xvals.reshape(-1, 1)))

    def _integration_grid(model):
        if not isinstance(model, dict):
            return np.linspace(0.0, 1.0, 400)

        support = model.get("support", "real")
        xmax = float(model.get("nonzero_max", np.nan))
        xmin = float(model.get("nonzero_min", np.nan))
        bw = float(model.get("bandwidth", 0.2))

        if support == "positive":
            hi = xmax + 4.0 * bw if np.isfinite(xmax) else 1.0
            hi = max(hi, (xmin if np.isfinite(xmin) else 0.0) + 1e-3)
            return np.linspace(0.0, hi, 400)

        lo = (xmin - 4.0 * bw) if np.isfinite(xmin) else -1.0
        hi = (xmax + 4.0 * bw) if np.isfinite(xmax) else 1.0
        if hi <= lo:
            lo, hi = -1.0, 1.0
        return np.linspace(lo, hi, 400)

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

        mean_abs = float(np.trapz(np.abs(x_grid) * y, x_grid))
        mean_val = float(np.trapz(x_grid * y, x_grid))
        second_moment = float(np.trapz((x_grid ** 2) * y, x_grid))
        var_val = max(second_moment - mean_val ** 2, 0.0)
        sd_val = float(np.sqrt(var_val))
        peak = float(np.max(y)) if y.size else 0.0

        cont_mass = float(np.trapz(y, x_grid))
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

    stats_df = pd.DataFrame([_model_summary_stats(row.get("kde_model")) for _, row in df.iterrows()])
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
                bw = float(model.get("bandwidth", 0.2))
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

    # Right panel: continuous KDE density
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
    ax1.set_title(f"Top {len(top_df)} zero-inflated KDE mixtures")
    ax1.legend(fontsize=8, ncol=1, loc="best")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return top_df
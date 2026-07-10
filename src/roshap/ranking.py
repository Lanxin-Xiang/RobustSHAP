"""RoSHAP robust feature ranking.

The RoSHAP statistic combines how often a feature matters, how much it
matters, and how stable that contribution is across bootstrap resamples:

    SNR         = median / sd_estimated
    roshap_stat = p_nonzero * nonzero_median * SNR

where, per feature: ``median`` is the empirical median of the aggregated
|SHAP| bootstrap draws, ``sd_estimated`` the standard deviation of the fitted
zero-inflated density, ``p_nonzero`` the probability the feature is used at
all, and ``nonzero_median`` the median of the nonzero draws.
"""
import numpy as np
import pandas as pd


def add_roshap_stat(feature_stats):
    """Add ``SNR`` and ``roshap_stat`` columns to a feature-stats DataFrame.

    Parameters
    ----------
    feature_stats : pd.DataFrame
        Output of ``estimate_feature_level_mixture_preagg`` — must contain
        ``median``, ``sd_estimated``, ``p_nonzero``, ``nonzero_median``.

    Returns
    -------
    pd.DataFrame
        Copy of the input with ``SNR`` and ``roshap_stat`` columns added.
    """
    required = {"median", "sd_estimated", "p_nonzero", "nonzero_median"}
    missing = required - set(feature_stats.columns)
    if missing:
        raise ValueError(f"feature_stats is missing required columns: {sorted(missing)}")

    out = feature_stats.copy()
    sd = out["sd_estimated"].to_numpy(dtype=float)
    med = out["median"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sd > 0, med / sd, 0.0)
    out["SNR"] = snr
    out["roshap_stat"] = out["p_nonzero"] * out["nonzero_median"] * out["SNR"]
    return out


def rank_features(feature_stats, k=None, by="roshap_stat", class_id=None):
    """Rank features by a robustness statistic.

    Parameters
    ----------
    feature_stats : pd.DataFrame
        Feature-level stats. If the ranking column named by *by* is absent
        but computable, ``add_roshap_stat`` is applied first.
    k : int, optional
        Keep only the top k features. None keeps all.
    by : str
        Column to rank by (default "roshap_stat").
    class_id : int, optional
        For multiclass results, restrict to one class. When None and a
        class_id column is present, the statistic is averaged across classes
        per feature.

    Returns
    -------
    pd.DataFrame
        Sorted descending by *by*, with a 1-based "Rank" index and an
        ``importance`` column equal to the ranking statistic.
    """
    df = feature_stats
    if by not in df.columns:
        df = add_roshap_stat(df)
        if by not in df.columns:
            raise ValueError(f"Ranking column '{by}' not found in feature_stats.")

    if "class_id" in df.columns:
        if class_id is not None:
            df = df[df["class_id"] == class_id]
            if df.empty:
                raise ValueError(f"No rows for class_id={class_id}.")
        else:
            df = (
                df.groupby("feature", as_index=False, sort=False, observed=True)[by]
                .mean()
            )

    top = df.sort_values(by, ascending=False).reset_index(drop=True)
    if k is not None:
        top = top.head(k)
    top.index = pd.RangeIndex(1, len(top) + 1, name="Rank")
    top = top.copy()
    top["importance"] = top[by]
    return top

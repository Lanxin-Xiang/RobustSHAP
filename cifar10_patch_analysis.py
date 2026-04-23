"""
Per-patch SHAP distribution analysis for CIFAR-10 bootstrap runs.

Data layout:
  bootstrap_runs 2/
    run_000/ ... run_049/   (50 runs)
      shap_results/
        patch_shap.npy   (100, 8, 8)  - SHAP per image per patch
        images.npy       (100, 32, 32, 3)
        labels.npy       (100,)
        predicted.npy    (100,)

For each (image, patch) pair we collect 50 SHAP values (one per run) and fit
a zero-inflated KDE via estimate_sample_feature_distribution.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from distribution_estimation import (
    estimate_sample_feature_distribution,
    zero_inflated_kde_pdf,
    _summarize_zero_inflated_kde_model,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
PATCH_GRID = (8, 8)
RUNS_DIR = Path(__file__).parent / "shap_results" / "bootstrap_runs 2"


# ─── 1. Load data ──────────────────────────────────────────────────────────────

def load_bootstrap_runs(runs_dir: Path = RUNS_DIR):
    """Return (boot_results, images, labels, predicted).

    boot_results : list of DataFrames, one per run, columns
        [sample_id, feature, shap_value]
    images : (100, 32, 32, 3) float32
    labels : (100,) int64
    predicted : (100,) int64  — majority vote across all runs
    """
    run_dirs = sorted(runs_dir.glob("run_*"))
    boot_results = []
    images_ref = labels_ref = None
    all_predicted = []

    for run_dir in run_dirs:
        sr = run_dir / "shap_results"
        shap = np.load(sr / "patch_shap.npy")          # (100, 8, 8)
        n_images = shap.shape[0]
        n_patches = PATCH_GRID[0] * PATCH_GRID[1]
        flat = shap.reshape(n_images, n_patches)        # (100, 64)

        if images_ref is None:
            images_ref = np.load(sr / "images.npy")
            labels_ref = np.load(sr / "labels.npy")

        all_predicted.append(np.load(sr / "predicted.npy"))

        sample_ids = np.repeat(np.arange(n_images), n_patches)
        feature_ids = np.tile(np.arange(n_patches), n_images)
        boot_results.append(pd.DataFrame({
            "sample_id": sample_ids,
            "feature":   feature_ids,
            "shap_value": flat.ravel(),
        }))

    # majority vote: shape (n_runs, n_images) → (n_images,)
    votes = np.stack(all_predicted, axis=0)
    n_classes = votes.max() + 1
    counts = np.apply_along_axis(lambda col: np.bincount(col, minlength=n_classes), 0, votes)
    predicted_ref = counts.argmax(axis=0).astype(votes.dtype)

    return boot_results, images_ref, labels_ref, predicted_ref


# ─── 2. Fit distributions ──────────────────────────────────────────────────────

def fit_patch_distributions(boot_results, bandwidth=None):
    """Run estimate_sample_feature_distribution on the loaded boot_results.

    If bandwidth is None, Scott's rule is applied:
        bw = 1.06 * global_std * n_runs^(-1/5)

    Returns dist_df with columns including kde_model plus KDE summary stats
    (mean_estimated, sd_estimated, p_nonzero, ...) from _summarize_zero_inflated_kde_model.
    """
    n_runs = len(boot_results)

    if bandwidth is None:
        all_shap = np.concatenate([df["shap_value"].values for df in boot_results])
        bandwidth = 1.06 * all_shap.std() * (n_runs ** -0.2)
        print(f"Auto bandwidth (Scott's rule, n={n_runs}): {bandwidth:.3e}")

    dist_df = estimate_sample_feature_distribution(
        boot_results,
        value_col="shap_value",
        bandwidth=bandwidth,
        kernel="gaussian",
        zero_tol=0.0,
    )

    summaries = dist_df["kde_model"].apply(_summarize_zero_inflated_kde_model)
    dist_df = pd.concat(
        [dist_df.reset_index(drop=True), pd.DataFrame(summaries.tolist())],
        axis=1,
    )
    abs_median = dist_df["nonzero_median"].abs()
    dist_df["snr"] = (abs_median / (dist_df["sd_estimated"] + 1e-12)) * abs_median

    print(f"Fitted distributions: {dist_df.shape[0]} (sample_id, patch) pairs")
    return dist_df


# ─── 3. Per-image data helpers ────────────────────────────────────────────────

def get_image_patch_stats(dist_df, image_idx):
    """Return (8×8 arrays, kde_models_flat, img_df) for one image."""
    img_df = (
        dist_df[dist_df["sample_id"] == image_idx]
        .sort_values("feature")
        .reset_index(drop=True)
    )
    ph, pw = PATCH_GRID
    median_shap = img_df["nonzero_median"].values.reshape(ph, pw)
    sd_shap     = img_df["sd_estimated"].values.reshape(ph, pw)
    snr         = img_df["snr"].values.reshape(ph, pw)
    kde_models  = img_df["kde_model"].values  # length-64 array of dicts
    return median_shap, sd_shap, snr, kde_models, img_df


def _upsample_patch_map(arr_8x8, scale=4):
    """Nearest-neighbour upsample (8,8) → (32,32)."""
    return np.kron(arr_8x8, np.ones((scale, scale)))


# ─── 4. Visualization ─────────────────────────────────────────────────────────

def plot_image_explanation(
    image_idx,
    dist_df,
    images,
    labels,
    predicted,
    top_k=6,
    figsize=(20, 9),
):
    """Comprehensive per-image explanation panel.

    Layout
    ------
    Top row (4 panels):
      [0] Original image
      [1] Median SHAP overlay on image  (red=+, blue=-)
      [2] SHAP std-dev heatmap          (shows variability across runs)
      [3] |Median|²/Std heatmap         (shows attribution confidence)

    Bottom row (top_k panels):
      KDE distribution for top-k patches ranked by |median SHAP|.
    """
    median_shap, sd_shap, snr, kde_models, img_df = get_image_patch_stats(dist_df, image_idx)

    true_cls = CIFAR10_CLASSES[labels[image_idx]]
    pred_cls = CIFAR10_CLASSES[predicted[image_idx]]
    match    = "✓" if labels[image_idx] == predicted[image_idx] else "✗"

    n_cols = max(4, top_k)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Image {image_idx}  |  True: {true_cls}  |  Predicted: {pred_cls} {match}",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(
        2, n_cols,
        height_ratios=[1.3, 1.0],
        hspace=0.45, wspace=0.30,
    )

    # ── Panel 0: original image ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(images[image_idx])
    ax0.set_title("Original Image", fontsize=10)
    ax0.axis("off")

    # ── Panel 1: median SHAP overlaid on image ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    median_up = _upsample_patch_map(median_shap)
    vabs = max(np.abs(median_shap).max(), 1e-12)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
    ax1.imshow(images[image_idx], alpha=0.45)
    im1 = ax1.imshow(median_up, cmap="RdBu_r", norm=norm, alpha=0.6)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    # patch grid lines
    for i in range(1, PATCH_GRID[0]):
        ax1.axhline(i * 4 - 0.5, color="white", lw=0.4, alpha=0.5)
        ax1.axvline(i * 4 - 0.5, color="white", lw=0.4, alpha=0.5)
    ax1.set_title("Median SHAP per Patch\n(red=+ toward pred class)", fontsize=10)
    ax1.axis("off")

    # ── Panel 2: std-dev heatmap ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(sd_shap, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("SHAP Std Dev\n(variability across 50 runs)", fontsize=10)
    _style_patch_ax(ax2)

    # ── Panel 3: SNR heatmap ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(snr, cmap="Greens", interpolation="nearest")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    # ax3.set_title("|Median SHAP|² / Std Dev\n(attribution confidence)", fontsize=10)
    ax3.set_title("Robust SHAP", fontsize=10)
    _style_patch_ax(ax3)

    # ── Bottom row: top-k KDE distributions ───────────────────────────────────
    abs_median = img_df["nonzero_median"].abs().values
    top_order  = np.argsort(abs_median)[::-1][:top_k]

    for plot_pos, patch_flat_idx in enumerate(top_order):
        ax = fig.add_subplot(gs[1, plot_pos])
        model      = kde_models[patch_flat_idx]
        median_val = img_df["nonzero_median"].iloc[patch_flat_idx]
        sd_val     = img_df["sd_estimated"].iloc[patch_flat_idx]
        row, col   = divmod(int(img_df["feature"].iloc[patch_flat_idx]), PATCH_GRID[1])

        color = "tomato" if median_val >= 0 else "steelblue"

        if model is not None and model.get("kde") is not None:
            lo = model["nonzero_min"] - 4 * model["bandwidth"]
            hi = model["nonzero_max"] + 4 * model["bandwidth"]
            x_grid = np.linspace(lo, hi, 400)
            y_pdf  = zero_inflated_kde_pdf(x_grid, model)
            ax.plot(x_grid, y_pdf, color=color, lw=1.8)
            ax.fill_between(x_grid, y_pdf, alpha=0.18, color=color)

        ax.set_title(
            f"Patch ({row}, {col})\nmed={median_val:.2e}, σ={sd_val:.2e}",
            fontsize=8,
        )
        ax.set_xlabel("SHAP value", fontsize=7)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax.tick_params(axis="x", labelsize=6)

    # Hide unused bottom-row axes
    for i in range(top_k, n_cols):
        fig.add_subplot(gs[1, i]).set_visible(False)

    return fig


def _style_patch_ax(ax):
    ax.set_xticks(range(PATCH_GRID[1]))
    ax.set_yticks(range(PATCH_GRID[0]))
    ax.tick_params(labelsize=6)


# ─── 5. Multi-image summary ────────────────────────────────────────────────────

def plot_patch_importance_summary(dist_df, labels, predicted, n_images=20, figsize=(14, 8)):
    """Heatmap grid: each row = one image, each column = one patch.

    The colour shows mean SHAP, giving a quick overview of which patches
    matter across images.  Images are grouped by predicted class.
    """
    image_ids = dist_df["sample_id"].unique()[:n_images]
    n_patches = PATCH_GRID[0] * PATCH_GRID[1]

    mat = np.zeros((len(image_ids), n_patches))
    row_labels = []
    for i, img_idx in enumerate(image_ids):
        img_df = dist_df[dist_df["sample_id"] == img_idx].sort_values("feature")
        mat[i] = img_df["nonzero_median"].values
        true_cls = CIFAR10_CLASSES[labels[img_idx]]
        pred_cls = CIFAR10_CLASSES[predicted[img_idx]]
        match = "✓" if labels[img_idx] == predicted[img_idx] else "✗"
        row_labels.append(f"#{img_idx} {true_cls}→{pred_cls}{match}")

    vabs = np.abs(mat).max() or 1e-8
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, label="Median SHAP")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(f"Patch index (0–{n_patches-1}, row-major in 8×8 grid)")
    ax.set_title(f"Per-patch Median SHAP across {len(image_ids)} images")

    # thin column separators at 8-patch boundaries
    for x in range(8, n_patches, 8):
        ax.axvline(x - 0.5, color="white", lw=0.6, alpha=0.6)

    plt.tight_layout()
    return fig


# ─── 6. Main entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading bootstrap runs…")
    boot_results, images, labels, predicted = load_bootstrap_runs()
    print(f"  {len(boot_results)} runs, {images.shape[0]} images, "
          f"{PATCH_GRID[0]*PATCH_GRID[1]} patches each.")

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("Fitting zero-inflated KDEs…")
    dist_df = fit_patch_distributions(boot_results)

    # ── Example plots ─────────────────────────────────────────────────────────
    out_dir = Path("figures") / "patch_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_idx in range(5):          # change range to plot more images
        fig = plot_image_explanation(img_idx, dist_df, images, labels, predicted)
        fig.savefig(out_dir / f"image_{img_idx:03d}_explanation.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved image {img_idx}")

    fig_summary = plot_patch_importance_summary(dist_df, labels, predicted, n_images=20)
    fig_summary.savefig(out_dir / "summary_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    print("Done. Figures written to", out_dir)

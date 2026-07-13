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
PATCH_GRID = (24, 24)
RUNS_DIR = Path(__file__).parent / "shap_results" / "bootstrap_runs_vit_base"


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
        shap = np.load(sr / "patch_shap.npy")          # (100, 24, 24)
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


def _upsample_patch_map(arr, img_size=None):
    """Nearest-neighbour upsample patch map to match image spatial size.

    If img_size is None, defaults to scale=1 (no upsampling).
    img_size can be an int (square) or (H, W) tuple.
    """
    ph, pw = arr.shape
    if img_size is None:
        return arr
    if isinstance(img_size, int):
        ih, iw = img_size, img_size
    else:
        ih, iw = img_size
    scale_h = ih // ph
    scale_w = iw // pw
    if scale_h < 1 or scale_w < 1:
        return arr
    return np.kron(arr, np.ones((scale_h, scale_w)))


# ─── 4. Visualization ─────────────────────────────────────────────────────────

def plot_image_explanation(
    image_idx,
    dist_df,
    images,
    labels,
    predicted,
    top_k=10,
    figsize=(15, 9),
):
    """Comprehensive per-image explanation panel.

    Layout
    ------
    Top row (4 panels):
      [0] Original image
      [1] Median SHAP overlay on image  (red=+, blue=-)
      [2] SHAP std-dev heatmap          (shows variability across runs)
      [3] |Median|²/Std heatmap         (shows attribution confidence)

    Bottom row (1 panel):
      Ridgeline KDE for top-k patches ranked by |median SHAP|.
    """
    median_shap, sd_shap, snr, kde_models, img_df = get_image_patch_stats(dist_df, image_idx)

    true_cls = CIFAR10_CLASSES[labels[image_idx]]
    pred_cls = CIFAR10_CLASSES[predicted[image_idx]]
    match    = "✓" if labels[image_idx] == predicted[image_idx] else "✗"

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        # f"Image {image_idx}  |  True: {true_cls}  |  Predicted: {pred_cls} {match}",
        f"True: {true_cls}  |  Predicted: {pred_cls} {match}",
        fontsize=20, fontweight="bold", y=1,
    )
    # gs = fig.add_gridspec(
    #     2, 4,
    #     height_ratios=[1.4, 0.8],
    #     hspace=0.35, wspace=0.30,
    #     top=0.93, bottom=0.05,
    # )
    gs = fig.add_gridspec(
        1, 4,
        wspace=0.30,
        top=0.93, bottom=0.05,
    )

    # ── Panel 0: original image ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(images[image_idx])
    ax0.set_title("Original Image", fontsize=16)
    ax0.axis("off")

    # ── Panel 1: median SHAP overlaid on image ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    img_h, img_w = images[image_idx].shape[:2]
    median_up = _upsample_patch_map(median_shap, img_size=(img_h, img_w))
    vabs = max(np.abs(median_shap).max(), 1e-12)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
    ax1.imshow(images[image_idx], alpha=0.45)
    im1 = ax1.imshow(median_up, cmap="RdBu_r", norm=norm, alpha=0.6)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    # patch grid lines — spacing in image pixels
    step_h = img_h / PATCH_GRID[0]
    step_w = img_w / PATCH_GRID[1]
    for i in range(1, PATCH_GRID[0]):
        ax1.axhline(i * step_h - 0.5, color="white", lw=0.4, alpha=0.5)
        ax1.axvline(i * step_w - 0.5, color="white", lw=0.4, alpha=0.5)
    # ax1.set_title("Median SHAP per Patch\n(red=+ toward pred class)", fontsize=16)
    ax1.set_title("Median SHAP per Patch", fontsize=15, pad=12)
    ax1.axis("off")

    # ── Panel 2: std-dev heatmap ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(sd_shap, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    # ax2.set_title("SHAP Std Dev\n(variability across runs)", fontsize=16)
    ax2.set_title("SHAP Std Dev", fontsize=15, pad=12)
    _style_patch_ax(ax2)

    # ── Panel 3: SNR heatmap ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(snr, cmap="Greens", interpolation="nearest")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("RoSHAP", fontsize=15, pad=12)
    _style_patch_ax(ax3)

    # ── Bottom row: ridgeline KDE for top-k patches ────────────────────────────
    # abs_median = img_df["nonzero_median"].abs().values
    # top_order  = np.argsort(abs_median)[::-1][:top_k]

    # # global x range — use raw data bounds (nonzero_min/max) across top patches,
    # # then clip to percentiles to avoid outlier patches dominating the x axis
    # x_vals = []
    # for patch_flat_idx in top_order:
    #     model = kde_models[patch_flat_idx]
    #     if model is not None and model.get("kde") is not None:
    #         x_vals += [model["nonzero_min"], model["nonzero_max"]]
    # if x_vals:
    #     x_arr = np.array(x_vals)
    #     x_min = np.nanpercentile(x_arr, 0.1)
    #     x_max = np.nanpercentile(x_arr, 99.9)
    #     # small tail pad so KDE curves don't get clipped at the edge
    #     pad = np.mean([kde_models[p]["bandwidth"] for p in top_order
    #                    if kde_models[p] is not None and kde_models[p].get("kde") is not None])
    #     x_min -= 3 * pad
    #     x_max += 3 * pad
    # else:
    #     x_min, x_max = -1e-6, 1e-6
    # x_grid = np.linspace(x_min, x_max, 500)

    # spacing = 0.50
    # ridge_scale = 0.40

    # ax_r = fig.add_subplot(gs[1, :])
    # ax_r.set_facecolor("#EBEBEB")
    # ax_r.set_axisbelow(True)
    # ax_r.grid(axis="x", color="white", linewidth=1.0, zorder=0)

    # patch_labels = []
    # for plot_pos, patch_flat_idx in enumerate(top_order):
    #     model     = kde_models[patch_flat_idx]
    #     row, col  = divmod(int(img_df["feature"].iloc[patch_flat_idx]), PATCH_GRID[1])
    #     patch_labels.append(f"({row},{col})")
    #     y_base = (top_k - 1 - plot_pos) * spacing

    #     if model is not None and model.get("kde") is not None:
    #         y_pdf = zero_inflated_kde_pdf(x_grid, model)
    #         peak  = y_pdf.max()
    #         dens  = (y_pdf / peak * ridge_scale) if peak > 0 else np.zeros_like(x_grid)
    #     else:
    #         dens = np.zeros_like(x_grid)

    #     ax_r.fill_between(
    #         x_grid, y_base, y_base + dens,
    #         facecolor="lightgray", edgecolor="black", linewidth=0.7,
    #         alpha=1.0, zorder=10 + plot_pos,
    #     )
    #     ax_r.plot(x_grid, y_base + dens, color="black", linewidth=0.7, zorder=11 + plot_pos)
    #     ax_r.hlines(y_base, x_min, x_max, color="black", linewidth=0.4, zorder=12 + plot_pos)

    # ax_r.set_yticks([(top_k - 1 - i) * spacing for i in range(top_k)])
    # ax_r.set_yticklabels(patch_labels, fontsize=8)
    # ax_r.set_xlabel("SHAP value", fontsize=10)
    # ax_r.set_xlim(x_min, x_max)
    # ax_r.set_ylim(-0.12, (top_k - 1) * spacing + ridge_scale + 0.12)
    # for spine in ax_r.spines.values():
    #     spine.set_visible(False)
    # ax_r.tick_params(axis="x", labelsize=8, length=0)
    # ax_r.tick_params(axis="y", labelsize=8, length=0)
    # ax_r.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=5)

    return fig


def _style_patch_ax(ax):
    ph, pw = PATCH_GRID
    ax.set_xticks(range(0, pw, max(1, pw // 8)))
    ax.set_yticks(range(0, ph, max(1, ph // 8)))
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
    ph, pw = PATCH_GRID
    ax.set_xlabel(f"Patch index (0–{n_patches-1}, row-major in {ph}×{pw} grid)")
    ax.set_title(f"Per-patch Median SHAP across {len(image_ids)} images")

    # thin column separators every pw patches (one per row boundary)
    for x in range(pw, n_patches, pw):
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

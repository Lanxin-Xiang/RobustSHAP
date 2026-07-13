# roshap

**Robust SHAP feature importance** — bootstrap SHAP with zero-inflated density
estimation and a signal-to-noise-weighted robust ranking (the *RoSHAP* statistic).

Single SHAP runs can rank unstable features highly: a feature may look important
in one train/test split and vanish in the next. `roshap` quantifies that
instability by refitting the model on bootstrap resamples, computing SHAP on the
out-of-bag samples each time, and modelling each feature's attribution
distribution across resamples — then ranks features by how *consistently* they
matter, not just how much they matter once.

The original paper experiments and notebooks live in [research/](research/).

## Install

```bash
pip install git+https://github.com/Lanxin-Xiang/RobustSHAP.git
```

(A PyPI release is coming; once it is live, plain `pip install roshap` will
work too.)

This installs everything needed to run roshap with its default model
(XGBoost) or any scikit-learn estimator — including numpy, pandas, shap, and
xgboost themselves.

### Optional model backends

LightGBM, CatBoost, PyTorch, and Keras/TensorFlow are **not** installed by
default. To use one of them as the model inside the bootstrap, add the
matching extra in square brackets:

```bash
pip install "roshap[lightgbm] @ git+https://github.com/Lanxin-Xiang/RobustSHAP.git"
```

| Extra | Adds support for |
|---|---|
| `lightgbm` | LightGBM models |
| `catboost` | CatBoost models |
| `torch` | PyTorch models (Deep/Gradient SHAP) |
| `tensorflow` | Keras/TensorFlow models |
| `smote` | SMOTE oversampling inside each bootstrap |
| `all` | all of the above |

### Developing roshap itself

```bash
git clone https://github.com/Lanxin-Xiang/RobustSHAP.git && cd RobustSHAP
pip install -e ".[dev]"
pytest   # 60 tests, ~2 s
```

## Quickstart

```python
import roshap

result = roshap.explain(
    X, y,                       # DataFrame + Series; classification or regression
    model="xgboost",            # or "lightgbm" / "catboost" / any ModelWrapper
    model_params={"eta": 0.05, "max_depth": 6},   # optional booster params
    task="auto",                # binary / multiclass / regression (auto-detected)
    n_bootstrap=500,
    inner_variance="permutation",
    approx="kde",               # or "normal" for a parametric approximation
    random_state=42,
    n_jobs=-1,
)

result.top_features(k=15)       # DataFrame: feature, RoSHAP, SNR, p_nonzero, ...
result.plot_ridge(top_k=15)     # ridge plot of attribution distributions
result.plot_top_features(top_k=15)          # bar chart with error bars
result.plot_feature("my_feature")           # one feature's fitted density + histogram
result.boot_results             # raw per-bootstrap aggregated SHAP frames
```

Any scikit-learn estimator works through the wrapper layer:

```python
from sklearn.ensemble import RandomForestClassifier

wrapper = roshap.SklearnWrapper(RandomForestClassifier, {"n_estimators": 200})
result = roshap.explain(X, y, model=wrapper, task="binary", n_bootstrap=200)
```

Sample-level evaluation (per-sample attribution distributions) needs the raw
SHAP rows:

```python
result = roshap.explain(X, y, keep="all", n_bootstrap=200)
result.sample_feature_distributions()       # per-(sample, feature) density fits
result.plot_sample(sample_id=3, top_k=10)   # top features for one sample
```

## Bring your own SHAP values

If SHAP values come from a model `roshap` doesn't train — a transformer, an
image classifier, anything — pass them directly and get the same distribution
estimation, ranking, and plots:

```python
import shap

explainer = shap.Explainer(my_transformer_pipeline)
sv = explainer(texts).values                # (n_samples, n_features)

result = roshap.explain_shap_values(sv, feature_names=feature_names)
result.top_features(k=20)
result.plot_sample(sample_id=0)             # sample-level evaluation included
```

Accepted forms: a single 2D `(n_samples, n_features)` array (samples become the
attribution draws), a 3D `(n_samples, n_classes, n_features)` multiclass array,
a **list** of such arrays (one per retrained replicate), or a long DataFrame
with `sample_id, feature, shap_value` columns.

## Method

1. **Bootstrap SHAP.** For each of `n_bootstrap` resamples, the model is refit
   on a (stratified) bootstrap sample and SHAP values are computed on the
   out-of-bag samples. With `inner_variance="permutation"`, feature columns are
   permuted before each fit (and un-permuted afterwards) to decorrelate
   tie-breaking artifacts. `|SHAP|` is aggregated per feature per resample.

2. **Zero-inflated density estimation.** Each feature's aggregated-|SHAP|
   distribution across resamples is modelled as a point mass at zero (the
   feature was not used) plus a continuous density on the positive reals —
   a log-KDE by default, or a lognormal with `approx="normal"`.

3. **RoSHAP ranking.** Features are ranked by

   `roshap_stat = p_nonzero × nonzero_median × (median / sd)`

   — the probability the feature matters at all, times its typical
   contribution, weighted by its signal-to-noise ratio across resamples.
   Features that are large *and stable* rank above features that are large
   but erratic.

## Low-level API

Every pipeline stage is importable directly for custom workflows:
`boot_multi_repeat_inference_keep_feature`, `boot_multi_repeat_inference_keep_all`,
`estimate_feature_level_mixture_preagg`, `estimate_sample_feature_distribution`,
`fit_zero_inflated_kde`, `fit_zero_inflated_normal`, `add_roshap_stat`,
`rank_features`, and the `roshap.plotting` module.

## License

MIT

"""Input validation, task inference, and seed handling."""
import warnings

import numpy as np
import pandas as pd


def infer_task(y):
    """Infer "binary", "multiclass", or "regression" from an outcome vector.

    Heuristic: bool / object / category dtypes are classification; numeric
    values are classification only when they are all integral with at most
    10 distinct values. Two classes -> "binary", more -> "multiclass".
    """
    y = pd.Series(y)
    n_unique = y.nunique(dropna=True)

    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        is_classification = True
    else:
        vals = y.dropna().to_numpy(dtype=float)
        is_classification = bool(np.all(vals == np.round(vals))) and n_unique <= 10

    if is_classification:
        if n_unique < 2:
            raise ValueError(f"y has {n_unique} unique value(s); need at least 2 classes.")
        task = "binary" if n_unique == 2 else "multiclass"
    else:
        task = "regression"

    warnings.warn(
        f"task='auto' resolved to '{task}'; pass task= explicitly to override.",
        stacklevel=3,
    )
    return task


def encode_labels(y, task):
    """Encode classification labels to 0..K-1 integers when needed.

    Returns (y_encoded, classes_) where classes_ is None if no encoding was
    applied (regression, or labels already 0..K-1 integers).
    """
    y = pd.Series(y)
    if task == "regression":
        return y, None

    vals = y.unique()
    expected = set(range(len(vals)))
    try:
        already_encoded = set(int(v) for v in vals) == expected and np.all(
            y.to_numpy(dtype=float) == y.to_numpy(dtype=float).round()
        )
    except (TypeError, ValueError):
        already_encoded = False
    if already_encoded:
        return y.astype(int), None

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    encoded = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    return encoded, np.asarray(le.classes_)


def spawn_bootstrap_seeds(random_state, n_bootstrap):
    """Derive one independent seed per bootstrap iteration from a single seed."""
    rng = np.random.default_rng(random_state)
    return rng.integers(0, 2**31 - 1, size=n_bootstrap).tolist()


def validate_X_y(X, y):
    """Coerce X/y to an index-aligned DataFrame/Series pair with a unique index.

    The bootstrap OOB logic identifies out-of-bag rows via ``~X.index.isin(...)``,
    which requires unique index labels.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
        X.columns = [f"f{j}" for j in range(X.shape[1])]

    if isinstance(y, pd.Series):
        if not y.index.equals(X.index):
            if len(y) != len(X):
                raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}.")
            y = pd.Series(y.to_numpy(), index=X.index, name=y.name)
    else:
        y_arr = np.asarray(y).reshape(-1)
        if len(y_arr) != len(X):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y_arr)}.")
        y = pd.Series(y_arr, index=X.index)

    if not X.index.is_unique:
        warnings.warn(
            "X has a non-unique index; resetting to a RangeIndex so out-of-bag "
            "samples can be identified.",
            stacklevel=3,
        )
        X = X.reset_index(drop=True)
        y = pd.Series(y.to_numpy(), index=X.index, name=y.name)

    return X, y

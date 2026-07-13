import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from roshap import XGBoostWrapper


def _to_frame(X):
    return pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])


@pytest.fixture(scope="session")
def binary_data():
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=4, n_redundant=2, random_state=0
    )
    return _to_frame(X), pd.Series(y)


@pytest.fixture(scope="session")
def multiclass_data():
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=5, n_redundant=1,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    return _to_frame(X), pd.Series(y)


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=120, n_features=8, n_informative=4, noise=0.5, random_state=0)
    return _to_frame(X), pd.Series(y)


@pytest.fixture
def xgb_wrapper_binary():
    return XGBoostWrapper(
        params={"objective": "binary:logistic", "eval_metric": "logloss", "seed": 0, "nthread": 1},
        num_boost_round=10,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)

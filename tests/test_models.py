import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from roshap import SklearnWrapper, XGBoostWrapper, create_model_wrapper


class TestXGBoostWrapper:
    def test_binary_shap_shape(self, binary_data):
        X, y = binary_data
        w = XGBoostWrapper({"objective": "binary:logistic", "seed": 0}, num_boost_round=10)
        w.fit(X, y)
        sv = w.compute_shap(X, task="binary")
        assert sv.shape == (len(X), X.shape[1])

    def test_multiclass_shap_shape(self, multiclass_data):
        X, y = multiclass_data
        w = XGBoostWrapper(
            {"objective": "multi:softprob", "num_class": 3, "seed": 0}, num_boost_round=10
        )
        w.fit(X, y)
        sv = w.compute_shap(X, task="multiclass")
        assert sv.shape == (len(X), 3, X.shape[1])

    def test_regression_shap_shape(self, regression_data):
        X, y = regression_data
        w = XGBoostWrapper({"objective": "reg:squarederror", "seed": 0}, num_boost_round=10)
        w.fit(X, y)
        sv = w.compute_shap(X, task="regression")
        assert sv.shape == (len(X), X.shape[1])


class TestSklearnWrapper:
    def test_binary_tree_shap_shape(self, binary_data):
        X, y = binary_data
        w = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10, "random_state": 0})
        w.fit(X, y)
        sv = w.compute_shap(X, task="binary")
        assert sv.shape == (len(X), X.shape[1])

    def test_multiclass_tree_shap_shape(self, multiclass_data):
        X, y = multiclass_data
        w = SklearnWrapper(RandomForestClassifier, {"n_estimators": 10, "random_state": 0})
        w.fit(X, y)
        sv = w.compute_shap(X, task="multiclass")
        assert sv.shape == (len(X), 3, X.shape[1])

    def test_regression_tree_shap_shape(self, regression_data):
        X, y = regression_data
        w = SklearnWrapper(RandomForestRegressor, {"n_estimators": 10, "random_state": 0})
        w.fit(X, y)
        sv = w.compute_shap(X, task="regression")
        assert sv.shape == (len(X), X.shape[1])
        assert np.isfinite(sv).all()

    def test_regression_linear_shap_shape(self, regression_data):
        X, y = regression_data
        w = SklearnWrapper(LinearRegression, use_tree_explainer=False, use_linear_explainer=True)
        w.fit(X, y)
        sv = w.compute_shap(X, task="regression")
        assert sv.shape == (len(X), X.shape[1])


class TestFactory:
    def test_xgboost(self):
        w = create_model_wrapper("xgboost", params={"objective": "binary:logistic"})
        assert isinstance(w, XGBoostWrapper)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model_wrapper("nope")


class TestOptionalWrappers:
    def test_lightgbm_binary(self, binary_data):
        pytest.importorskip("lightgbm")
        from roshap import LightGBMWrapper

        X, y = binary_data
        w = LightGBMWrapper({"objective": "binary", "verbose": -1, "seed": 0}, num_boost_round=10)
        w.fit(X, y)
        sv = w.compute_shap(X, task="binary")
        assert sv.shape == (len(X), X.shape[1])

    def test_catboost_binary(self, binary_data):
        pytest.importorskip("catboost")
        from roshap import CatBoostWrapper

        X, y = binary_data
        w = CatBoostWrapper({"loss_function": "Logloss", "random_seed": 0}, num_boost_round=10)
        w.fit(X, y)
        sv = w.compute_shap(X, task="binary")
        assert sv.shape == (len(X), X.shape[1])

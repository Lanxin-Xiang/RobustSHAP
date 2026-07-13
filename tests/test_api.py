import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import roshap
from roshap import RoshapResult, explain, explain_shap_values
from roshap._utils import infer_task

EXPLAIN_KW = dict(n_bootstrap=6, n_jobs=1, show_progress=False, num_boost_round=8)


class TestInferTask:
    def test_binary_int(self):
        with pytest.warns(UserWarning, match="resolved to 'binary'"):
            assert infer_task(pd.Series([0, 1, 0, 1])) == "binary"

    def test_multiclass_str(self):
        with pytest.warns(UserWarning, match="resolved to 'multiclass'"):
            assert infer_task(pd.Series(["a", "b", "c", "a"])) == "multiclass"

    def test_regression_float(self):
        with pytest.warns(UserWarning, match="resolved to 'regression'"):
            assert infer_task(pd.Series(np.linspace(0, 5, 50))) == "regression"


class TestExplainEndToEnd:
    def test_binary(self, binary_data):
        X, y = binary_data
        res = explain(X, y, task="binary", random_state=0, **EXPLAIN_KW)
        assert isinstance(res, RoshapResult)
        assert res.task == "binary"
        assert len(res.boot_results) == 6
        assert not res.feature_stats.empty
        assert {"SNR", "roshap_stat"}.issubset(res.ranking_.columns)

        top = res.top_features(k=3)
        assert len(top) == 3
        assert top.index.tolist() == [1, 2, 3]

        fig = res.plot_ridge(top_k=3, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multiclass(self, multiclass_data):
        X, y = multiclass_data
        res = explain(X, y, task="multiclass", random_state=0, **EXPLAIN_KW)
        assert "class_id" in res.feature_stats.columns
        top = res.top_features(k=3)          # averaged across classes
        assert len(top) == 3
        top_c1 = res.top_features(k=3, class_id=1)
        assert len(top_c1) == 3

    def test_regression(self, regression_data):
        X, y = regression_data
        res = explain(X, y, task="regression", random_state=0, **EXPLAIN_KW)
        assert res.task == "regression"
        assert not res.top_features(k=5).empty

    def test_string_labels_encoded(self, binary_data):
        X, y = binary_data
        y_str = y.map({0: "no", 1: "yes"})
        res = explain(X, y_str, task="binary", random_state=0, **EXPLAIN_KW)
        assert list(res.classes_) == ["no", "yes"]

    def test_seed_determinism(self, binary_data):
        X, y = binary_data
        r1 = explain(X, y, task="binary", random_state=7, **EXPLAIN_KW)
        r2 = explain(X, y, task="binary", random_state=7, **EXPLAIN_KW)
        pd.testing.assert_series_equal(r1.ranking_["roshap_stat"], r2.ranking_["roshap_stat"])

        r3 = explain(X, y, task="binary", random_state=8, **EXPLAIN_KW)
        assert not np.allclose(
            r1.ranking_["roshap_stat"].to_numpy(), r3.ranking_["roshap_stat"].to_numpy()
        )

    def test_keep_all_enables_sample_level(self, binary_data):
        X, y = binary_data
        res = explain(X, y, task="binary", keep="all", random_state=0, **EXPLAIN_KW)
        assert res.raw_results is not None
        mix_df = res.sample_feature_distributions()
        assert {"sample_id", "feature", "kde_model"}.issubset(mix_df.columns)

        sid = mix_df["sample_id"].iloc[0]
        fig, top_df = res.plot_sample(sid, top_k=3, xlim=(-2, 2), show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_keep_feature_blocks_sample_level(self, binary_data):
        X, y = binary_data
        res = explain(X, y, task="binary", random_state=0, **EXPLAIN_KW)
        with pytest.raises(ValueError, match="keep='all'"):
            res.sample_feature_distributions()

    def test_normal_approx(self, binary_data):
        X, y = binary_data
        res = explain(X, y, task="binary", approx="normal", random_state=0, **EXPLAIN_KW)
        assert all(m["dist"] == "normal" for m in res.feature_stats["kde_model"])
        assert not res.top_features(k=3).empty

    def test_custom_wrapper(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier

        X, y = binary_data
        wrapper = roshap.SklearnWrapper(
            RandomForestClassifier, {"n_estimators": 10, "random_state": 0}
        )
        res = explain(X, y, model=wrapper, task="binary", random_state=0,
                      n_bootstrap=3, n_jobs=1, show_progress=False)
        assert not res.feature_stats.empty

    def test_unknown_model_string_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="Unknown model string"):
            explain(X, y, model="mystery", task="binary", **EXPLAIN_KW)

    def test_nonunique_index_warns_and_works(self, binary_data):
        X, y = binary_data
        X_dup = X.copy()
        X_dup.index = [0] * len(X_dup)
        y_dup = pd.Series(y.to_numpy(), index=X_dup.index)
        with pytest.warns(UserWarning, match="non-unique index"):
            res = explain(X_dup, y_dup, task="binary", random_state=0,
                          n_bootstrap=3, n_jobs=1, show_progress=False)
        assert not res.feature_stats.empty


class TestExplainShapValues:
    def test_single_2d_array(self, rng):
        sv = rng.normal(size=(40, 5))
        res = explain_shap_values(sv, feature_names=list("abcde"))
        assert len(res.boot_results) == 40           # samples as pseudo-draws
        assert res.raw_results is not None
        assert len(res.top_features(k=5)) == 5
        fig = res.plot_ridge(top_k=3, show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_3d_array_multiclass(self, rng):
        sv = rng.normal(size=(20, 3, 4))
        res = explain_shap_values(sv, task="multiclass")
        assert "class_id" in res.feature_stats.columns

    def test_list_of_arrays(self, rng):
        sv_list = [rng.normal(size=(30, 4)) for _ in range(5)]
        res = explain_shap_values(sv_list)
        assert len(res.boot_results) == 5
        assert len(res.top_features(k=4)) == 4

    def test_long_dataframe(self, rng):
        rows = []
        for b in range(4):
            for sid in range(10):
                for feat in ("x1", "x2", "x3"):
                    rows.append({
                        "sample_id": sid, "feature": feat,
                        "shap_value": rng.normal(), "bootstrap_id": b,
                    })
        res = explain_shap_values(pd.DataFrame(rows))
        assert len(res.boot_results) == 4
        assert set(res.feature_names) == {"x1", "x2", "x3"}

    def test_sample_level_evaluation(self, rng):
        sv = rng.normal(size=(15, 4))
        res = explain_shap_values(sv)
        mix_df = res.sample_feature_distributions()
        assert mix_df["sample_id"].nunique() == 15

    def test_default_feature_names(self, rng):
        sv = rng.normal(size=(10, 3))
        res = explain_shap_values(sv)
        assert res.feature_names == ["f0", "f1", "f2"]

    def test_normal_approx(self, rng):
        sv = rng.normal(size=(30, 4))
        res = explain_shap_values(sv, approx="normal")
        assert all(m["dist"] == "normal" for m in res.feature_stats["kde_model"])

"""roshap quickstart: bootstrap-SHAP robust feature importance on synthetic data.

Run:  python examples/quickstart.py
"""
import matplotlib

matplotlib.use("Agg")

import pandas as pd
from sklearn.datasets import make_classification

import roshap


def main():
    X_arr, y = make_classification(
        n_samples=300, n_features=12, n_informative=4, n_redundant=3, random_state=0
    )
    X = pd.DataFrame(X_arr, columns=[f"feature_{j}" for j in range(X_arr.shape[1])])

    result = roshap.explain(
        X, y,
        model="xgboost",
        task="binary",
        n_bootstrap=30,
        inner_variance="permutation",
        random_state=42,
        n_jobs=-1,
        num_boost_round=50,
    )

    print("\nTop 10 features by RoSHAP stat:")
    print(result.top_features(k=10)[["feature", "importance", "SNR", "p_nonzero"]].round(3))

    fig = result.plot_ridge(top_k=10, show=False)
    fig.savefig("ridge.png", dpi=150)
    print("\nSaved ridge plot to ridge.png")

    fig, _ = result.plot_top_features(top_k=10, show=False)
    fig.savefig("top_features.png", dpi=150)
    print("Saved top-features bar chart to top_features.png")


if __name__ == "__main__":
    main()

from types import SimpleNamespace

import numpy as np
import polars as pl

from track_dashboard.confirmation import MLModelSpec
from track_dashboard.analysis.feature_analysis import (
    analyze_features,
    build_default_model,
)


def make_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "signal": [9.0, 10.0, 8.0, 9.0, 1.0, 2.0, 2.0, 1.0],
            "noise": [1.0, 3.0, 2.0, 4.0, 1.5, 3.5, 2.5, 4.5],
            "label": ["matched"] * 4 + ["unmatched"] * 4,
        }
    )


def test_feature_analysis_only_includes_selected_features():
    result = analyze_features(
        make_data(),
        features=["signal"],
        methods=["Point-biserial correlation"],
        label_col="label",
        matched_value="matched",
    )

    assert result.get_column("feature").to_list() == ["signal"]
    assert result.row(0, named=True)["score"] > 0.9


def test_mutual_information_and_auc_rank_predictive_feature_higher():
    result = analyze_features(
        make_data(),
        features=["signal", "noise"],
        methods=["Mutual information", "Univariate ROC AUC"],
        label_col="label",
        matched_value="matched",
    )

    for method in ["Mutual information", "Univariate ROC AUC"]:
        method_rows = result.filter(pl.col("method") == method)
        scores = dict(
            method_rows.select("feature", "importance").iter_rows()
        )
        assert scores["signal"] > scores["noise"]


def test_permutation_importance_uses_selected_registered_model_features():
    model = MLModelSpec(
        name="signal_model",
        model=lambda features: features.get_column("signal") / 10,
        features=("signal", "noise"),
    )

    result = analyze_features(
        make_data(),
        features=["signal"],
        methods=["Permutation importance"],
        label_col="label",
        matched_value="matched",
        model=model,
    )

    assert result.get_column("feature").to_list() == ["signal"]
    assert result.row(0, named=True)["importance"] > 0


def test_default_model_can_be_built_for_feature_importance():
    model = build_default_model(
        make_data(),
        features=["signal", "noise"],
        label_col="label",
        positive_class="matched",
        model_type="Logistic regression",
    )

    result = analyze_features(
        make_data(),
        features=["signal", "noise"],
        methods=["Permutation importance"],
        label_col="label",
        matched_value="matched",
        model=model,
    )

    assert model.name == "Default Logistic regression"
    assert set(result["feature"]) == {"signal", "noise"}


def test_default_random_forest_accepts_hyperparameters_and_class_weights():
    model = build_default_model(
        make_data(),
        features=["signal", "noise"],
        label_col="label",
        positive_class="matched",
        model_type="Random forest",
        class_weight_mode="Custom",
        positive_class_weight=3.0,
        negative_class_weight=1.0,
        random_forest_params={
            "n_estimators": 25,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": None,
            "criterion": "entropy",
            "bootstrap": False,
        },
    )

    forest = model.model.steps[-1][1]
    assert forest.n_estimators == 25
    assert forest.max_depth == 4
    assert forest.max_features is None
    assert forest.criterion == "entropy"
    assert forest.bootstrap is False


def test_other_default_models_accept_hyperparameters():
    logistic = build_default_model(
        make_data(),
        features=["signal", "noise"],
        label_col="label",
        positive_class="matched",
        model_type="Logistic regression",
        logistic_regression_params={
            "C": 0.25,
            "penalty": "l1",
            "max_iter": 250,
            "fit_intercept": False,
        },
    )
    logistic_estimator = logistic.model.steps[-1][1]
    assert logistic_estimator.C == 0.25
    assert (
        logistic_estimator.penalty == "l1"
        or logistic_estimator.l1_ratio == 1.0
    )
    assert logistic_estimator.max_iter == 250
    assert logistic_estimator.fit_intercept is False

    boosting = build_default_model(
        make_data(),
        features=["signal", "noise"],
        label_col="label",
        positive_class="matched",
        model_type="Gradient boosting",
        gradient_boosting_params={
            "n_estimators": 25,
            "learning_rate": 0.2,
            "max_depth": 2,
            "min_samples_leaf": 2,
            "subsample": 0.75,
        },
    )
    boosting_estimator = boosting.model.steps[-1][1]
    assert boosting_estimator.n_estimators == 25
    assert boosting_estimator.learning_rate == 0.2
    assert boosting_estimator.max_depth == 2
    assert boosting_estimator.min_samples_leaf == 2
    assert boosting_estimator.subsample == 0.75


def test_incremental_auc_measures_information_beyond_other_features():
    data = pl.DataFrame(
        {
            "signal": [float(index >= 50) for index in range(100)],
            "noise": [float((index * 17) % 23) for index in range(100)],
            "outcome": ["yes" if index >= 50 else "no" for index in range(100)],
        }
    )

    result = analyze_features(
        data,
        features=["signal", "noise"],
        methods=["Incremental CV ROC AUC"],
        label_col="outcome",
        matched_value="yes",
    )

    gains = dict(result.select("feature", "score").iter_rows())
    assert gains["signal"] > 0.4
    assert gains["noise"] <= gains["signal"]


def test_shap_only_reports_selected_model_features(monkeypatch):
    class FakeExplainer:
        def __init__(self, predict, background, algorithm):
            assert algorithm == "permutation"
            predict(background)

        def __call__(self, matrix):
            values = np.tile([2.0, 0.5], (len(matrix), 1))
            return SimpleNamespace(values=values)

    monkeypatch.setitem(
        __import__("sys").modules,
        "shap",
        SimpleNamespace(Explainer=FakeExplainer),
    )
    model = MLModelSpec(
        name="signal_model",
        model=lambda features: features.get_column("signal") / 10,
        features=("signal", "noise"),
    )

    result = analyze_features(
        make_data(),
        features=["signal"],
        methods=["SHAP"],
        label_col="label",
        matched_value="matched",
        model=model,
    )

    assert result.get_column("feature").to_list() == ["signal"]
    assert result.row(0, named=True)["importance"] == 2.0

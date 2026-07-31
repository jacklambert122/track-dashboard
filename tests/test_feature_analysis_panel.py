import polars as pl
import panel as pn

from track_dashboard import DashboardApp
from track_dashboard.confirmation import MLModelSpec
from track_dashboard.confirmation.dashboard import TrackConfirmationDashboard
from track_dashboard.confirmation.example import make_example_confirmation_config
from track_dashboard.example_data import make_example_data


def test_analysis_dashboard_runs_feature_analysis_for_selected_features():
    dashboard = DashboardApp(
        make_example_data(tracks=8, points_per_track=5)
    )
    panel = dashboard.feature_analysis
    panel.features = ["snr", "residual"]
    panel.methods = [
        "Point-biserial correlation",
        "Mutual information",
        "Full correlation matrix",
    ]

    panel._run_analysis()

    assert set(panel._result["feature"]) == {"snr", "residual"}
    assert set(panel._result["method"]) == {
        "Point-biserial correlation",
        "Mutual information",
    }
    assert panel._correlation_matrix.height == 9
    matrix_names = set(panel._correlation_matrix["row"])
    assert matrix_names == {"snr", "residual", "label = matched"}
    assert panel.message.startswith("Analyzed 2")
    assert panel.results() is not None


def test_correlation_matrix_only_appears_when_selected():
    panel = DashboardApp(
        make_example_data(tracks=5, points_per_track=4)
    ).feature_analysis
    panel.features = ["snr", "residual"]
    panel.methods = ["Mutual information"]

    panel._run_analysis()

    assert panel._correlation_matrix.is_empty()
    assert all(
        getattr(output, "title", None) != "Full correlation matrix"
        for output in panel.results()
    )

    panel.methods = ["Full correlation matrix"]
    panel._run_analysis()

    assert panel._result.is_empty()
    assert panel._correlation_matrix.height == 9
    assert any(
        getattr(output, "title", None) == "Full correlation matrix"
        for output in panel.results()
    )


def test_analysis_dashboard_allows_label_and_positive_class_selection():
    data = make_example_data(tracks=8, points_per_track=5).with_columns(
        pl.when(pl.col("label") == "matched")
        .then(pl.lit("accepted"))
        .otherwise(pl.lit("rejected"))
        .alias("review_status")
    )
    panel = DashboardApp(data).feature_analysis
    panel.label_column = "review_status"
    panel.positive_class = "accepted"
    panel.features = ["snr", "residual"]
    panel.methods = ["Incremental CV ROC AUC"]

    panel._run_analysis()

    assert set(panel._result["feature"]) == {"snr", "residual"}
    assert panel.message.startswith("Analyzed 2")


def test_analysis_dashboard_selects_default_or_provided_importance_model():
    data = make_example_data(tracks=8, points_per_track=5)
    provided = MLModelSpec(
        name="provided_snr",
        model=lambda features: features.get_column("snr") / 30,
        features=("snr",),
    )
    panel = DashboardApp(
        data,
        feature_analysis_models=[provided],
    ).feature_analysis
    panel.features = ["snr"]
    panel.methods = ["Permutation importance"]

    panel.model_source = "Build default model"
    panel.model_type = "Random forest"
    panel.class_weight_mode = "Custom"
    panel.positive_class_weight = 2.5
    panel.negative_class_weight = 1.0
    panel.rf_n_estimators = 25
    panel.rf_max_depth = 4
    panel.rf_min_samples_leaf = 1
    panel.rf_max_features = "All"
    panel.rf_criterion = "entropy"
    panel.rf_bootstrap = False
    panel._run_analysis()
    assert panel._result.height == 1

    panel.model_source = "Use provided model"
    panel.model = "provided_snr"
    panel._run_analysis()
    assert panel._result.height == 1


def test_shap_explainer_selector_only_appears_for_shap():
    panel = DashboardApp(
        make_example_data(tracks=4, points_per_track=3)
    ).feature_analysis

    panel.methods = ["Mutual information"]
    assert isinstance(panel.shap_options(), pn.Spacer)

    panel.methods = ["SHAP"]
    assert isinstance(panel.shap_options(), pn.widgets.Select)
    assert panel.param.shap_explainer.objects == [
        "Auto",
        "Tree",
        "Linear",
        "Permutation",
    ]


def test_shap_analysis_creates_beeswarm_values():
    panel = DashboardApp(
        make_example_data(tracks=6, points_per_track=4)
    ).feature_analysis
    panel.features = ["snr", "residual"]
    panel.methods = ["SHAP"]
    panel.model_type = "Random forest"
    panel.rf_n_estimators = 20
    panel.shap_explainer = "Tree"

    panel._run_analysis()

    assert panel._shap_values.height == 48
    assert set(panel._shap_values["feature"]) == {"snr", "residual"}
    assert panel._shap_beeswarm() is not None
    assert panel._shap_importance_bar() is not None
    assert len(panel._shap_base_values) == 24
    assert panel.shap_waterfall() is not None


def test_shap_beeswarm_offsets_show_x_density():
    offsets = DashboardApp(
        make_example_data(tracks=2, points_per_track=3)
    ).feature_analysis._density_offsets(
        [0.0, 0.01, 0.02, 0.03, 0.9]
    )

    dense_spread = max(offsets[:4]) - min(offsets[:4])
    assert dense_spread > 0
    assert offsets[-1] == 0


def test_confirmation_dashboard_no_longer_contains_feature_analysis():
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=4, points_per_track=3),
        make_example_confirmation_config(),
    )

    assert not hasattr(dashboard, "feature_analysis")

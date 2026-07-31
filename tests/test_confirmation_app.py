import json

from track_dashboard.confirmation import MLConfirmationPath, MLModelSpec
from track_dashboard.confirmation.dashboard import TrackConfirmationDashboard
from track_dashboard.confirmation.example import make_example_confirmation_config
from track_dashboard.example_data import make_example_data


def test_dashboard_adds_rules_and_downloads_nested_candidate_config():
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=3, points_per_track=4),
        make_example_confirmation_config(),
    )
    assert dashboard.enabled_default_paths == ["quality_path", "limb_path"]
    dashboard.enabled_default_paths = ["quality_path"]
    assert dashboard._comparison().default_path_columns == (
        "quality_path",
        "limb_path",
    )

    dashboard.path_name = "candidate"
    dashboard.feature = "snr"
    dashboard.minimum = 8.0

    dashboard._add_rule()
    dashboard.path_name = "second_candidate"
    dashboard.feature = "residual"
    dashboard.maximum = 3.0
    dashboard._add_rule()

    assert dashboard._comparison().candidate_path_columns == (
        "candidate",
        "second_candidate",
    )
    assert dashboard.first_confirmation_distributions() is not None
    assert dashboard.first_confirmation_details() is not None
    assert dashboard.path_summary() is not None
    changed_data = dashboard.changed_track_data()
    assert set(changed_data["confirmation_logic"]) <= {"Default", "Candidate"}
    assert "label" in changed_data.columns
    assert "snr" in changed_data.columns
    dashboard.changed_track_x = "snr"
    dashboard.changed_track_y = "residual"
    changed_data = dashboard.changed_track_data()
    assert {"snr", "residual"} <= set(changed_data.columns)
    assert dashboard.changed_tracks_view() is not None
    saved = json.load(dashboard._download_candidate_config())
    paths = saved["dynamic_specific"]["track_qa_config"]["experimental_paths"]
    enabled = saved["dynamic_specific"]["track_qa_config"]["enabled_paths"]
    assert enabled == ["quality_path"]
    assert paths == [
        {
            "name": "candidate",
            "ranges": {"snr": {"min": 8.0}},
        },
        {
            "name": "second_candidate",
            "ranges": {"residual": {"max": 3.0}},
        },
    ]


def test_dashboard_filters_recompute_confirmation_comparison():
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=4, points_per_track=5),
        make_example_confirmation_config(),
    )
    original_height = dashboard._comparison().points.height
    dashboard.filters._add_filter()
    row = dashboard.filters.rows[0]
    row.feature = "snr"
    row.lower = 20.0

    filtered = dashboard._comparison().points

    assert filtered.height < original_height
    assert filtered.get_column("snr").min() >= 20.0

    row.feature = "label"
    row.values = ["matched"]
    filtered = dashboard._comparison().points
    assert filtered.get_column("label").unique().to_list() == ["matched"]


def test_dashboard_identifies_tracks_lost_by_candidate_path_selection():
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=12, points_per_track=5),
        make_example_confirmation_config(),
    )
    dashboard.enabled_default_paths = []

    changed = dashboard._changed_tracks_frame()

    assert changed.height > 0
    assert set(changed["confirmation_change"]) == {"Lost"}
    lost_data = dashboard.changed_track_data()
    assert set(lost_data["confirmation_logic"]) == {"Default"}
    assert set(lost_data["confirmation_change"]) == {"Lost"}


def test_dashboard_adds_and_restores_registered_ml_paths():
    model = MLModelSpec(
        name="snr_model",
        model=lambda features: features.get_column("snr") / 30,
        features=("snr",),
    )
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=3, points_per_track=4),
        make_example_confirmation_config(),
        ml_models=[model],
    )
    dashboard.ml_path_name = "ml_quality"
    dashboard.ml_threshold = 0.6

    dashboard._add_ml_path()

    assert dashboard._comparison().candidate_path_columns == ("ml_quality",)
    saved = json.load(dashboard._download_candidate_config())
    ml_config = saved["dynamic_specific"]["track_qa_config"][
        "ml_confirmation_paths"
    ]
    assert ml_config == [
        {
            "name": "ml_quality",
            "model": "snr_model",
            "features": ["snr"],
            "threshold": 0.6,
            "positive_class_index": 1,
        }
    ]

    restored = TrackConfirmationDashboard(
        make_example_data(tracks=3, points_per_track=4),
        saved,
        ml_models=[model],
    )
    assert restored._comparison().candidate_path_columns == ("ml_quality",)


def test_dashboard_uses_provided_ml_path_as_existing_confirmation_logic():
    data = make_example_data(tracks=8, points_per_track=5)
    model = MLModelSpec(
        name="current_quality_model",
        model=lambda features: features.get_column("snr") / 30,
        features=("snr",),
    )
    default_path = MLConfirmationPath(
        path="current_ml_quality",
        model=model,
        threshold=0.55,
    )
    dashboard = TrackConfirmationDashboard(
        data,
        make_example_confirmation_config(),
        default_ml_paths=[default_path],
    )

    assert "current_ml_quality" in dashboard.available_default_paths
    assert "current_quality_model" in dashboard.param.ml_model.objects
    assert (
        dashboard._comparison()
        .points.get_column("current_ml_quality")
        .sum()
        > 0
    )
    dashboard.enabled_default_paths = [
        path
        for path in dashboard.enabled_default_paths
        if path != "current_ml_quality"
    ]
    assert dashboard._comparison().default_path_columns == (
        "quality_path",
        "limb_path",
        "current_ml_quality",
    )


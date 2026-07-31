import polars as pl

from track_dashboard.core.data_model import DataModel
from track_dashboard.core.state import DashboardState


def make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "snr": [1.0, 3.0, 10.0, 14.0],
            "label": ["a", "a", "b", "b"],
        }
    )


def test_point_mode_returns_filtered_points():
    state = DashboardState(make_df())
    state.filter_expressions = [pl.col("snr") > 3]
    model = DataModel(state)

    assert model.analysis_df().get_column("track_id").to_list() == [2, 2]


def test_track_aggregation_configuration_defaults_to_empty():
    state = DashboardState(make_df())

    assert state.track_agg_features == []
    assert state.track_agg_methods_by_feature == {}


def test_track_mode_uses_complete_matching_tracks():
    state = DashboardState(make_df())
    state.filter_expressions = [pl.col("snr") > 12]
    state.analysis_level = "Track"
    state.track_agg_features = ["snr"]
    state.track_agg_methods_by_feature = {"snr": ["mean"]}
    model = DataModel(state)

    result = model.analysis_df()

    assert result.get_column("track_id").to_list() == [2]
    assert result.get_column("snr_mean").to_list() == [12.0]


def test_track_mode_only_aggregates_selected_features():
    state = DashboardState(make_df())
    state.analysis_level = "Track"
    state.track_agg_features = []
    state.track_agg_methods_by_feature = {}
    model = DataModel(state)

    result = model.analysis_df()

    assert result.columns == ["track_id", "track_length", "label"]


def test_track_mode_uses_aggregation_methods_per_feature():
    df = make_df().with_columns((pl.col("snr") * 2).alias("score"))
    state = DashboardState(df)
    state.analysis_level = "Track"
    state.track_agg_features = ["snr", "score"]
    state.track_agg_methods_by_feature = {
        "snr": ["mean", "max"],
        "score": ["min"],
    }
    model = DataModel(state)

    result = model.analysis_df()

    assert "snr_mean" in result.columns
    assert "snr_max" in result.columns
    assert "score_min" in result.columns
    assert "score_mean" not in result.columns


def test_point_selection_label_updates_only_selected_rows():
    state = DashboardState(make_df())
    selected = state.point_df.filter(pl.col("snr") == 3.0)
    state.set_selection([1], selected)

    state.label_selection("review", "keep")

    assert state.point_df.get_column("review").to_list() == [
        None,
        "keep",
        None,
        None,
    ]
    assert state.selected_data.get_column("review").to_list() == ["keep"]
    assert "review" in DataModel(state).grouping_features()


def test_track_selection_label_updates_all_points_in_selected_tracks():
    state = DashboardState(make_df())
    state.analysis_level = "Track"
    state.set_selection([0], pl.DataFrame({"track_id": [2]}))

    state.label_selection("review", "reject")

    assert state.point_df.get_column("review").to_list() == [
        None,
        None,
        "reject",
        "reject",
    ]

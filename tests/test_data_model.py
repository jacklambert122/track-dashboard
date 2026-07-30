import polars as pl

from track_dashboard.data_model import DataModel
from track_dashboard.state import DashboardState


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


def test_track_mode_uses_complete_matching_tracks():
    state = DashboardState(make_df())
    state.filter_expressions = [pl.col("snr") > 12]
    state.analysis_level = "Track"
    state.track_agg_methods = ["mean"]
    model = DataModel(state)

    result = model.analysis_df()

    assert result.get_column("track_id").to_list() == [2]
    assert result.get_column("snr_mean").to_list() == [12.0]

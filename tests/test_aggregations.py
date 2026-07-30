import polars as pl

from track_dashboard.aggregations import aggregate_tracks


def test_aggregate_tracks_generates_named_metrics():
    df = pl.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "snr": [1.0, 3.0, 10.0, 14.0],
            "label": ["a", "a", "b", "b"],
        }
    )

    result = aggregate_tracks(
        df,
        track_id_col="track_id",
        methods=["mean", "max"],
    ).sort("track_id")

    assert result.columns == [
        "track_id",
        "track_length",
        "snr_mean",
        "snr_max",
        "label",
    ]
    assert result.get_column("snr_mean").to_list() == [2.0, 12.0]
    assert result.get_column("snr_max").to_list() == [3.0, 14.0]

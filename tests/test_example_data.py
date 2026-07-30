import polars as pl

from track_dashboard.example_data import make_example_data


def test_example_data_has_distinct_reproducible_tracks():
    first = make_example_data(tracks=4, points_per_track=5, seed=12)
    second = make_example_data(tracks=4, points_per_track=5, seed=12)

    assert first.equals(second)
    assert first.shape == (20, 10)
    assert first.columns[:5] == ["track_id", "x", "y", "frame", "time"]

    centroids = (
        first.group_by("track_id")
        .agg(pl.col("x").mean(), pl.col("y").mean())
        .select("x", "y")
    )
    assert centroids.unique().height == 4


def test_points_move_within_each_example_track():
    data = make_example_data(tracks=4, points_per_track=5)

    point_counts = data.group_by("track_id").agg(
        pl.struct("x", "y").n_unique().alias("unique_points")
    )
    assert point_counts.get_column("unique_points").to_list() == [5, 5, 5, 5]

from __future__ import annotations

import polars as pl

from .aggregations import aggregate_tracks
from .state import INTERNAL_ROW_ID, DashboardState


class DataModel:
    """Transforms shared point data into the active analysis dataframe."""

    def __init__(
        self,
        state: DashboardState,
        *,
        excluded_track_metrics: set[str] | None = None,
    ) -> None:
        self.state = state
        self.excluded_track_metrics = {
            *(excluded_track_metrics or set()),
            INTERNAL_ROW_ID,
        }

    def filtered_points(self) -> pl.DataFrame:
        df = self.state.point_df
        for expression in self.state.filter_expressions:
            df = df.filter(expression)
        return df

    def complete_matching_tracks(self) -> pl.DataFrame:
        matching_points = self.filtered_points()
        if matching_points.is_empty():
            return self.state.point_df.clear()

        track_ids = matching_points.get_column(self.state.track_id_col).unique()
        return self.state.point_df.filter(
            pl.col(self.state.track_id_col).is_in(track_ids)
        )

    def analysis_df(self) -> pl.DataFrame:
        if self.state.analysis_level == "Point":
            return self.filtered_points()

        return aggregate_tracks(
            self.complete_matching_tracks(),
            track_id_col=self.state.track_id_col,
            excluded_numeric_columns=self.excluded_track_metrics,
            included_numeric_columns=self.state.track_agg_features,
            methods_by_column=self.state.track_agg_methods_by_feature,
        )

    def track_aggregation_features(self) -> list[str]:
        return [
            column
            for column, dtype in self.state.point_df.schema.items()
            if column != self.state.track_id_col
            and column not in self.excluded_track_metrics
            and dtype.is_numeric()
        ]

    def numeric_features(self) -> list[str]:
        df = self.analysis_df()
        return [
            column
            for column, dtype in df.schema.items()
            if column not in {self.state.track_id_col, INTERNAL_ROW_ID}
            and dtype.is_numeric()
        ]

    def grouping_features(self) -> list[str]:
        return [
            column
            for column in self.analysis_df().columns
            if column not in {self.state.track_id_col, INTERNAL_ROW_ID}
        ]

from __future__ import annotations

import param
import polars as pl


class DashboardState(param.Parameterized):
    analysis_level = param.Selector(default="Point", objects=["Point", "Track"])
    track_agg_methods = param.ListSelector(
        default=["mean"],
        objects=["mean", "median", "min", "max", "std", "sum", "first", "last"],
    )
    filter_expressions = param.List(default=[])
    selected_indices = param.List(default=[])
    selected_track_ids = param.List(default=[])
    data_revision = param.Integer(default=0, precedence=-1)
    selection_revision = param.Integer(default=0, precedence=-1)

    def __init__(
        self,
        point_df: pl.DataFrame,
        *,
        track_id_col: str = "track_id",
        **params,
    ) -> None:
        if track_id_col not in point_df.columns:
            raise ValueError(f"Missing track ID column: {track_id_col!r}")

        super().__init__(**params)
        self.point_df = point_df
        self.track_id_col = track_id_col
        self.selected_data = point_df.clear()
        self.rendered_scatter_data = point_df.clear()

    def mark_data_changed(self) -> None:
        self.data_revision += 1
        self.clear_selection()

    def set_selection(self, indices: list[int], data: pl.DataFrame) -> None:
        self.selected_indices = indices
        self.selected_data = data
        self.selected_track_ids = (
            data.get_column(self.track_id_col).unique().to_list()
            if self.track_id_col in data.columns
            else []
        )
        self.selection_revision += 1

    def clear_selection(self) -> None:
        self.selected_indices = []
        self.selected_track_ids = []
        self.selected_data = self.point_df.clear()
        self.selection_revision += 1

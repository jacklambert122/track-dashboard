from __future__ import annotations

import param
import polars as pl

INTERNAL_ROW_ID = "__track_dashboard_row_id"


class DashboardState(param.Parameterized):
    analysis_level = param.Selector(default="Point", objects=["Point", "Track"])
    track_agg_features = param.ListSelector(default=[], objects=[])
    track_agg_methods_by_feature = param.Dict(default={})
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
        if INTERNAL_ROW_ID in point_df.columns:
            raise ValueError(
                f"Input data cannot contain reserved column {INTERNAL_ROW_ID!r}"
            )

        super().__init__(**params)
        self.point_df = point_df.with_row_index(INTERNAL_ROW_ID)
        self.track_id_col = track_id_col
        aggregatable = [
            column
            for column, dtype in point_df.schema.items()
            if column != track_id_col and dtype.is_numeric()
        ]
        self.param.track_agg_features.objects = aggregatable
        self.track_agg_features = []
        self.track_agg_methods_by_feature = {}
        self.selected_data = self.point_df.clear()
        self.rendered_scatter_data = self.point_df.clear()

    def selected_data_for_display(self) -> pl.DataFrame:
        return self.selected_data.drop(INTERNAL_ROW_ID, strict=False)

    def label_selection(self, column: str, value: str) -> None:
        column = column.strip()
        if not column:
            raise ValueError("Enter a label column name.")
        if column in {self.track_id_col, INTERNAL_ROW_ID}:
            raise ValueError(f"Cannot use reserved column name {column!r}.")
        if self.selected_data.is_empty():
            raise ValueError("Select data before applying a label.")

        if column not in self.point_df.columns:
            self.point_df = self.point_df.with_columns(
                pl.lit(None, dtype=pl.String).alias(column)
            )
        elif self.point_df.schema[column] != pl.String:
            raise ValueError(
                f"Existing label column {column!r} must contain string values."
            )

        if INTERNAL_ROW_ID in self.selected_data.columns:
            selected_values = self.selected_data.get_column(INTERNAL_ROW_ID)
            selected = pl.col(INTERNAL_ROW_ID).is_in(selected_values)
        else:
            selected_values = self.selected_data.get_column(
                self.track_id_col
            ).unique()
            selected = pl.col(self.track_id_col).is_in(selected_values)

        self.point_df = self.point_df.with_columns(
            pl.when(selected)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
        if INTERNAL_ROW_ID in self.selected_data.columns:
            self.selected_data = self.point_df.filter(selected)
        else:
            self.selected_data = self.selected_data.with_columns(
                pl.lit(value).alias(column)
            )
        self.data_revision += 1
        self.selection_revision += 1

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

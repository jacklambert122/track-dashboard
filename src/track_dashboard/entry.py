from __future__ import annotations

from io import BytesIO
from pathlib import Path

import panel as pn
import polars as pl

from .analysis.app import DashboardApp
from .confirmation.engine import MLModelSpec
from .example_data import make_example_data

SUPPORTED_INPUT_SUFFIXES = {".csv", ".parquet"}


def load_input_data(path: str | Path) -> pl.DataFrame:
    """Load a supported dashboard input file from disk."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(input_path)
    if suffix == ".parquet":
        return pl.read_parquet(input_path)
    raise ValueError(
        f"Unsupported input file {input_path!s}. Use a .csv or .parquet file."
    )


def load_uploaded_data(data: bytes, filename: str) -> pl.DataFrame:
    """Load bytes supplied by Panel's file input widget."""
    suffix = Path(filename).suffix.lower()
    source = BytesIO(data)
    if suffix == ".csv":
        return pl.read_csv(source)
    if suffix == ".parquet":
        return pl.read_parquet(source)
    raise ValueError("Unsupported upload. Choose a .csv or .parquet file.")


class DashboardEntry:
    """Load data and host a replaceable dashboard application."""

    def __init__(
        self,
        input_file: str | Path | None = None,
        *,
        track_id_col: str = "track_id",
        excluded_track_metrics: set[str] | None = None,
        label_col: str = "label",
        matched_value: object = "matched",
        feature_analysis_models: list[MLModelSpec] | None = None,
    ) -> None:
        self.track_id_col = track_id_col
        self.excluded_track_metrics = excluded_track_metrics or {"frame", "time"}
        self.label_col = label_col
        self.matched_value = matched_value
        self.feature_analysis_models = feature_analysis_models or []
        self.file_input = pn.widgets.FileInput(
            name="Input data",
            accept=".csv,.parquet",
            sizing_mode="stretch_width",
        )
        self.status = pn.pane.Alert("", alert_type="info", visible=False)
        self.dashboard_container = pn.Column(sizing_mode="stretch_both")
        self.file_input.param.watch(self._load_upload, "value")

        if input_file is None:
            self._set_data(make_example_data(), "Using generated example data.")
        else:
            path = Path(input_file)
            self._set_data(load_input_data(path), f"Loaded {path.name}.")

    def _set_data(self, data: pl.DataFrame, message: str) -> None:
        self.dashboard = DashboardApp(
            data,
            track_id_col=self.track_id_col,
            excluded_track_metrics=self.excluded_track_metrics,
            label_col=self.label_col,
            matched_value=self.matched_value,
            feature_analysis_models=self.feature_analysis_models,
        )
        self.dashboard_container[:] = [self.dashboard.view()]
        self.status.object = message
        self.status.alert_type = "success"
        self.status.visible = True

    def _load_upload(self, event) -> None:
        if not event.new:
            return
        try:
            data = load_uploaded_data(event.new, self.file_input.filename)
            self._set_data(data, f"Loaded {self.file_input.filename}.")
        except Exception as exc:
            self.status.object = f"Could not load input: {exc}"
            self.status.alert_type = "danger"
            self.status.visible = True

    def view(self) -> pn.Column:
        input_controls = pn.Card(
            self.file_input,
            self.status,
            title="Data source",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            input_controls,
            self.dashboard_container,
            sizing_mode="stretch_both",
            css_classes=["track-dashboard-root"],
        )

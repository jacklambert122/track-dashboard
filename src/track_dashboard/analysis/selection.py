from __future__ import annotations

from pathlib import Path

import panel as pn
import param

from ..core.state import DashboardState


class SelectionPanel(param.Parameterized):
    export_csv = param.Event()
    export_parquet = param.Event()
    apply_label = param.Event()
    label_column = param.String(default="review_label")
    label_value = param.String(default="")
    status_message = param.String(default="")
    status_type = param.Selector(
        default="info",
        objects=["info", "success", "warning", "danger"],
    )

    def __init__(
        self,
        state: DashboardState,
        *,
        export_directory: str | Path = "exports",
        **params,
    ) -> None:
        super().__init__(**params)
        self.state = state
        self.export_directory = Path(export_directory)
        self.param.watch(self._export_csv, "export_csv")
        self.param.watch(self._export_parquet, "export_parquet")
        self.param.watch(self._apply_label, "apply_label")

    @param.depends("state.selection_revision")
    def summary(self):
        data = self.state.selected_data_for_display()
        if data.is_empty():
            return pn.pane.Markdown("No data selected.")
        return pn.pane.Markdown(
            f"Selected **{data.height:,} rows** from "
            f"**{len(self.state.selected_track_ids):,} tracks**."
        )

    @param.depends("state.selection_revision")
    def table(self):
        data = self.state.selected_data_for_display()
        if data.is_empty():
            return pn.Spacer(height=120)
        return pn.widgets.Tabulator(
            data.to_pandas(),
            pagination="remote",
            page_size=20,
            height=350,
            sizing_mode="stretch_width",
        )

    def _export_csv(self, _event=None) -> None:
        data = self.state.selected_data_for_display()
        if data.is_empty():
            self._set_status("Select data before exporting.", "warning")
            return
        self.export_directory.mkdir(parents=True, exist_ok=True)
        path = (self.export_directory / "selected_data.csv").resolve()
        data.write_csv(path)
        self._set_status(f"Saved CSV to `{path}`.", "success")

    def _export_parquet(self, _event=None) -> None:
        data = self.state.selected_data_for_display()
        if data.is_empty():
            self._set_status("Select data before exporting.", "warning")
            return
        self.export_directory.mkdir(parents=True, exist_ok=True)
        path = (self.export_directory / "selected_data.parquet").resolve()
        data.write_parquet(path)
        self._set_status(f"Saved Parquet to `{path}`.", "success")

    def _apply_label(self, _event=None) -> None:
        try:
            self.state.label_selection(self.label_column, self.label_value)
        except ValueError as exc:
            self._set_status(str(exc), "danger")
            return
        self._set_status(
            f"Applied `{self.label_column.strip()}` = `{self.label_value}`. "
            "The label is now available in plot grouping and color controls.",
            "success",
        )

    def _set_status(self, message: str, status_type: str) -> None:
        self.status_message = message
        self.status_type = status_type

    @param.depends("status_message", "status_type")
    def status(self):
        return pn.pane.Alert(
            self.status_message,
            alert_type=self.status_type,
            visible=bool(self.status_message),
            sizing_mode="stretch_width",
        )

    def view(self) -> pn.Column:
        return pn.Column(
            self.summary,
            pn.Card(
                pn.widgets.TextInput.from_param(
                    self.param.label_column,
                    name="Label column",
                ),
                pn.widgets.TextInput.from_param(
                    self.param.label_value,
                    name="Label value",
                ),
                pn.widgets.Button.from_param(
                    self.param.apply_label,
                    name="Apply label to selection",
                    button_type="primary",
                ),
                title="Label selected data",
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.widgets.Button.from_param(
                    self.param.export_csv, name="Export CSV", button_type="primary"
                ),
                pn.widgets.Button.from_param(
                    self.param.export_parquet,
                    name="Export Parquet",
                    button_type="primary",
                ),
            ),
            self.status,
            self.table,
            sizing_mode="stretch_both",
        )

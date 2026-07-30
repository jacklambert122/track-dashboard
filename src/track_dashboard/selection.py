from __future__ import annotations

from pathlib import Path

import panel as pn
import param

from .state import DashboardState


class SelectionPanel(param.Parameterized):
    export_csv = param.Event()
    export_parquet = param.Event()

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

    @param.depends("state.selection_revision")
    def summary(self):
        data = self.state.selected_data
        if data.is_empty():
            return pn.pane.Markdown("No data selected.")
        return pn.pane.Markdown(
            f"Selected **{data.height:,} rows** from "
            f"**{len(self.state.selected_track_ids):,} tracks**."
        )

    @param.depends("state.selection_revision")
    def table(self):
        data = self.state.selected_data
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
        if self.state.selected_data.is_empty():
            return
        self.export_directory.mkdir(parents=True, exist_ok=True)
        self.state.selected_data.write_csv(self.export_directory / "selected_data.csv")

    def _export_parquet(self, _event=None) -> None:
        if self.state.selected_data.is_empty():
            return
        self.export_directory.mkdir(parents=True, exist_ok=True)
        self.state.selected_data.write_parquet(
            self.export_directory / "selected_data.parquet"
        )

    def view(self) -> pn.Column:
        return pn.Column(
            self.summary,
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
            self.table,
            sizing_mode="stretch_both",
        )

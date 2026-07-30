from __future__ import annotations

import holoviews as hv
import panel as pn
import param

from .base import AnalysisComponent


class ScatterPanel(AnalysisComponent):
    x = param.Selector(default=None, allow_None=True)
    y = param.Selector(default=None, allow_None=True)
    color = param.Selector(default=None, allow_None=True)

    def __init__(self, *args, **kwargs) -> None:
        self._selection_stream: hv.streams.Selection1D | None = None
        self._selection_watcher = None
        self._updating_options = False
        super().__init__(*args, **kwargs)
        self.refresh_options()

    def refresh_options(self) -> None:
        if self._updating_options:
            return
        self._updating_options = True
        try:
            numeric = self.data_model.numeric_features()
            grouping = self.data_model.grouping_features()
            old_x, old_y, old_color = self.x, self.y, self.color

            self.param.x.objects = numeric
            self.param.y.objects = numeric
            self.param.color.objects = [None, *grouping]

            self.x = old_x if old_x in numeric else (numeric[0] if numeric else None)
            self.y = (
                old_y
                if old_y in numeric
                else (numeric[min(1, len(numeric) - 1)] if numeric else None)
            )
            self.color = old_color if old_color in self.param.color.objects else None
        finally:
            self._updating_options = False

    @param.depends(
        "x",
        "y",
        "color",
        "state.analysis_level",
        "state.track_agg_methods",
        "state.data_revision",
    )
    def plot(self):
        df = self.data_model.analysis_df()
        if df.is_empty():
            return pn.pane.Markdown("No data matches the current filters.")
        if self.x not in df.columns or self.y not in df.columns:
            return pn.pane.Markdown("Select valid X and Y features.")

        kwargs = {
            "x": self.x,
            "y": self.y,
            "height": 475,
            "responsive": True,
            "tools": ["box_select", "lasso_select", "tap", "hover"],
            "active_tools": ["box_select"],
            "hover_cols": [self.state.track_id_col],
        }
        if self.color in df.columns:
            kwargs["color"] = self.color

        plot = df.hvplot.scatter(**kwargs)
        self.state.rendered_scatter_data = df
        self._bind_selection(plot)
        return plot

    def _bind_selection(self, plot) -> None:
        self._selection_stream = hv.streams.Selection1D(source=plot, index=[])
        self._selection_watcher = self._selection_stream.param.watch(
            self._selection_changed, "index"
        )

    def _selection_changed(self, event) -> None:
        rendered = self.state.rendered_scatter_data
        indices = [
            index
            for index in list(event.new or [])
            if 0 <= index < rendered.height
        ]
        if not indices:
            self.state.clear_selection()
            return
        self.state.set_selection(indices, rendered[indices])

    def view(self) -> pn.Column:
        controls = pn.Row(
            pn.widgets.Select.from_param(self.param.x, name="X feature"),
            pn.widgets.Select.from_param(self.param.y, name="Y feature"),
            pn.widgets.Select.from_param(self.param.color, name="Color"),
            sizing_mode="stretch_width",
        )
        return pn.Column(controls, self.plot, sizing_mode="stretch_both")

from __future__ import annotations

import panel as pn
import param

from .base import AnalysisComponent


class DistributionPanel(AnalysisComponent):
    feature = param.Selector(default=None, allow_None=True)
    group_by = param.Selector(default=None, allow_None=True)
    plot_type = param.Selector(default="Histogram", objects=["Histogram", "ECDF"])
    bins = param.Integer(default=40, bounds=(5, 200))

    def __init__(self, *args, **kwargs) -> None:
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
            old_feature, old_group = self.feature, self.group_by

            self.param.feature.objects = numeric
            self.param.group_by.objects = [None, *grouping]
            self.feature = (
                old_feature
                if old_feature in numeric
                else (numeric[0] if numeric else None)
            )
            self.group_by = (
                old_group if old_group in self.param.group_by.objects else None
            )
        finally:
            self._updating_options = False

    @param.depends(
        "feature",
        "group_by",
        "plot_type",
        "bins",
        "state.analysis_level",
        "state.track_agg_methods",
        "state.data_revision",
    )
    def plot(self):
        df = self.data_model.analysis_df()
        if df.is_empty():
            return pn.pane.Markdown("No data matches the current filters.")
        if self.feature not in df.columns:
            return pn.pane.Markdown("Select a valid feature.")

        kwargs = {
            "y": self.feature,
            "height": 400,
            "responsive": True,
        }
        if self.group_by in df.columns:
            kwargs["by"] = self.group_by

        if self.plot_type == "ECDF":
            return df.hvplot.ecdf(**kwargs)

        return df.hvplot.hist(bins=self.bins, **kwargs)

    def view(self) -> pn.Column:
        controls = pn.Row(
            pn.widgets.Select.from_param(self.param.feature, name="Feature"),
            pn.widgets.Select.from_param(self.param.group_by, name="Group by"),
            pn.widgets.Select.from_param(self.param.plot_type, name="Plot type"),
            pn.widgets.IntSlider.from_param(self.param.bins, name="Bins"),
            sizing_mode="stretch_width",
        )
        return pn.Column(controls, self.plot, sizing_mode="stretch_both")

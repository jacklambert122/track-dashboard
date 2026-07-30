from __future__ import annotations

import panel as pn
import param
import polars as pl

from .base import AnalysisComponent


class DistributionPanel(AnalysisComponent):
    feature = param.Selector(default=None, allow_None=True)
    group_by = param.Selector(default=None, allow_None=True)
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
        "bins",
        "state.analysis_level",
        "state.track_agg_features",
        "state.track_agg_methods_by_feature",
        "state.data_revision",
    )
    def matching_plots(self):
        return self._plots(self.data_model.analysis_df(), empty_message=None)

    @param.depends(
        "feature",
        "group_by",
        "bins",
        "state.selection_revision",
    )
    def selected_plots(self):
        return self._plots(
            self.state.selected_data,
            empty_message=(
                "Select points in the scatter plot to see their distributions."
            ),
        )

    def _plots(self, df, *, empty_message: str | None):
        if df.is_empty():
            return pn.pane.Markdown(
                empty_message or "No data matches the current filters."
            )
        if self.feature not in df.columns:
            return pn.pane.Markdown("Select a valid feature.")

        histogram_kwargs = {
            "y": self.feature,
            "height": 400,
            "responsive": True,
        }
        if self.group_by in df.columns:
            histogram_kwargs["by"] = self.group_by

        histogram = df.hvplot.hist(
            bins=self.bins, title="Histogram", **histogram_kwargs
        )

        ecdf_data = df.filter(pl.col(self.feature).is_not_null())
        if self.group_by in df.columns:
            ecdf_data = ecdf_data.sort([self.group_by, self.feature]).with_columns(
                (
                    pl.col(self.feature).rank(method="max").over(self.group_by)
                    / pl.len().over(self.group_by)
                ).alias("_ecdf")
            )
        else:
            ecdf_data = ecdf_data.sort(self.feature).with_columns(
                (
                    pl.col(self.feature).rank(method="max")
                    / pl.lit(ecdf_data.height)
                ).alias("_ecdf")
            )

        ecdf_kwargs = {
            "x": self.feature,
            "y": "_ecdf",
            "height": 400,
            "responsive": True,
            "title": "ECDF",
            "ylabel": "Cumulative probability",
        }
        if self.group_by in ecdf_data.columns:
            ecdf_kwargs["by"] = self.group_by
        ecdf = ecdf_data.hvplot.step(
            **ecdf_kwargs
        )
        return pn.Row(histogram, ecdf, sizing_mode="stretch_both")

    def view(self) -> pn.Column:
        controls = pn.Row(
            pn.widgets.Select.from_param(self.param.feature, name="Feature"),
            pn.widgets.Select.from_param(self.param.group_by, name="Group by"),
            pn.widgets.IntSlider.from_param(self.param.bins, name="Bins"),
            sizing_mode="stretch_width",
        )
        plots = pn.Tabs(
            ("Matching data", self.matching_plots),
            ("Selected data", self.selected_plots),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        return pn.Column(controls, plots, sizing_mode="stretch_both")

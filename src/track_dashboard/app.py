from __future__ import annotations

import panel as pn
import param
import polars as pl

from .aggregations import AGGREGATIONS
from .data_model import DataModel
from .distributions import DistributionPanel
from .filters import FilterPanel
from .scatter import ScatterPanel
from .selection import SelectionPanel
from .state import DashboardState


class DashboardApp(param.Parameterized):
    """Composition root: creates components and arranges the application."""

    sidebar_width = param.Integer(default=560, bounds=(400, 900))

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        track_id_col: str = "track_id",
        excluded_track_metrics: set[str] | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self.state = DashboardState(df, track_id_col=track_id_col)
        self.data_model = DataModel(
            self.state,
            excluded_track_metrics=excluded_track_metrics,
        )
        aggregatable = self.data_model.track_aggregation_features()
        self.state.param.track_agg_features.objects = aggregatable
        self.state.track_agg_features = []
        self.state.track_agg_methods_by_feature = {}
        self.state.param.watch(
            self._sync_track_aggregation_methods, "track_agg_features"
        )
        self.filters = FilterPanel(self.state)
        self.scatter = ScatterPanel(self.state, self.data_model)
        self.distributions = DistributionPanel(self.state, self.data_model)
        self.selection = SelectionPanel(self.state)

    @param.depends("state.analysis_level", "state.track_agg_features")
    def aggregation_controls(self):
        if self.state.analysis_level != "Track":
            return pn.Spacer(height=0)

        method_controls = []
        for feature in self.state.track_agg_features:
            widget = pn.widgets.MultiChoice(
                name=feature,
                options=list(AGGREGATIONS),
                value=list(self.state.track_agg_methods_by_feature.get(feature, [])),
            )
            widget.param.watch(
                lambda event, selected_feature=feature: self._set_track_methods(
                    selected_feature, event.new
                ),
                "value",
            )
            method_controls.append(widget)

        return pn.Card(
            pn.widgets.MultiChoice.from_param(
                self.state.param.track_agg_features,
                name="Features to aggregate",
            ),
            pn.Column(
                *method_controls,
                sizing_mode="stretch_width",
            ),
            title="Track metric configuration",
            sizing_mode="stretch_width",
        )

    def _sync_track_aggregation_methods(self, event) -> None:
        methods = dict(self.state.track_agg_methods_by_feature)
        for feature in event.new:
            methods.setdefault(feature, [])
        self.state.track_agg_methods_by_feature = methods

    def _set_track_methods(self, feature: str, methods: list[str]) -> None:
        methods_by_feature = dict(self.state.track_agg_methods_by_feature)
        methods_by_feature[feature] = list(methods)
        self.state.track_agg_methods_by_feature = methods_by_feature

    def view(self):
        sidebar = pn.Column(
            pn.widgets.RadioButtonGroup.from_param(
                self.state.param.analysis_level,
                name="Analysis level",
                button_type="primary",
            ),
            self.aggregation_controls,
            pn.Card(self.filters.view(), title="Filters", collapsed=False),
            width=self.sidebar_width,
            min_width=400,
        )

        tabs = pn.Tabs(
            ("Scatter", self.scatter.view()),
            ("Distributions", self.distributions.view()),
            ("Selected data", self.selection.view()),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        return pn.Row(sidebar, tabs, sizing_mode="stretch_both")

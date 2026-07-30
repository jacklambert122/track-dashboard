from __future__ import annotations

import panel as pn
import param
import polars as pl

from .aggregations import numeric_columns
from .data_model import DataModel
from .distributions import DistributionPanel
from .filters import FilterPanel
from .scatter import ScatterPanel
from .selection import SelectionPanel
from .state import DashboardState


class DashboardApp(param.Parameterized):
    """Composition root: creates components and arranges the application."""

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
        filterable = numeric_columns(df, exclude={track_id_col})
        self.filters = FilterPanel(self.state, numeric_features=filterable)
        self.scatter = ScatterPanel(self.state, self.data_model)
        self.distributions = DistributionPanel(self.state, self.data_model)
        self.selection = SelectionPanel(self.state)

    @param.depends("state.analysis_level")
    def aggregation_controls(self):
        if self.state.analysis_level != "Track":
            return pn.Spacer(height=0)
        return pn.Card(
            pn.widgets.MultiChoice.from_param(
                self.state.param.track_agg_methods,
                name="Track aggregations",
            ),
            title="Track metric configuration",
            sizing_mode="stretch_width",
        )

    def view(self):
        sidebar = pn.Column(
            pn.widgets.RadioButtonGroup.from_param(
                self.state.param.analysis_level,
                name="Analysis level",
                button_type="primary",
            ),
            self.aggregation_controls,
            pn.Card(self.filters.view(), title="Filters", collapsed=False),
            width=380,
        )

        tabs = pn.Tabs(
            ("Scatter", self.scatter.view()),
            ("Distributions", self.distributions.view()),
            ("Selected data", self.selection.view()),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        return pn.Row(sidebar, tabs, sizing_mode="stretch_both")

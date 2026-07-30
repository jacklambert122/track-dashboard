from __future__ import annotations

import panel as pn
import param
import polars as pl

from .state import DashboardState


class NumericFilterRow(param.Parameterized):
    feature = param.Selector(default=None, allow_None=True)
    lower = param.Number(default=None, allow_None=True)
    upper = param.Number(default=None, allow_None=True)
    remove = param.Event()

    def __init__(self, *, features: list[str], **params) -> None:
        super().__init__(**params)
        self.param.feature.objects = features
        self.feature = features[0] if features else None

    def expression(self) -> pl.Expr | None:
        if self.feature is None:
            return None

        expression = pl.lit(True)
        if self.lower is not None:
            expression &= pl.col(self.feature) >= self.lower
        if self.upper is not None:
            expression &= pl.col(self.feature) <= self.upper
        return expression

    def view(self) -> pn.Row:
        return pn.Row(
            pn.widgets.Select.from_param(self.param.feature, name="Feature"),
            pn.widgets.FloatInput.from_param(self.param.lower, name="Minimum"),
            pn.widgets.FloatInput.from_param(self.param.upper, name="Maximum"),
            pn.widgets.Button.from_param(
                self.param.remove, name="Remove", button_type="danger"
            ),
            sizing_mode="stretch_width",
        )


class FilterPanel(param.Parameterized):
    add_filter = param.Event()

    def __init__(
        self,
        state: DashboardState,
        *,
        numeric_features: list[str],
        **params,
    ) -> None:
        super().__init__(**params)
        self.state = state
        self.numeric_features = numeric_features
        self.rows: list[NumericFilterRow] = []
        self.rows_container = pn.Column(sizing_mode="stretch_width")
        self.param.watch(self._add_filter, "add_filter")

    def _add_filter(self, _event=None) -> None:
        row = NumericFilterRow(features=self.numeric_features)
        row.param.watch(lambda _event, row=row: self._remove_filter(row), "remove")
        row.param.watch(self._update_state, ["feature", "lower", "upper"])
        self.rows.append(row)
        self.rows_container.append(row.view())
        self._update_state()

    def _remove_filter(self, row: NumericFilterRow) -> None:
        if row not in self.rows:
            return
        index = self.rows.index(row)
        self.rows.pop(index)
        self.rows_container.pop(index)
        self._update_state()

    def _update_state(self, _event=None) -> None:
        self.state.filter_expressions = [
            expression
            for row in self.rows
            if (expression := row.expression()) is not None
        ]
        self.state.mark_data_changed()

    def view(self) -> pn.Column:
        return pn.Column(
            pn.widgets.Button.from_param(
                self.param.add_filter, name="Add filter", button_type="primary"
            ),
            self.rows_container,
            sizing_mode="stretch_width",
        )

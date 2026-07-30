from __future__ import annotations

import panel as pn
import param
import polars as pl

from .aggregations import is_categorical
from .state import INTERNAL_ROW_ID, DashboardState


def filter_feature_types(state: DashboardState) -> dict[str, str]:
    """Return filterable columns mapped to continuous or discrete controls."""
    excluded = {state.track_id_col, INTERNAL_ROW_ID}
    return {
        column: "continuous" if dtype.is_numeric() else "discrete"
        for column, dtype in state.point_df.schema.items()
        if column not in excluded
        and (dtype.is_numeric() or is_categorical(dtype))
    }


class FilterRow(param.Parameterized):
    feature = param.Selector(default=None, allow_None=True)
    lower = param.Number(default=None, allow_None=True)
    upper = param.Number(default=None, allow_None=True)
    values = param.ListSelector(default=[], objects=[])
    remove = param.Event()

    def __init__(
        self,
        *,
        state: DashboardState,
        feature_types: dict[str, str],
        **params,
    ) -> None:
        super().__init__(**params)
        self.state = state
        self.feature_types = feature_types
        self.param.watch(self._feature_changed, "feature")
        self.update_features(feature_types)

    def update_features(self, feature_types: dict[str, str]) -> None:
        old_feature = self.feature
        self.feature_types = feature_types
        features = list(feature_types)
        self.param.feature.objects = features
        self.feature = (
            old_feature
            if old_feature in features
            else (features[0] if features else None)
        )
        self._refresh_discrete_values()

    def _feature_changed(self, _event=None) -> None:
        self.lower = None
        self.upper = None
        self.values = []
        self._refresh_discrete_values()

    def _refresh_discrete_values(self) -> None:
        if self.feature is None or self.feature_types.get(self.feature) != "discrete":
            self.param["values"].objects = []
            self.values = []
            return
        values = (
            self.state.point_df.get_column(self.feature)
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        self.param["values"].objects = values
        self.values = [value for value in self.values if value in values]

    def expression(self) -> pl.Expr | None:
        if self.feature is None:
            return None

        if self.feature_types[self.feature] == "discrete":
            return pl.col(self.feature).is_in(self.values) if self.values else None

        expression = pl.lit(True)
        if self.lower is not None:
            expression &= pl.col(self.feature) >= self.lower
        if self.upper is not None:
            expression &= pl.col(self.feature) <= self.upper
        return expression if self.lower is not None or self.upper is not None else None

    def _range_changed(self, event) -> None:
        self.lower, self.upper = event.new

    def _continuous_range_control(self):
        values = self.state.point_df.get_column(self.feature).drop_nulls()
        if values.is_empty():
            start, end = 0.0, 1.0
        else:
            start = float(values.min())
            end = float(values.max())
            if start == end:
                padding = max(abs(start) * 0.05, 1.0)
                start -= padding
                end += padding

        dtype = self.state.point_df.schema[self.feature]
        step = 1.0 if dtype.is_integer() else max((end - start) / 1000, 1e-12)
        selected_range = (
            self.lower if self.lower is not None else start,
            self.upper if self.upper is not None else end,
        )
        slider = pn.widgets.EditableRangeSlider(
            label="Allowed range",
            start=start,
            end=end,
            value=selected_range,
            step=step,
            format="0,0.[000000]",
            sizing_mode="stretch_width",
        )
        slider.param.watch(self._range_changed, "value")
        return slider

    @param.depends("feature")
    def value_controls(self):
        if self.feature is None:
            return pn.Spacer()
        if self.feature_types[self.feature] == "discrete":
            return pn.widgets.MultiChoice.from_param(
                self.param["values"],
                name="Values",
                sizing_mode="stretch_width",
            )
        return self._continuous_range_control()

    def view(self) -> pn.Column:
        return pn.Column(
            pn.Row(
                pn.widgets.Select.from_param(
                    self.param.feature,
                    name="Feature",
                    sizing_mode="stretch_width",
                ),
                pn.widgets.Button.from_param(
                    self.param.remove,
                    name="Remove",
                    button_type="danger",
                    width=80,
                ),
                sizing_mode="stretch_width",
            ),
            self.value_controls,
            sizing_mode="stretch_width",
        )


class FilterPanel(param.Parameterized):
    add_filter = param.Event()

    def __init__(self, state: DashboardState, **params) -> None:
        super().__init__(**params)
        self.state = state
        self.feature_types = filter_feature_types(state)
        self.rows: list[FilterRow] = []
        self.rows_container = pn.Column(sizing_mode="stretch_width")
        self.param.watch(self._add_filter, "add_filter")
        self.state.param.watch(self._refresh_features, "data_revision")

    def _add_filter(self, _event=None) -> None:
        row = FilterRow(state=self.state, feature_types=self.feature_types)
        row.param.watch(lambda _event, row=row: self._remove_filter(row), "remove")
        row.param.watch(
            self._update_state,
            ["feature", "lower", "upper", "values"],
        )
        self.rows.append(row)
        self.rows_container.append(row.view())
        self._update_state()

    def _remove_filter(self, row: FilterRow) -> None:
        if row not in self.rows:
            return
        index = self.rows.index(row)
        self.rows.pop(index)
        self.rows_container.pop(index)
        self._update_state()

    def _refresh_features(self, _event=None) -> None:
        feature_types = filter_feature_types(self.state)
        if feature_types == self.feature_types:
            return
        self.feature_types = feature_types
        for row in self.rows:
            row.update_features(feature_types)

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

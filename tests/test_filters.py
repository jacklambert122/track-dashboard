import polars as pl

from track_dashboard.core.filters import FilterPanel, FilterRow, filter_feature_types
from track_dashboard.core.state import DashboardState


def make_state() -> DashboardState:
    return DashboardState(
        pl.DataFrame(
            {
                "track_id": [1, 2, 3],
                "score": [1.0, 2.0, 3.0],
                "label": ["keep", "reject", "keep"],
            }
        )
    )


def test_filter_feature_types_classifies_numeric_and_categorical_columns():
    state = make_state()

    assert filter_feature_types(state) == {
        "score": "continuous",
        "label": "discrete",
    }


def test_continuous_filter_uses_range_expression():
    state = make_state()
    row = FilterRow(state=state, feature_types=filter_feature_types(state))
    row.feature = "score"
    row.lower = 2.0

    result = state.point_df.filter(row.expression())

    assert result.get_column("track_id").to_list() == [2, 3]


def test_continuous_filter_control_uses_data_bounds():
    state = make_state()
    row = FilterRow(state=state, feature_types=filter_feature_types(state))
    row.feature = "score"

    control = row.value_controls()

    assert control.start == 1.0
    assert control.end == 3.0
    assert control.value == (1.0, 3.0)
    assert control.editable


def test_discrete_filter_uses_selected_values():
    state = make_state()
    row = FilterRow(state=state, feature_types=filter_feature_types(state))
    row.feature = "label"
    row.values = ["keep"]

    result = state.point_df.filter(row.expression())

    assert result.get_column("track_id").to_list() == [1, 3]


def test_filter_panel_discovers_new_label_columns():
    state = make_state()
    panel = FilterPanel(state)
    state.set_selection([0], state.point_df.head(1))

    state.label_selection("review", "approved")

    assert panel.feature_types["review"] == "discrete"

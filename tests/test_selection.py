import polars as pl

from track_dashboard.analysis.selection import SelectionPanel
from track_dashboard.core.state import INTERNAL_ROW_ID, DashboardState


def test_export_reports_path_and_hides_internal_row_id(tmp_path):
    state = DashboardState(
        pl.DataFrame({"track_id": [1, 2], "value": [3.0, 4.0]})
    )
    state.set_selection([0], state.point_df.head(1))
    panel = SelectionPanel(state, export_directory=tmp_path)

    panel._export_csv()

    path = (tmp_path / "selected_data.csv").resolve()
    exported = pl.read_csv(path)
    assert INTERNAL_ROW_ID not in exported.columns
    assert str(path) in panel.status_message
    assert panel.status_type == "success"

from __future__ import annotations

import panel as pn

from .app import DashboardApp
from .example_data import make_example_data


def main() -> None:
    pn.extension("tabulator", sizing_mode="stretch_width")
    dashboard = DashboardApp(
        make_example_data(),
        track_id_col="track_id",
        excluded_track_metrics={"frame", "time"},
    )
    pn.serve(
        dashboard.view(),
        title="Track Dashboard",
        show=True,
    )

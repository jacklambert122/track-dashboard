import panel as pn

from track_dashboard import DashboardApp
from track_dashboard.example_data import make_example_data

pn.extension("tabulator", sizing_mode="stretch_width")

app = DashboardApp(
    make_example_data(),
    track_id_col="track_id",
    excluded_track_metrics={"frame", "time"},
)

app.view().servable(title="Track Dashboard")

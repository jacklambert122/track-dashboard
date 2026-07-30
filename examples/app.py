import panel as pn

from track_dashboard import DashboardEntry

pn.extension("tabulator", sizing_mode="stretch_width")

app = DashboardEntry()

app.view().servable(title="Track Dashboard")

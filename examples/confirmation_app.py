from pathlib import Path

import panel as pn

from track_dashboard import TrackConfirmationDashboard
from track_dashboard.example_data import make_example_data

pn.extension(
    "tabulator",
    sizing_mode="stretch_width",
    design="material",
    theme="dark",
)

config_path = Path(__file__).with_name("track_qa_config.json")
app = TrackConfirmationDashboard(
    make_example_data(),
    config_path,
)

app.view().servable(title="Track Confirmation Dashboard")

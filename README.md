# Track Dashboard

A modular interactive dashboard for point-level and track-level analysis using:

- Panel and Param for the application and controls
- hvPlot/HoloViews for interactive plots and linked selection
- Polars for filtering and track aggregation

The main application only composes components and shares state. Filtering, data
preparation, scatter selection, distributions, and selected-data export are kept
in separate modules.

## Architecture

```text
DashboardApp
├── DashboardState        shared reactive state
├── DataModel             filtering and point-to-track transformation
├── FilterPanel           dynamic numeric filters
├── ScatterPanel          scatter plot and selection
├── DistributionPanel     histogram and ECDF plots
└── SelectionPanel        selected rows and export
```

## Install with uv

```bash
uv sync --extra dev
```

## Run

```bash
uv run panel serve examples/app.py --show --autoreload
```

or use the installed command:

```bash
uv run track-dashboard
```

## Test

```bash
uv run pytest
uv run ruff check .
```

## Track-level metrics

When `Track` analysis is selected, aggregation controls appear. Choosing methods
such as `mean`, `max`, and `std` creates columns such as:

```text
snr_mean
snr_max
snr_std
residual_mean
residual_max
residual_std
```

The scatter and distribution components continue to operate on ordinary feature
names; only the active dataframe and selector options change.

## Customize for your data

Replace `make_example_data()` in `src/track_dashboard/example_data.py`, or create
`DashboardApp(your_polars_dataframe, track_id_col="track_id")` in your own Panel
entry point.

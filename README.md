# Track Dashboard

A modular interactive dashboard for point-level and track-level analysis using:

- Panel and Param for the application and controls
- hvPlot/HoloViews for interactive plots and linked selection
- Polars for filtering and track aggregation

The main application only composes components and shares state. Filtering, data
preparation, scatter selection, distributions, and selected-data export are kept
in separate modules.

Filters adapt to the selected feature type: numeric features use minimum and
maximum inputs, while string, boolean, categorical, and enum features use a
dropdown for selecting one or more values.

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

Both entry points include a file picker for replacing the current data with a
CSV or Parquet file. A file can also be loaded when starting the CLI:

```bash
uv run track-dashboard tracks.parquet
uv run track-dashboard tracks.csv --track-id-col track_id
```

When no file is supplied, the dashboard starts with generated example data.

The installed command uses port `5006` by default. When rerun, it stops the
existing listener on that default port before starting the new server. To use a
different port:

```bash
uv run track-dashboard --port 5007
```

For safety, a listener on a custom port is not stopped automatically; the
command exits with a clear error instead.

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

The track controls also let you choose which numeric source features are
aggregated and configure aggregation methods independently for each feature.
This keeps the generated track-level dataframe focused on the metrics needed
for the current analysis.

The scatter and distribution components continue to operate on ordinary feature
names; only the active dataframe and selector options change.

## Distributions

The distributions tab shows a histogram and ECDF together for the selected
feature. Use its `Selected data` sub-tab to inspect the points or tracks selected
in the scatter plot with the same distribution controls.

## Label and export selected data

The selected-data tab can add or update a string label column for the current
selection. Point-mode labels apply to the exact selected rows; track-mode labels
apply to every source row in the selected tracks. New label columns immediately
become available in scatter color and distribution grouping controls.

CSV and Parquet exports include the applied labels. After an export, the
dashboard displays the complete path of the saved file.

## Customize for your data

Replace `make_example_data()` in `src/track_dashboard/example_data.py`, or create
`DashboardApp(your_polars_dataframe, track_id_col="track_id")` in your own Panel
entry point.

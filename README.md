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

## Contents

- [Installation](#installation)
- [Run the analysis dashboard](#run-the-analysis-dashboard)
- [Analysis features](#analysis-features)
- [Track confirmation dashboard](#track-confirmation-dashboard)
- [Development](#development)

<details>
<summary><strong>Package organization and architecture</strong></summary>

## Package organization

Implementation modules are grouped by responsibility:

```text
src/track_dashboard/
├── analysis/       # Main dashboard, plots, and feature analysis
├── confirmation/   # Confirmation engine, dashboard, CLI, and examples
├── core/           # State, filtering, aggregation, and shared styling
├── models/         # External model adapters such as ONNX
├── cli.py          # Shared main-dashboard server command
├── entry.py        # File loading and replaceable dashboard entry
└── example_data.py # Generated sample dataframe
```

The subpackages are the canonical module API; the former flat compatibility
modules have been removed. Import implementation components from
`track_dashboard.core`, `track_dashboard.analysis`,
`track_dashboard.confirmation`, or `track_dashboard.models`. Common public
classes such as `DashboardApp`, `TrackConfirmationDashboard`, and
`MLModelSpec` remain available directly from `track_dashboard`.

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

</details>

## Installation

The project targets Python 3.11 and pins the SHAP/Numba/llvmlite stack to
versions with Python 3.11 wheels.

```bash
uv sync --extra dev
```

## Run the analysis dashboard

```bash
uv run panel serve examples/app.py --show --autoreload
```

or use the installed command:

```bash
uv run track-dashboard
```

Both entry points include a file picker for replacing the current data with a
CSV or Parquet file. A file can also be loaded when starting the CLI:

Both the analysis and track-confirmation dashboards use a shared dark Material
theme with coordinated cards, controls, tables, tabs, and plot styling.

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

## Analysis features

### Track-level metrics

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

### Distributions

The distributions tab shows a histogram and ECDF together for the selected
feature. Use its `Selected data` sub-tab to inspect the points or tracks selected
in the scatter plot with the same distribution controls.

### Feature analysis

The main analysis dashboard's **Feature analysis** tab evaluates selected
numeric features against a selectable binary label column and positive class.
It uses the active point or track analysis dataframe, respects dashboard
filters and track aggregations, and runs only when **Run feature analysis** is
clicked.

Choose any subset of available features and one or more methods:

- **Point-biserial correlation** reports signed linear association with the
  matched label.
- **Mutual information** estimates non-linear dependence using quantile bins.
- **Full correlation matrix** displays the annotated Pearson correlation matrix
  for the selected features and binary positive-label indicator.
- **Univariate ROC AUC** measures each feature's standalone class separation.
- **Incremental CV ROC AUC** fits cross-validated multivariate logistic models
  and reports the AUC lost when each feature is removed, showing what that
  feature adds beyond the other selected features.
- **Permutation importance** measures a registered model's ROC AUC decrease
  after shuffling each included model feature.
- **SHAP** reports mean absolute SHAP values for included features used by a
  registered model.

<details>
<summary><strong>Model choices, SHAP explainers, and provided-model setup</strong></summary>

For SHAP and permutation importance, choose **Build default model** and select
logistic regression, random forest, or gradient boosting. The dashboard trains
the chosen classifier from the selected features and current label selection.
Choose no class weighting, automatic balanced weighting based on inverse class
frequency, or custom positive/other-class weights. Random forests also expose
tree count, maximum depth, minimum leaf size, features per split, split
criterion, and bootstrap sampling.

Each built-in model exposes relevant hyperparameters: logistic regression
provides regularization strength, penalty, iteration limit, and intercept;
random forest provides its tree and sampling controls; gradient boosting
provides boosting stages, learning rate, tree depth, leaf size, and row
subsampling.

When **SHAP** is selected, choose an explainer:

- **Auto** uses Tree for fitted tree models, Linear for fitted linear models,
  and Permutation as the general fallback.
- **Tree** is optimized for compatible tree estimators.
- **Linear** is optimized for compatible linear estimators.
- **Permutation** works with arbitrary provided models, including ONNX, but is
  usually much slower.

SHAP results include a horizontal mean-absolute-importance bar chart with
features on the y-axis, plus a beeswarm plot. Each point in the beeswarm
represents one analyzed row: horizontal
position is its SHAP impact, vertical position is the feature, and color
represents the feature's low-to-high value within the analyzed sample. Points
with nearby SHAP values stack symmetrically in the y direction, so thicker
sections of each feature row indicate greater local density along the x axis.
Use the **Explained track** selector to inspect a force-style plot for a track.
In Track analysis mode, the explanation corresponds to that track's aggregated
model-input row. In Point mode, every analyzed measurement from the selected
track is displayed as an individual force row. Its y-axis entry includes the
measurement time and selected label, and its prediction marker is color-coded
by that label. SHAP sampling keeps complete tracks together, so it never shows
only part of a sampled track. Each point gets a separate, independently scaled
force panel with direct feature/contribution labels and hover details. Red
features push the output higher, blue features push it lower, and the panel
header reports the point time, label, baseline, and final prediction.

Alternatively, choose **Use provided model** after registering one when creating
`DashboardApp`:

```python
dashboard = DashboardApp(
    data,
    feature_analysis_models=[
        MLModelSpec(
            name="quality_model",
            model=quality_model,
            features=("snr", "residual"),
        )
    ],
)
```

Saved ONNX models can also be registered from the main dashboard command line:

```bash
uv run track-dashboard tracks.parquet \
  --label-col label \
  --positive-class matched \
  --onnx-model quality_model models/quality.onnx snr residual
```

The registration format is `NAME FILE FEATURE [FEATURE ...]`; features must be
listed in the model input order. Repeat `--onnx-model` to register multiple
models. Their names appear in **Use provided model → Provided model**:

```bash
uv run track-dashboard tracks.parquet \
  --onnx-model quality_model models/quality.onnx snr residual \
  --onnx-model limb_model models/limb.onnx earth_limb_score snr
```

The sample-size control limits expensive analysis on large datasets. Model
analysis scores all declared model inputs but reports only selected features.
When **Full correlation matrix** is selected, the tab shows an annotated Pearson
correlation heatmap for the selected features and a binary indicator
representing the selected positive label class. Mutual-information scores
always measure each feature directly against that same selected label and do
not automatically display the correlation matrix.

</details>

### Label and export selected data

The selected-data tab can add or update a string label column for the current
selection. Point-mode labels apply to the exact selected rows; track-mode labels
apply to every source row in the selected tracks. New label columns immediately
become available in scatter color and distribution grouping controls.

CSV and Parquet exports include the applied labels. After an export, the
dashboard displays the complete path of the saved file.

### Customize for your data

Replace `make_example_data()` in `src/track_dashboard/example_data.py`, or create
`DashboardApp(your_polars_dataframe, track_id_col="track_id")` in your own Panel
entry point.

## Track confirmation dashboard

The repository also includes a separate dashboard for comparing the current
track-confirmation logic with experimental range-based confirmation paths:

```bash
uv run panel serve examples/confirmation_app.py --show --autoreload
```

or, after installing the project:

```bash
uv run track-confirmation-dashboard \
  --data tracks.parquet \
  --config confirmation.json
```

The config loader reads the current rules from:

```python
payload["dynamic_specific"]["track_qa_config"]
```

The input dataframe must contain track ID, time, and matched/unmatched label
columns. Their defaults are `track_id`, `time`, and `label`; CLI flags can
change all three.

<details>
<summary><strong>Existing Python confirmation logic and rule behavior</strong></summary>

### Existing Python confirmation logic

`TrackConfirmationDashboard` accepts an evaluator with this contract:

```python
def run_current_confirmation(track_qa_config, data):
    return data.with_columns(
        current_quality_path(track_qa_config, data).alias("quality_path"),
        current_limb_path(track_qa_config, data).alias("limb_path"),
    )


dashboard = TrackConfirmationDashboard(
    data,
    full_json_payload,
    evaluator=run_current_confirmation,
    default_path_columns=["quality_path", "limb_path"],
)
```

The evaluator must preserve row order and row count. Before it is called, the
dashboard sorts measurements by track ID and time so stateful confirmation
functions see each track chronologically. Evaluated path columns are mapped back
to the input's original row order afterward. Each confirmation-path column
contains `1` at times where that path confirms the track. When path columns are
newly added by the evaluator, they are detected automatically;
`default_path_columns` is useful when those columns already exist in the input.

The first confirmation time is the minimum time at which any default or
candidate path contains `1`. Confirmation is latched at the track level: every
measurement for that track at or after its first confirmation time is treated
as confirmed, even when a later point no longer satisfies the triggering rule.
Ranges within one experimental path are combined with AND, while separate paths
are combined with OR.

The control panel has independent multi-selects for the loaded paths enabled in
the Default baseline and in the Candidate. Path definitions still come from the
configuration; the selectors only control which loaded paths participate in
each side of the comparison. Set `default_enabled_paths` or `enabled_paths` in
`track_qa_config` to initialize the Default or Candidate selection respectively.
When these fields are omitted, Default enables every loaded path and Candidate
starts with the same selection. Disabling a Candidate path can identify tracks
lost relative to Default, while disabling a Default path supports alternate
baseline comparisons.
Add range rows under the same experimental path name to build an AND path, or
enter additional path names to evaluate multiple new OR paths simultaneously.
Using the name of a loaded range path creates a Candidate override: only the
specified feature range is replaced, the path's other configured ranges are
inherited, and the Overview reports the boundary change (for example,
`min: 20 → 25`) instead of treating the path as newly added.
Downloaded candidate configs preserve both the selected defaults in
`default_enabled_paths`, the Candidate selection in `enabled_paths`, and all
proposed `experimental_paths`.

The same type-aware filters as the analysis dashboard are available in the
confirmation sidebar. Numeric features use editable ranges, while string,
boolean, categorical, and enum features use value dropdowns. Filters are
applied before default and experimental confirmation, and every timing plot,
summary table, and false-alarm metric recomputes from the filtered population.

</details>

<details>
<summary><strong>Register ML confirmation paths from Python</strong></summary>

### ML confirmation paths

Models are registered from trusted Python code rather than uploaded as pickle
files. A registered model declares its input features and may expose
`predict_proba`, `predict`, or a callable accepting a Polars feature dataframe:

```python
from track_dashboard import MLModelSpec, TrackConfirmationDashboard


def quality_score(features):
    return (
        features.get_column("snr") / 30
        * (1 - features.get_column("residual") / 8).clip(0, 1)
    ).clip(0, 1)


dashboard = TrackConfirmationDashboard(
    data,
    config,
    ml_models=[
        MLModelSpec(
            name="quality_model",
            model=quality_score,
            features=("snr", "residual"),
        )
    ],
)
```

For estimators with `predict_proba`, the positive-class column defaults to index
`1` and can be changed with `positive_class_index`. The dashboard lets users
select a registered model, name the confirmation path, choose a probability
threshold, and enable multiple ML paths simultaneously. An ML score crossing
the threshold triggers confirmation; confirmation then remains latched for all
later measurements in that track.

Candidate JSON records the model name, features, class index, threshold, and
path name under `ml_confirmation_paths`. Model binaries are intentionally not
serialized; reload the config with a trusted model registered under the same
name.

</details>

<details>
<summary><strong>Add ONNX models from the command line</strong></summary>

### Add ONNX models from the command line

Install the project dependencies, including ONNX Runtime:

```bash
uv sync
```

Register an ONNX model by supplying its dashboard name, model file, and feature
columns in the exact order expected by the model's input tensor:

```bash
uv run track-confirmation-dashboard \
  --data tracks.parquet \
  --config confirmation.json \
  --onnx-model quality_model models/quality.onnx snr residual earth_limb_score
```

Repeat `--onnx-model` to register several saved models:

```bash
uv run track-confirmation-dashboard \
  --data tracks.parquet \
  --config confirmation.json \
  --onnx-model quality_model models/quality.onnx snr residual \
  --onnx-model limb_model models/limb.onnx earth_limb_score snr
```

Each registration has the form:

```text
--onnx-model NAME FILE FEATURE [FEATURE ...]
```

The adapter uses the model's first input tensor. It selects an output containing
`probab` or `score` in its name, falling back to the last model output. One-column
or one-dimensional probability outputs are converted to two-class
probabilities. After startup, use the **ML confirmation paths** card to select a
registered model, set its probability threshold, and add one or more ML-backed
confirmation paths.

For models requiring explicit input or output tensor names, register
`ONNXProbabilityModel(path, input_name=..., output_name=...)` from Python rather
than through the shorthand CLI option.

</details>

### Comparison metrics

The dashboard reports:

- default, candidate, and newly confirmed track counts;
- newly confirmed measurements split by matched/unmatched label;
- candidate-confirmed matched measurements;
- first-confirmation time for every track and its change from default;
- overlaid histograms and empirical CDFs of default and candidate first
  confirmation times;
- a confirmation-path table with triggering measurement and unique-track counts
  split into matched and false cohorts for default and experimental paths;
- point false-alarm rate: unmatched candidate-confirmed measurements divided by
  all candidate-confirmed measurements;
- track false-alarm rate: unmatched candidate-confirmed tracks divided by all
  candidate-confirmed tracks.

The **Changed tracks** tab classifies differences as **Added**, **Lost**,
**Earlier**, or **Later** based on confirmation status and first-confirmation
time. Select any numeric dataframe columns independently for the scatter plot's
X and Y axes, and optionally include more numeric columns in the measurement
table.
Plots and tables carry a `confirmation_logic` value of `Default` or `Candidate`,
and a selectable label column—such as matched/unmatched—is included in hover
details and displayed table rows.

<details>
<summary><strong>Example default ONNX confirmation model and config</strong></summary>

### Example provided default ML confirmation model

Here, "default model" means a model already used by the current confirmation
logic—not a model trained by the dashboard. The recommended layout keeps model
artifacts beside the confirmation config:

```text
examples/
├── track_qa_config.json
└── models/
    ├── build_example_model.py
    └── example_quality_model.onnx
```

The example config contains the complete model registry and baseline mapping:

```json
{
  "dynamic_specific": {
    "track_qa_config": {
      "models": {
        "example_quality_model": {
          "type": "onnx",
          "file": "models/example_quality_model.onnx",
          "features": ["snr", "residual"],
          "input_name": "features",
          "output_name": "probabilities",
          "positive_class_index": 1
        }
      },
      "default_ml_paths": [
        {
          "name": "default_ml_quality_path",
          "model": "example_quality_model",
          "threshold": 0.55
        }
      ]
    }
  }
}
```

Model `file` values are resolved relative to the config file, not the current
working directory. Each model requires:

- a unique registry name;
- `type`, currently `onnx` or `linear_json`;
- `file`, pointing to the saved model artifact;
- `features`, in exact model-input order;
- optional `positive_class_index`, defaulting to `1`.

ONNX entries may also set `input_name` and `output_name`. Use `--model-dir` to
override the config-relative model directory when deploying the same config
with artifacts stored elsewhere.

Each default ML path requires `name`, a registered `model`, and `threshold`.
Candidate paths use the same model registry under `ml_confirmation_paths`.

The bundled ONNX model is a small logistic graph:

```text
features → MatMul([0.18, -0.9]) → Add(-2.5) → Sigmoid → probabilities
```

The binary artifact is checked into the repository. Rebuild and validate it
from its source script with:

```bash
uv run python examples/models/build_example_model.py
```

The dashboard now needs only the config path:

```python
dashboard = TrackConfirmationDashboard(
    data,
    "examples/track_qa_config.json",
)
```

Or from the command line:

```bash
uv run track-confirmation-dashboard \
  --data tracks.parquet \
  --config examples/track_qa_config.json
```

For direct Python registration instead, the equivalent model defines a
deterministic score from `snr` and `residual`:

```python
import polars as pl

from track_dashboard import (
    MLConfirmationPath,
    MLModelSpec,
    TrackConfirmationDashboard,
)


def default_quality_score(features: pl.DataFrame) -> pl.Series:
    logits = (
        0.18 * features.get_column("snr")
        - 0.9 * features.get_column("residual")
        - 2.5
    )
    return 1 / (1 + (-logits).exp())


default_ml_path = MLConfirmationPath(
    path="default_ml_quality_path",
    model=MLModelSpec(
        name="example_default_quality_model",
        model=default_quality_score,
        features=("snr", "residual"),
    ),
    threshold=0.55,
)

dashboard = TrackConfirmationDashboard(
    data,
    config,
    default_ml_paths=[default_ml_path],
)
```

The `features` tuple defines the exact model-input order. A row triggers
`default_ml_quality_path` when its score is at least `0.55`; confirmation then
latches for the remaining measurements in that track.

Run the complete example with:

```bash
uv run panel serve examples/confirmation_app.py --show --autoreload
```

The ML path is evaluated as part of the original baseline logic, appears in the
enabled-path selector, and can be disabled in the candidate to measure lost
tracks. Its model is also available when creating experimental ML paths.

For an ONNX baseline path, register the model and reference it from the
confirmation CLI:

```bash
uv run track-confirmation-dashboard \
  --data tracks.parquet \
  --config confirmation.json \
  --onnx-model quality_model models/quality.onnx snr residual \
  --default-ml-path current_ml_quality quality_model 0.6
```

The baseline-path format is `PATH MODEL THRESHOLD`, where `MODEL` is a name
registered by `--onnx-model`.

The download button saves the complete JSON payload with the proposed paths at
`dynamic_specific.track_qa_config.experimental_paths`.

</details>

## Development

Run the test and lint suites before submitting changes:

```bash
uv run pytest
uv run ruff check .
```

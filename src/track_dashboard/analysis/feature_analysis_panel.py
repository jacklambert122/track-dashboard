from __future__ import annotations

import math

import hvplot.polars  # noqa: F401
import holoviews as hv
import panel as pn
import param
import polars as pl

from ..confirmation.engine import MLModelSpec
from ..core.base import AnalysisComponent
from .feature_analysis import (
    DEFAULT_MODEL_TYPES,
    FEATURE_ANALYSIS_METHODS,
    analyze_features,
    build_default_model,
)


class FeatureAnalysisPanel(AnalysisComponent):
    """On-demand feature analysis for the active dashboard dataframe."""

    features = param.ListSelector(default=[], objects=[])
    label_column = param.Selector(default=None, objects=[], allow_None=True)
    positive_class = param.Selector(default=None, objects=[], allow_None=True)
    methods = param.ListSelector(
        default=["Point-biserial correlation"],
        objects=list(FEATURE_ANALYSIS_METHODS),
    )
    model_source = param.Selector(
        default="Build default model",
        objects=["Build default model", "Use provided model"],
    )
    model_type = param.Selector(
        default=DEFAULT_MODEL_TYPES[0],
        objects=list(DEFAULT_MODEL_TYPES),
    )
    class_weight_mode = param.Selector(
        default="Balanced",
        objects=["None", "Balanced", "Custom"],
    )
    positive_class_weight = param.Number(default=1.0, bounds=(0.01, None))
    negative_class_weight = param.Number(default=1.0, bounds=(0.01, None))
    rf_n_estimators = param.Integer(default=200, bounds=(10, 5_000))
    rf_max_depth = param.Integer(default=0, bounds=(0, 1_000))
    rf_min_samples_leaf = param.Integer(default=2, bounds=(1, 10_000))
    rf_max_features = param.Selector(
        default="sqrt",
        objects=["sqrt", "log2", "All"],
    )
    rf_criterion = param.Selector(
        default="gini",
        objects=["gini", "entropy", "log_loss"],
    )
    rf_bootstrap = param.Boolean(default=True)
    logistic_c = param.Number(default=1.0, bounds=(0.0001, None))
    logistic_penalty = param.Selector(default="l2", objects=["l1", "l2"])
    logistic_max_iter = param.Integer(default=1_000, bounds=(50, 100_000))
    logistic_fit_intercept = param.Boolean(default=True)
    boosting_n_estimators = param.Integer(default=100, bounds=(10, 5_000))
    boosting_learning_rate = param.Number(default=0.1, bounds=(0.0001, 10.0))
    boosting_max_depth = param.Integer(default=3, bounds=(1, 1_000))
    boosting_min_samples_leaf = param.Integer(default=1, bounds=(1, 10_000))
    boosting_subsample = param.Number(default=1.0, bounds=(0.01, 1.0))
    shap_explainer = param.Selector(
        default="Auto",
        objects=["Auto", "Tree", "Linear", "Permutation"],
    )
    shap_waterfall_row = param.Integer(default=0, bounds=(0, 0))
    model = param.Selector(default=None, allow_None=True)
    sample_size = param.Integer(default=500, bounds=(20, 100_000))
    run_analysis = param.Event()
    result_revision = param.Integer(default=0, precedence=-1)
    message = param.String(default="")

    def __init__(
        self,
        *args,
        label_col: str = "label",
        matched_value: str = "matched",
        ml_models: list[MLModelSpec] | None = None,
        **kwargs,
    ) -> None:
        self.default_label_col = label_col
        self.default_matched_value = matched_value
        self.model_registry = {
            model.name: model for model in (ml_models or [])
        }
        if len(self.model_registry) != len(ml_models or []):
            raise ValueError("Registered feature-analysis model names must be unique.")
        self._result = self._empty_result()
        self._correlation_matrix = self._empty_correlation_matrix()
        self._shap_values = self._empty_shap_values()
        self._shap_base_values: list[float] = []
        super().__init__(*args, **kwargs)
        self.param.model.objects = [None, *self.model_registry]
        self.model = (
            next(iter(self.model_registry)) if self.model_registry else None
        )
        self.param.watch(self._run_analysis, "run_analysis")
        self.param.watch(self._label_changed, "label_column")
        self.refresh_options()

    @staticmethod
    def _empty_result() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "method": pl.String,
                "score": pl.Float64,
                "importance": pl.Float64,
                "direction": pl.String,
            }
        )

    @staticmethod
    def _empty_correlation_matrix() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "row": pl.String,
                "column": pl.String,
                "correlation": pl.Float64,
            }
        )

    @staticmethod
    def _empty_shap_values() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "feature": pl.String,
                "row_index": pl.Int64,
                "feature_value": pl.Float64,
                "relative_value": pl.Float64,
                "shap_value": pl.Float64,
                "density_offset": pl.Float64,
            }
        )

    def refresh_options(self) -> None:
        data = self.data_model.analysis_df()
        label_columns = [
            column
            for column in data.columns
            if 1 < data.get_column(column).drop_nulls().n_unique() <= 50
        ]
        current_label = self.label_column
        self.param.label_column.objects = label_columns
        if current_label in label_columns:
            self.label_column = current_label
        elif self.default_label_col in label_columns:
            self.label_column = self.default_label_col
        else:
            self.label_column = label_columns[0] if label_columns else None
        self._refresh_features()
        self._refresh_positive_classes()

    def _label_changed(self, _event=None) -> None:
        self._refresh_features()
        self._refresh_positive_classes()

    def _refresh_features(self) -> None:
        numeric = [
            feature
            for feature in self.data_model.numeric_features()
            if feature != self.label_column
        ]
        self.param.features.objects = numeric
        self.features = [feature for feature in self.features if feature in numeric]

    def _refresh_positive_classes(self) -> None:
        data = self.data_model.analysis_df()
        values = (
            data.get_column(self.label_column).drop_nulls().unique().to_list()
            if self.label_column in data.columns
            else []
        )
        values = sorted(values, key=str)
        current_value = self.positive_class
        self.param.positive_class.objects = values
        if current_value in values:
            self.positive_class = current_value
        elif (
            self.label_column == self.default_label_col
            and self.default_matched_value in values
        ):
            self.positive_class = self.default_matched_value
        else:
            self.positive_class = values[0] if values else None

    def _data_changed(self, event=None) -> None:
        self._result = self._empty_result()
        self._correlation_matrix = self._empty_correlation_matrix()
        self._shap_values = self._empty_shap_values()
        self._shap_base_values = []
        self.message = "Dashboard data changed. Run feature analysis again."
        self.result_revision += 1
        super()._data_changed(event)

    def _run_analysis(self, _event=None) -> None:
        try:
            analysis_data = self.data_model.analysis_df()
            selected_model = self._selected_model(analysis_data)
            self._shap_values = self._empty_shap_values()
            self._shap_base_values = []
            self._result = analyze_features(
                analysis_data,
                features=self.features,
                methods=self.methods,
                label_col=self.label_column or "",
                matched_value=self.positive_class,
                model=selected_model,
                shap_explainer=self.shap_explainer,
                shap_values_callback=self._capture_shap_values,
                sample_size=self.sample_size,
            )
            self._correlation_matrix = (
                self._build_correlation_matrix(analysis_data)
                if "Full correlation matrix" in self.methods
                else self._empty_correlation_matrix()
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._result = self._empty_result()
            self._correlation_matrix = self._empty_correlation_matrix()
            self._shap_values = self._empty_shap_values()
            self._shap_base_values = []
            self.message = str(exc)
            self.result_revision += 1
            return
        self.message = (
            f"Analyzed {len(self.features)} feature(s) using "
            f"{len(self.methods)} method(s)."
        )
        self.result_revision += 1

    def _capture_shap_values(
        self,
        feature_data: pl.DataFrame,
        features: list[str],
        shap_values,
        base_values,
    ) -> None:
        rows = []
        for feature_index, feature in enumerate(features):
            values = feature_data.get_column(feature).cast(pl.Float64)
            minimum = values.min()
            maximum = values.max()
            span = maximum - minimum if maximum != minimum else 1.0
            feature_shap = [
                float(value) for value in shap_values[:, feature_index]
            ]
            density_offsets = self._density_offsets(feature_shap)
            for row_index, (feature_value, shap_value) in enumerate(
                zip(values, feature_shap, strict=True)
            ):
                rows.append(
                    {
                        "feature": feature,
                        "row_index": row_index,
                        "feature_value": float(feature_value),
                        "relative_value": float(
                            (feature_value - minimum) / span
                        ),
                        "shap_value": float(shap_value),
                        "density_offset": density_offsets[row_index],
                    }
                )
        self._shap_values = pl.DataFrame(rows)
        self._shap_base_values = [float(value) for value in base_values]
        maximum_row = max(0, len(self._shap_base_values) - 1)
        self.param.shap_waterfall_row.bounds = (0, maximum_row)
        self.shap_waterfall_row = min(self.shap_waterfall_row, maximum_row)

    @staticmethod
    def _density_offsets(values: list[float]) -> list[float]:
        """Stack nearby SHAP values so dense x regions become vertically thick."""
        if len(values) < 2 or min(values) == max(values):
            return [0.0] * len(values)
        bin_count = max(8, min(40, round(math.sqrt(len(values)) * 2)))
        lower = min(values)
        width = (max(values) - lower) / bin_count
        bins: dict[int, list[int]] = {}
        for index, value in enumerate(values):
            bin_index = min(bin_count - 1, int((value - lower) / width))
            bins.setdefault(bin_index, []).append(index)
        maximum_count = max(len(indices) for indices in bins.values())
        offsets = [0.0] * len(values)
        for indices in bins.values():
            midpoint = (len(indices) - 1) / 2
            for rank, index in enumerate(indices):
                offsets[index] = (rank - midpoint) * 0.7 / maximum_count
        return offsets

    def _selected_model(self, data: pl.DataFrame) -> MLModelSpec | None:
        needs_model = bool(
            {"Permutation importance", "SHAP"} & set(self.methods)
        )
        if not needs_model:
            return None
        if self.model_source == "Build default model":
            return build_default_model(
                data,
                features=self.features,
                label_col=self.label_column or "",
                positive_class=self.positive_class,
                model_type=self.model_type,
                class_weight_mode=self.class_weight_mode,
                positive_class_weight=self.positive_class_weight,
                negative_class_weight=self.negative_class_weight,
                random_forest_params={
                    "n_estimators": self.rf_n_estimators,
                    "max_depth": self.rf_max_depth or None,
                    "min_samples_leaf": self.rf_min_samples_leaf,
                    "max_features": (
                        None
                        if self.rf_max_features == "All"
                        else self.rf_max_features
                    ),
                    "criterion": self.rf_criterion,
                    "bootstrap": self.rf_bootstrap,
                },
                logistic_regression_params={
                    "C": self.logistic_c,
                    "penalty": self.logistic_penalty,
                    "max_iter": self.logistic_max_iter,
                    "fit_intercept": self.logistic_fit_intercept,
                },
                gradient_boosting_params={
                    "n_estimators": self.boosting_n_estimators,
                    "learning_rate": self.boosting_learning_rate,
                    "max_depth": self.boosting_max_depth,
                    "min_samples_leaf": self.boosting_min_samples_leaf,
                    "subsample": self.boosting_subsample,
                },
            )
        if self.model is None:
            raise ValueError("Select a provided model for SHAP or permutation.")
        return self.model_registry[self.model]

    def _build_correlation_matrix(self, data: pl.DataFrame) -> pl.DataFrame:
        sample = (
            data.sample(n=self.sample_size, seed=7, shuffle=True)
            if data.height > self.sample_size
            else data
        )
        label_name = f"{self.label_column} = {self.positive_class}"
        expressions = [
            pl.col(feature).cast(pl.Float64).fill_nan(None).alias(feature)
            for feature in self.features
        ]
        frame = sample.select(
            *expressions,
            (pl.col(self.label_column) == self.positive_class)
            .cast(pl.Float64)
            .alias(label_name),
        )
        columns = [*self.features, label_name]
        rows = []
        for row_name in columns:
            for column_name in columns:
                correlation = frame.select(
                    pl.corr(row_name, column_name)
                ).item()
                value = float(correlation) if correlation is not None else 0.0
                if not math.isfinite(value):
                    value = 0.0
                rows.append(
                    {
                        "row": row_name,
                        "column": column_name,
                        "correlation": value,
                    }
                )
        return pl.DataFrame(rows)

    @param.depends("result_revision")
    def results(self):
        has_result_table = not self._result.is_empty()
        has_correlation_matrix = not self._correlation_matrix.is_empty()
        has_shap = not self._shap_values.is_empty()
        if not (has_result_table or has_correlation_matrix or has_shap):
            return pn.pane.Alert(
                self.message or "Select features and run an analysis.",
                alert_type="warning" if self.message else "info",
                sizing_mode="stretch_width",
            )
        outputs = []
        if has_result_table:
            table = pn.widgets.Tabulator(
                self._result.to_pandas(),
                pagination="local",
                page_size=30,
                height=430,
                sizing_mode="stretch_width",
            )
            non_shap = self._result.filter(pl.col("method") != "SHAP")
            if non_shap.is_empty():
                outputs.append(table)
            else:
                plot = non_shap.hvplot.bar(
                    x="feature",
                    y="importance",
                    by="method",
                    rot=45,
                    height=430,
                    responsive=True,
                    title="Feature importance by method",
                )
                outputs.append(
                    pn.Row(plot, table, sizing_mode="stretch_both")
                )
        if has_correlation_matrix:
            outputs.append(
                pn.Card(
                    self._correlation_matrix_plot(),
                    title="Full correlation matrix",
                    sizing_mode="stretch_width",
                )
            )
        shap_plot = self._shap_beeswarm()
        if shap_plot is not None:
            outputs.append(
                pn.Card(
                    self._shap_importance_bar(),
                    title="Mean absolute SHAP importance",
                    sizing_mode="stretch_width",
                )
            )
            outputs.append(
                pn.Card(
                    shap_plot,
                    title="SHAP beeswarm",
                    sizing_mode="stretch_width",
                )
            )
            outputs.append(
                pn.Card(
                    pn.widgets.IntSlider.from_param(
                        self.param.shap_waterfall_row,
                        name="Explained row",
                        sizing_mode="stretch_width",
                    ),
                    self.shap_waterfall,
                    title="SHAP force plot",
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(
            pn.pane.Alert(
                self.message,
                alert_type="success",
                sizing_mode="stretch_width",
            ),
            *outputs,
            sizing_mode="stretch_both",
        )

    def _correlation_matrix_plot(self):
        matrix_data = self._correlation_matrix.with_columns(
            pl.col("correlation").round(2).cast(pl.String).alias("display_value")
        )
        matrix_height = max(400, 45 * (len(self.features) + 1))
        heatmap = matrix_data.hvplot.heatmap(
            x="row",
            y="column",
            C="correlation",
            cmap="PuOr",
            clim=(-1, 1),
            colorbar=True,
            height=matrix_height,
            rot=45,
            title="Full feature and label correlation matrix",
        ).opts(tools=["hover"], invert_yaxis=True)
        labels = matrix_data.hvplot.labels(
            x="row",
            y="column",
            text="display_value",
        ).opts(
            height=matrix_height,
            text_color="black",
            text_font_size="9pt",
        )
        return heatmap * labels

    def _shap_importance_bar(self):
        shap_importance = (
            self._result.filter(pl.col("method") == "SHAP")
            .sort("importance")
        )
        return shap_importance.hvplot.barh(
            x="feature",
            y="importance",
            height=max(320, 55 * shap_importance.height),
            color="#7b3294",
            xlabel="Feature",
            ylabel="Mean absolute SHAP value",
            title="Global SHAP feature importance",
        )

    def _shap_beeswarm(self):
        if self._shap_values.is_empty():
            return None
        feature_order = list(
            self._result.filter(pl.col("method") == "SHAP")
            .sort("importance")
            .get_column("feature")
        )
        positions = {
            feature: index for index, feature in enumerate(feature_order)
        }
        plot_data = self._shap_values.with_columns(
            (
                pl.col("feature").replace_strict(positions).cast(pl.Float64)
                + pl.col("density_offset")
            ).alias("plot_y")
        )
        scatter = plot_data.hvplot.scatter(
            x="shap_value",
            y="plot_y",
            c="relative_value",
            cmap="coolwarm",
            clim=(0, 1),
            colorbar=True,
            hover_cols=["feature", "feature_value"],
            alpha=0.7,
            size=35,
            height=max(350, 70 * len(feature_order)),
            title="SHAP impact and feature value",
        ).opts(
            yticks=[(index, feature) for feature, index in positions.items()],
            ylabel="Feature",
            xlabel="SHAP value",
        )
        return scatter * hv.VLine(0).opts(
            color="gray",
            line_dash="dashed",
            line_width=1,
        )

    @param.depends("shap_waterfall_row")
    def shap_waterfall(self):
        if self._shap_values.is_empty() or not self._shap_base_values:
            return pn.pane.Markdown("Run SHAP to create a waterfall plot.")
        row_index = self.shap_waterfall_row
        row_values = (
            self._shap_values.filter(pl.col("row_index") == row_index)
            .with_columns(pl.col("shap_value").abs().alias("__absolute"))
            .sort("__absolute", descending=True)
        )
        baseline = self._shap_base_values[row_index]
        output = baseline + row_values.get_column("shap_value").sum()
        cumulative = baseline
        polygons = []
        labels = []
        for row in row_values.iter_rows(named=True):
            contribution = row["shap_value"]
            next_value = cumulative + contribution
            direction = "Increases output" if contribution >= 0 else "Decreases output"
            arrow_width = min(
                abs(contribution) * 0.3,
                max(abs(output - baseline), 0.01) * 0.04,
            )
            if contribution >= 0:
                coordinates = [
                    (cumulative, -0.28),
                    (next_value - arrow_width, -0.28),
                    (next_value, 0.0),
                    (next_value - arrow_width, 0.28),
                    (cumulative, 0.28),
                ]
            else:
                coordinates = [
                    (cumulative, -0.28),
                    (next_value + arrow_width, -0.28),
                    (next_value, 0.0),
                    (next_value + arrow_width, 0.28),
                    (cumulative, 0.28),
                ]
            polygons.append(
                {
                    "x": [point[0] for point in coordinates],
                    "y": [point[1] for point in coordinates],
                    "direction": direction,
                    "feature": row["feature"],
                    "feature_value": row["feature_value"],
                    "contribution": contribution,
                }
            )
            labels.append(
                (
                    (cumulative + next_value) / 2,
                    0.42 if contribution >= 0 else -0.42,
                    (
                        f"{row['feature']}={row['feature_value']:.3g} "
                        f"({contribution:+.3f})"
                    ),
                )
            )
            cumulative = next_value
        force = hv.Polygons(
            polygons,
            vdims=["direction", "feature", "feature_value", "contribution"],
        ).opts(
            color="direction",
            cmap={
                "Increases output": "#f04b4c",
                "Decreases output": "#3b8ed0",
            },
            height=330,
            show_legend=True,
            tools=["hover"],
            xaxis="bottom",
            yaxis=None,
            ylim=(-0.85, 0.85),
            xlabel="Model output",
            title=(
                f"Row {row_index}: base value {baseline:.3f} "
                f"→ prediction {output:.3f}"
            ),
        )
        value_labels = hv.Labels(
            labels,
            kdims=["x", "y"],
            vdims=["text"],
        ).opts(
            text_font_size="8pt",
            text_color="black",
            text_align="center",
        )
        baseline_line = hv.VLine(baseline).opts(
            color="#666666",
            line_dash="dashed",
            line_width=1,
        )
        output_line = hv.VLine(output).opts(
            color="black",
            line_width=2,
        )
        markers = hv.Labels(
            [
                (baseline, 0.72, f"base {baseline:.3f}"),
                (output, 0.72, f"prediction {output:.3f}"),
            ],
            kdims=["x", "y"],
            vdims=["text"],
        ).opts(text_font_size="9pt", text_color="black")
        return force * value_labels * baseline_line * output_line * markers

    @param.depends("model_source")
    def model_configuration(self):
        if self.model_source == "Build default model":
            return pn.Column(
                pn.widgets.Select.from_param(
                    self.param.model_type,
                    name="Default model type",
                    sizing_mode="stretch_width",
                ),
                self.default_model_options,
                sizing_mode="stretch_width",
            )
        if not self.model_registry:
            return pn.pane.Alert(
                "No models were provided to DashboardApp.",
                alert_type="warning",
                sizing_mode="stretch_width",
            )
        return pn.widgets.Select.from_param(
            self.param.model,
            name="Provided model",
            sizing_mode="stretch_width",
        )

    @param.depends("model_type", "class_weight_mode")
    def default_model_options(self):
        controls = [
            pn.widgets.Select.from_param(
                self.param.class_weight_mode,
                name="Label class weighting",
                sizing_mode="stretch_width",
            )
        ]
        if self.class_weight_mode == "Custom":
            controls.append(
                pn.Row(
                    pn.widgets.FloatInput.from_param(
                        self.param.positive_class_weight,
                        name="Positive-class weight",
                        sizing_mode="stretch_width",
                    ),
                    pn.widgets.FloatInput.from_param(
                        self.param.negative_class_weight,
                        name="Other-class weight",
                        sizing_mode="stretch_width",
                    ),
                    sizing_mode="stretch_width",
                )
            )
        if self.model_type == "Random forest":
            controls.append(
                pn.Card(
                    pn.Row(
                        pn.widgets.IntInput.from_param(
                            self.param.rf_n_estimators,
                            name="Number of trees",
                        ),
                        pn.widgets.IntInput.from_param(
                            self.param.rf_max_depth,
                            name="Maximum depth (0 = unlimited)",
                        ),
                    ),
                    pn.Row(
                        pn.widgets.IntInput.from_param(
                            self.param.rf_min_samples_leaf,
                            name="Minimum samples per leaf",
                        ),
                        pn.widgets.Select.from_param(
                            self.param.rf_max_features,
                            name="Features per split",
                        ),
                    ),
                    pn.Row(
                        pn.widgets.Select.from_param(
                            self.param.rf_criterion,
                            name="Split criterion",
                        ),
                        pn.widgets.Checkbox.from_param(
                            self.param.rf_bootstrap,
                            name="Bootstrap samples",
                        ),
                    ),
                    title="Random forest hyperparameters",
                    sizing_mode="stretch_width",
                )
            )
        elif self.model_type == "Logistic regression":
            controls.append(
                pn.Card(
                    pn.Row(
                        pn.widgets.FloatInput.from_param(
                            self.param.logistic_c,
                            name="Inverse regularization (C)",
                        ),
                        pn.widgets.Select.from_param(
                            self.param.logistic_penalty,
                            name="Penalty",
                        ),
                    ),
                    pn.Row(
                        pn.widgets.IntInput.from_param(
                            self.param.logistic_max_iter,
                            name="Maximum iterations",
                        ),
                        pn.widgets.Checkbox.from_param(
                            self.param.logistic_fit_intercept,
                            name="Fit intercept",
                        ),
                    ),
                    title="Logistic regression hyperparameters",
                    sizing_mode="stretch_width",
                )
            )
        elif self.model_type == "Gradient boosting":
            controls.append(
                pn.Card(
                    pn.Row(
                        pn.widgets.IntInput.from_param(
                            self.param.boosting_n_estimators,
                            name="Number of boosting stages",
                        ),
                        pn.widgets.FloatInput.from_param(
                            self.param.boosting_learning_rate,
                            name="Learning rate",
                        ),
                    ),
                    pn.Row(
                        pn.widgets.IntInput.from_param(
                            self.param.boosting_max_depth,
                            name="Maximum tree depth",
                        ),
                        pn.widgets.IntInput.from_param(
                            self.param.boosting_min_samples_leaf,
                            name="Minimum samples per leaf",
                        ),
                    ),
                    pn.widgets.FloatSlider.from_param(
                        self.param.boosting_subsample,
                        name="Training-row subsample",
                        step=0.05,
                        sizing_mode="stretch_width",
                    ),
                    title="Gradient boosting hyperparameters",
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(*controls, sizing_mode="stretch_width")

    @param.depends("methods")
    def shap_options(self):
        if "SHAP" not in self.methods:
            return pn.Spacer(height=0)
        return pn.widgets.Select.from_param(
            self.param.shap_explainer,
            name="SHAP explainer",
            sizing_mode="stretch_width",
        )

    def view(self) -> pn.Column:
        controls = pn.Card(
            pn.pane.Markdown(
                "Choose the label and positive class, then select numeric "
                "features. **Incremental CV ROC AUC** measures the AUC each "
                "feature adds beyond the other selected features. **Mutual "
                "information** measures each feature's non-linear relationship "
                "to the selected label."
            ),
            pn.Row(
                pn.widgets.Select.from_param(
                    self.param.label_column,
                    name="Label column",
                    sizing_mode="stretch_width",
                ),
                pn.widgets.Select.from_param(
                    self.param.positive_class,
                    name="Positive class",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            pn.widgets.MultiChoice.from_param(
                self.param.features,
                name="Features to include",
                sizing_mode="stretch_width",
            ),
            pn.widgets.MultiChoice.from_param(
                self.param.methods,
                name="Analysis methods",
                sizing_mode="stretch_width",
            ),
            self.shap_options,
            pn.widgets.RadioButtonGroup.from_param(
                self.param.model_source,
                name="SHAP / permutation model source",
                button_type="primary",
                sizing_mode="stretch_width",
            ),
            self.model_configuration,
            pn.widgets.IntInput.from_param(
                self.param.sample_size,
                name="Maximum sampled rows",
                sizing_mode="stretch_width",
            ),
            pn.widgets.Button.from_param(
                self.param.run_analysis,
                name="Run feature analysis",
                color="primary",
                sizing_mode="stretch_width",
            ),
            title="Feature analysis configuration",
            sizing_mode="stretch_width",
        )
        return pn.Column(controls, self.results, sizing_mode="stretch_both")

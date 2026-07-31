from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import hvplot.polars  # noqa: F401
import panel as pn
import param
import polars as pl

from ..core.filters import FilterPanel
from ..core.state import INTERNAL_ROW_ID, DashboardState
from ..core.theme import apply_dark_theme
from ..models.registry import (
    load_configured_default_paths,
    load_configured_models,
)
from .engine import (
    ConfirmationComparison,
    ConfirmationEvaluator,
    MLConfirmationPath,
    MLModelSpec,
    RangeRule,
    candidate_config,
    evaluate_comparison,
    generic_confirmation_evaluator,
    load_confirmation_config,
    track_qa_config,
)


class TrackConfirmationDashboard(param.Parameterized):
    """Interactive comparison of default and experimental confirmation paths."""

    rule_revision = param.Integer(default=0, precedence=-1)
    enabled_default_paths = param.ListSelector(default=[], objects=[])
    path_name = param.String(default="experimental_path")
    feature = param.Selector(default=None, allow_None=True)
    minimum = param.Number(default=None, allow_None=True)
    maximum = param.Number(default=None, allow_None=True)
    add_rule = param.Event()
    remove_rules = param.Event()
    ml_model = param.Selector(default=None, allow_None=True)
    ml_path_name = param.String(default="ml_confirmation_path")
    ml_threshold = param.Number(default=0.5, bounds=(0.0, 1.0))
    add_ml_path = param.Event()
    remove_ml_paths = param.Event()
    status_message = param.String(default="")
    status_type = param.Selector(
        default="info",
        objects=["info", "success", "warning", "danger"],
    )
    changed_track_features = param.ListSelector(default=[], objects=[])
    changed_track_x = param.Selector(default=None, objects=[], allow_None=True)
    changed_track_y = param.Selector(default=None, objects=[], allow_None=True)
    changed_track_label = param.Selector(default=None, objects=[], allow_None=True)

    def __init__(
        self,
        data: pl.DataFrame,
        config: dict[str, Any] | str | Path,
        *,
        evaluator: ConfirmationEvaluator = generic_confirmation_evaluator,
        default_path_columns: list[str] | None = None,
        track_id_col: str = "track_id",
        time_col: str = "time",
        label_col: str = "label",
        matched_value: str = "matched",
        ml_models: list[MLModelSpec] | None = None,
        default_ml_paths: list[MLConfirmationPath] | None = None,
        model_base_dir: str | Path | None = None,
        **params,
    ) -> None:
        apply_dark_theme()
        super().__init__(**params)
        self.data = data
        config_path = Path(config) if isinstance(config, (str, Path)) else None
        self.config_payload = (
            load_confirmation_config(config_path)
            if config_path is not None
            else config
        )
        confirmation_config = track_qa_config(self.config_payload)
        self.evaluator = evaluator
        self.default_path_columns = default_path_columns
        self.track_id_col = track_id_col
        self.time_col = time_col
        self.label_col = label_col
        self.matched_value = matched_value
        configured_models = load_configured_models(
            self.config_payload,
            base_dir=(
                model_base_dir
                if model_base_dir is not None
                else (
                    config_path.parent
                    if config_path is not None
                    else Path.cwd()
                )
            ),
        )
        registered_models = [
            *configured_models,
            *(ml_models or []),
            *(path.model for path in (default_ml_paths or [])),
        ]
        self.ml_model_registry = {
            model.name: model for model in registered_models
        }
        if len(self.ml_model_registry) != len(registered_models):
            raise ValueError("Registered ML model names must be unique.")
        self.default_ml_paths = [
            *load_configured_default_paths(
                self.config_payload,
                registered_models,
            ),
            *(default_ml_paths or []),
        ]
        self.param.ml_model.objects = list(self.ml_model_registry)
        self.ml_model = (
            next(iter(self.ml_model_registry))
            if self.ml_model_registry
            else None
        )
        self.rules = self._rules_from_config(confirmation_config)
        self.ml_paths = self._ml_paths_from_config(confirmation_config)
        self.filter_state = DashboardState(data, track_id_col=track_id_col)
        self.filters = FilterPanel(self.filter_state)
        initial_comparison = evaluate_comparison(
            self._filtered_data(),
            self.config_payload,
            rules=self.rules,
            ml_paths=self.ml_paths,
            default_ml_paths=self.default_ml_paths,
            evaluator=self.evaluator,
            default_path_columns=self.default_path_columns,
            track_id_col=self.track_id_col,
            time_col=self.time_col,
            label_col=self.label_col,
            matched_value=self.matched_value,
        )
        available_paths = list(initial_comparison.default_path_columns)
        self.available_default_paths = available_paths
        configured_paths = track_qa_config(self.config_payload).get("enabled_paths")
        enabled_paths = (
            available_paths
            if configured_paths is None
            else [
                path for path in configured_paths if path in available_paths
            ]
        )
        self.param.enabled_default_paths.objects = available_paths
        self.enabled_default_paths = enabled_paths
        self._cached_revision = (
            self.rule_revision if enabled_paths == available_paths else -1
        )
        self._cached_comparison: ConfirmationComparison | None = (
            initial_comparison if enabled_paths == available_paths else None
        )

        numeric_features = [
            column
            for column, dtype in data.schema.items()
            if dtype.is_numeric()
            and column not in {track_id_col, time_col}
        ]
        self.param.feature.objects = numeric_features
        self.feature = numeric_features[0] if numeric_features else None
        self.param.changed_track_features.objects = numeric_features
        self.changed_track_features = numeric_features[:1]
        scatter_features = [
            column
            for column, dtype in data.schema.items()
            if dtype.is_numeric() and column != track_id_col
        ]
        self.param.changed_track_x.objects = scatter_features
        self.param.changed_track_y.objects = scatter_features
        self.changed_track_x = (
            time_col
            if time_col in scatter_features
            else (scatter_features[0] if scatter_features else None)
        )
        self.changed_track_y = (
            numeric_features[0]
            if numeric_features
            else (scatter_features[0] if scatter_features else None)
        )
        label_options = [
            column
            for column in data.columns
            if data.get_column(column).drop_nulls().n_unique() <= 50
            and column not in {track_id_col, time_col}
        ]
        self.param.changed_track_label.objects = label_options
        self.changed_track_label = (
            label_col
            if label_col in label_options
            else (label_options[0] if label_options else None)
        )

        self.rule_table = pn.widgets.Tabulator(
            self._rules_frame().to_pandas(),
            selectable="checkbox",
            pagination="local",
            page_size=8,
            height=260,
            sizing_mode="stretch_width",
        )
        self.ml_path_table = pn.widgets.Tabulator(
            self._ml_paths_frame().to_pandas(),
            selectable="checkbox",
            pagination="local",
            page_size=6,
            height=220,
            sizing_mode="stretch_width",
        )
        self.param.watch(self._add_rule, "add_rule")
        self.param.watch(self._remove_rules, "remove_rules")
        self.param.watch(self._add_ml_path, "add_ml_path")
        self.param.watch(self._remove_ml_paths, "remove_ml_paths")
        self.param.watch(self._default_paths_changed, "enabled_default_paths")
        self.filter_state.param.watch(self._filters_changed, "data_revision")

    @staticmethod
    def _rules_from_config(config: dict[str, Any]) -> list[RangeRule]:
        rules = []
        for path in config.get("experimental_paths", []):
            for feature, bounds in path.get("ranges", {}).items():
                rules.append(
                    RangeRule(
                        path=path["name"],
                        feature=feature,
                        minimum=bounds.get("min"),
                        maximum=bounds.get("max"),
                    )
                )
        return rules

    def _ml_paths_from_config(
        self,
        config: dict[str, Any],
    ) -> list[MLConfirmationPath]:
        paths = []
        for path in config.get("ml_confirmation_paths", []):
            model_name = path["model"]
            if model_name not in self.ml_model_registry:
                raise ValueError(
                    f"Config ML path {path['name']!r} requires unregistered "
                    f"model {model_name!r}."
                )
            paths.append(
                MLConfirmationPath(
                    path=path["name"],
                    model=self.ml_model_registry[model_name],
                    threshold=path.get("threshold", 0.5),
                )
            )
        return paths

    def _comparison(self) -> ConfirmationComparison:
        if (
            self._cached_comparison is None
            or self._cached_revision != self.rule_revision
        ):
            self._cached_comparison = evaluate_comparison(
                self._filtered_data(),
                self.config_payload,
                rules=self.rules,
                ml_paths=self.ml_paths,
                default_ml_paths=self.default_ml_paths,
                evaluator=self.evaluator,
                default_path_columns=self.available_default_paths,
                candidate_default_path_columns=self.enabled_default_paths,
                track_id_col=self.track_id_col,
                time_col=self.time_col,
                label_col=self.label_col,
                matched_value=self.matched_value,
            )
            self._cached_revision = self.rule_revision
        return self._cached_comparison

    def _filtered_data(self) -> pl.DataFrame:
        data = self.filter_state.point_df
        for expression in self.filter_state.filter_expressions:
            data = data.filter(expression)
        return data.drop(INTERNAL_ROW_ID)

    def _default_paths_changed(self, _event=None) -> None:
        self.rule_revision += 1

    def _filters_changed(self, _event=None) -> None:
        self.rule_revision += 1

    def _add_rule(self, _event=None) -> None:
        try:
            rule = RangeRule(
                path=self.path_name.strip(),
                feature=self.feature or "",
                minimum=self.minimum,
                maximum=self.maximum,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self._set_status(str(exc), "danger")
            return
        self.rules.append(rule)
        self.minimum = None
        self.maximum = None
        self._rules_changed()
        self._set_status(
            f"Added range for `{rule.feature}` to `{rule.path}`.",
            "success",
        )

    def _remove_rules(self, _event=None) -> None:
        selected = set(self.rule_table.selection)
        if not selected:
            self._set_status("Select rule rows to remove.", "warning")
            return
        self.rules = [
            rule for index, rule in enumerate(self.rules) if index not in selected
        ]
        self.rule_table.selection = []
        self._rules_changed()
        self._set_status(f"Removed {len(selected)} rule(s).", "success")

    def _rules_changed(self) -> None:
        self.rule_table.value = self._rules_frame().to_pandas()
        self.rule_revision += 1

    def _add_ml_path(self, _event=None) -> None:
        if self.ml_model is None:
            self._set_status(
                "Register an ML model in Python before adding an ML path.",
                "danger",
            )
            return
        path_name = self.ml_path_name.strip()
        existing_paths = {
            *self.enabled_default_paths,
            *(rule.path for rule in self.rules),
            *(path.path for path in self.ml_paths),
        }
        if path_name in existing_paths:
            self._set_status(
                f"Confirmation path {path_name!r} already exists.",
                "danger",
            )
            return
        try:
            path = MLConfirmationPath(
                path=path_name,
                model=self.ml_model_registry[self.ml_model],
                threshold=float(self.ml_threshold),
            )
        except ValueError as exc:
            self._set_status(str(exc), "danger")
            return
        self.ml_paths.append(path)
        self._ml_paths_changed()
        self._set_status(
            f"Added ML path `{path.path}` using `{path.model.name}`.",
            "success",
        )

    def _remove_ml_paths(self, _event=None) -> None:
        selected = set(self.ml_path_table.selection)
        if not selected:
            self._set_status("Select ML path rows to remove.", "warning")
            return
        self.ml_paths = [
            path
            for index, path in enumerate(self.ml_paths)
            if index not in selected
        ]
        self.ml_path_table.selection = []
        self._ml_paths_changed()
        self._set_status(f"Removed {len(selected)} ML path(s).", "success")

    def _ml_paths_changed(self) -> None:
        self.ml_path_table.value = self._ml_paths_frame().to_pandas()
        self.rule_revision += 1

    @param.depends("rule_revision")
    def experimental_path_summary(self):
        paths = list(
            dict.fromkeys(
                [*(rule.path for rule in self.rules), *(p.path for p in self.ml_paths)]
            )
        )
        if not paths:
            return pn.pane.Markdown("No experimental paths defined.")
        return pn.pane.Markdown(
            "**Active experimental paths:** " + ", ".join(f"`{path}`" for path in paths)
        )

    def _rules_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "path": [rule.path for rule in self.rules],
                "feature": [rule.feature for rule in self.rules],
                "minimum": [rule.minimum for rule in self.rules],
                "maximum": [rule.maximum for rule in self.rules],
            },
            schema={
                "path": pl.String,
                "feature": pl.String,
                "minimum": pl.Float64,
                "maximum": pl.Float64,
            },
        )

    def _ml_paths_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "path": [path.path for path in self.ml_paths],
                "model": [path.model.name for path in self.ml_paths],
                "features": [
                    ", ".join(path.model.features) for path in self.ml_paths
                ],
                "threshold": [path.threshold for path in self.ml_paths],
            },
            schema={
                "path": pl.String,
                "model": pl.String,
                "features": pl.String,
                "threshold": pl.Float64,
            },
        )

    def _set_status(self, message: str, status_type: str) -> None:
        self.status_message = message
        self.status_type = status_type

    @param.depends("status_message", "status_type")
    def status(self):
        return pn.pane.Alert(
            self.status_message,
            alert_type=self.status_type,
            visible=bool(self.status_message),
            sizing_mode="stretch_width",
        )

    def _download_candidate_config(self) -> BytesIO:
        payload = candidate_config(
            self.config_payload,
            self.rules,
            enabled_default_paths=self.enabled_default_paths,
            ml_paths=self.ml_paths,
        )
        return BytesIO(json.dumps(payload, indent=2).encode())

    def controls(self) -> pn.Column:
        rule_inputs = pn.Column(
            pn.widgets.MultiChoice.from_param(
                self.param.enabled_default_paths,
                name="Enabled default confirmation paths",
                sizing_mode="stretch_width",
            ),
            pn.Card(
                self.filters.view(),
                title="Data filters",
                collapsed=True,
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(),
            pn.widgets.TextInput.from_param(
                self.param.path_name,
                name="Experimental path name",
                placeholder="Reuse a name or enter another path",
                sizing_mode="stretch_width",
            ),
            pn.widgets.Select.from_param(
                self.param.feature,
                name="Track feature",
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.widgets.FloatInput.from_param(
                    self.param.minimum,
                    name="Minimum",
                    sizing_mode="stretch_width",
                ),
                pn.widgets.FloatInput.from_param(
                    self.param.maximum,
                    name="Maximum",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            pn.widgets.Button.from_param(
                self.param.add_rule,
                name="Add range rule",
                button_type="primary",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
        if self.ml_model_registry:
            ml_controls = pn.Card(
                pn.widgets.Select.from_param(
                    self.param.ml_model,
                    name="Registered model",
                    sizing_mode="stretch_width",
                ),
                pn.widgets.TextInput.from_param(
                    self.param.ml_path_name,
                    name="ML confirmation path",
                    sizing_mode="stretch_width",
                ),
                pn.widgets.FloatSlider.from_param(
                    self.param.ml_threshold,
                    name="Probability threshold",
                    step=0.01,
                    sizing_mode="stretch_width",
                ),
                pn.widgets.Button.from_param(
                    self.param.add_ml_path,
                    name="Add ML path",
                    color="primary",
                    sizing_mode="stretch_width",
                ),
                self.ml_path_table,
                pn.widgets.Button.from_param(
                    self.param.remove_ml_paths,
                    name="Remove selected ML paths",
                    color="danger",
                    sizing_mode="stretch_width",
                ),
                title="ML confirmation paths",
                collapsed=False,
                sizing_mode="stretch_width",
            )
        else:
            ml_controls = pn.Card(
                pn.pane.Markdown(
                    "No ML models registered. Pass `ml_models=[...]` when "
                    "creating the dashboard."
                ),
                title="ML confirmation paths",
                collapsed=True,
                sizing_mode="stretch_width",
            )
        return pn.Column(
            pn.pane.Markdown(
                "Ranges in the same path are combined with **AND**. "
                "Different paths are combined with **OR**."
            ),
            rule_inputs,
            self.rule_table,
            self.experimental_path_summary,
            pn.widgets.Button.from_param(
                self.param.remove_rules,
                name="Remove selected rules",
                button_type="danger",
                sizing_mode="stretch_width",
            ),
            ml_controls,
            pn.widgets.FileDownload(
                callback=self._download_candidate_config,
                filename="candidate_track_confirmation.json",
                label="Download candidate config",
                button_type="success",
                sizing_mode="stretch_width",
            ),
            self.status,
            width=520,
            min_width=440,
            sizing_mode=None,
        )

    @param.depends("rule_revision")
    def headline_metrics(self):
        metrics = self._comparison().metrics
        lost_tracks = self._changed_tracks_frame().filter(
            pl.col("confirmation_change") == "Lost"
        ).height
        cards = [
            ("Default confirmed tracks", metrics["default_confirmed_tracks"]),
            ("Candidate confirmed tracks", metrics["candidate_confirmed_tracks"]),
            ("New tracks", metrics["new_confirmed_tracks"]),
            ("Lost tracks", lost_tracks),
            ("New measurements", metrics["new_confirmed_measurements"]),
            (
                "Matched confirmed points",
                metrics["matched_confirmed_measurements"],
            ),
            (
                "Point false-alarm rate",
                f"{metrics['point_false_alarm_rate']:.2%}",
            ),
            (
                "Track false-alarm rate",
                f"{metrics['track_false_alarm_rate']:.2%}",
            ),
        ]
        return pn.FlexBox(
            *[
                pn.Card(
                    pn.pane.Markdown(f"## {value}"),
                    title=title,
                    width=220,
                    height=120,
                )
                for title, value in cards
            ],
            sizing_mode="stretch_width",
        )

    @param.depends("rule_revision")
    def first_confirmation_plot(self):
        first_times = self._comparison().first_times.filter(
            pl.col("default_first_confirmation_time").is_not_null()
            & pl.col("candidate_first_confirmation_time").is_not_null()
        )
        if first_times.is_empty():
            return pn.pane.Markdown(
                "No tracks are confirmed by both configurations."
            )
        return first_times.hvplot.scatter(
            x="default_first_confirmation_time",
            y="candidate_first_confirmation_time",
            color=self.label_col,
            hover_cols=[self.track_id_col],
            height=430,
            responsive=True,
            title="First confirmation time: default vs candidate",
        )

    @param.depends("rule_revision")
    def first_confirmation_distributions(self):
        first_times = self._comparison().first_times
        timing_frames = []
        for logic, column in (
            ("Default", "default_first_confirmation_time"),
            ("Candidate", "candidate_first_confirmation_time"),
        ):
            timing_frames.append(
                first_times.select(
                    pl.col(column).alias("first_confirmation_time")
                )
                .drop_nulls()
                .with_columns(pl.lit(logic).alias("logic"))
            )
        timing_data = pl.concat(timing_frames)
        if timing_data.is_empty():
            return pn.pane.Markdown("No confirmed tracks to plot.")

        histogram = timing_data.hvplot.hist(
            y="first_confirmation_time",
            by="logic",
            bins=40,
            alpha=0.55,
            height=400,
            responsive=True,
            title="First confirmation time histogram",
        )
        cdf_data = timing_data.sort(
            ["logic", "first_confirmation_time"]
        ).with_columns(
            (
                pl.col("first_confirmation_time")
                .rank(method="max")
                .over("logic")
                / pl.len().over("logic")
            ).alias("cdf")
        )
        cdf = cdf_data.hvplot.step(
            x="first_confirmation_time",
            y="cdf",
            by="logic",
            height=400,
            responsive=True,
            ylabel="Cumulative probability",
            title="First confirmation time CDF",
        )
        return pn.Row(histogram, cdf, sizing_mode="stretch_both")

    @param.depends("rule_revision")
    def timing_table(self):
        return pn.widgets.Tabulator(
            self._comparison().first_times.to_pandas(),
            pagination="remote",
            page_size=20,
            height=440,
            sizing_mode="stretch_width",
        )

    @param.depends("rule_revision")
    def first_confirmation_details(self):
        return pn.Column(
            pn.Card(
                self.first_confirmation_distributions,
                title="First confirmation distributions",
                sizing_mode="stretch_width",
            ),
            pn.Card(
                self.timing_table,
                title="First confirmation times by track",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
        )

    @param.depends("rule_revision")
    def label_summary(self):
        return pn.widgets.Tabulator(
            self._comparison().label_summary.to_pandas(),
            pagination="local",
            height=300,
            sizing_mode="stretch_width",
        )

    @param.depends("rule_revision")
    def path_summary(self):
        return pn.widgets.Tabulator(
            self._comparison().path_summary.to_pandas(),
            pagination="local",
            page_size=20,
            height=380,
            sizing_mode="stretch_width",
        )

    @param.depends("rule_revision")
    def evaluated_points(self):
        return pn.widgets.Tabulator(
            self._comparison().points.to_pandas(),
            pagination="remote",
            page_size=25,
            height=520,
            sizing_mode="stretch_width",
        )

    def changed_track_data(self) -> pl.DataFrame:
        comparison = self._comparison()
        changed_tracks = self._changed_tracks_frame()
        changed_ids = changed_tracks.get_column(self.track_id_col)
        base_columns = [
            self.track_id_col,
            self.time_col,
            *(
                [self.changed_track_label]
                if self.changed_track_label is not None
                else []
            ),
            *self.changed_track_features,
            *(
                [self.changed_track_x]
                if self.changed_track_x is not None
                else []
            ),
            *(
                [self.changed_track_y]
                if self.changed_track_y is not None
                else []
            ),
        ]
        base_columns = list(dict.fromkeys(base_columns))
        changed_points = comparison.points.filter(
            pl.col(self.track_id_col).is_in(changed_ids)
        ).join(
            changed_tracks.select(
                self.track_id_col,
                "confirmation_change",
            ),
            on=self.track_id_col,
            how="left",
        )
        base_columns.append("confirmation_change")
        frames = []
        for logic, confirmed_column, first_time_column in (
            (
                "Default",
                "default_confirmed",
                "default_first_confirmation_time",
            ),
            (
                "Candidate",
                "candidate_confirmed",
                "candidate_first_confirmation_time",
            ),
        ):
            frames.append(
                changed_points.filter(pl.col(confirmed_column))
                .select(
                    *base_columns,
                    pl.col(first_time_column).alias(
                        "first_confirmation_time"
                    ),
                )
                .with_columns(
                    pl.lit(logic).alias("confirmation_logic"),
                )
            )
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal").sort(
            [self.track_id_col, self.time_col, "confirmation_logic"]
        )

    def _changed_tracks_frame(self) -> pl.DataFrame:
        return (
            self._comparison()
            .first_times.filter(
                pl.col("default_first_confirmation_time").ne_missing(
                    pl.col("candidate_first_confirmation_time")
                )
            )
            .with_columns(
                pl.when(
                    pl.col("default_first_confirmation_time").is_not_null()
                    & pl.col("candidate_first_confirmation_time").is_null()
                )
                .then(pl.lit("Lost"))
                .when(
                    pl.col("default_first_confirmation_time").is_null()
                    & pl.col("candidate_first_confirmation_time").is_not_null()
                )
                .then(pl.lit("Added"))
                .when(
                    pl.col("candidate_first_confirmation_time")
                    < pl.col("default_first_confirmation_time")
                )
                .then(pl.lit("Earlier"))
                .otherwise(pl.lit("Later"))
                .alias("confirmation_change")
            )
        )

    @param.depends(
        "rule_revision",
        "changed_track_features",
        "changed_track_x",
        "changed_track_y",
        "changed_track_label",
    )
    def changed_tracks_view(self):
        changed_tracks = self._changed_tracks_frame()
        controls = pn.Row(
            pn.widgets.Select.from_param(
                self.param.changed_track_x,
                name="Scatter X",
                sizing_mode="stretch_width",
            ),
            pn.widgets.Select.from_param(
                self.param.changed_track_y,
                name="Scatter Y",
                sizing_mode="stretch_width",
            ),
            pn.widgets.MultiChoice.from_param(
                self.param.changed_track_features,
                name="Additional table features",
                sizing_mode="stretch_width",
            ),
            pn.widgets.Select.from_param(
                self.param.changed_track_label,
                name="Label column",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            css_classes=["changed-track-controls"],
        )
        if changed_tracks.is_empty():
            return pn.Column(
                controls,
                pn.pane.Alert(
                    "No track confirmation status or timing changed.",
                    alert_type="info",
                    sizing_mode="stretch_width",
                ),
            )

        data = self.changed_track_data()
        scatter = (
            data.hvplot.scatter(
                x=self.changed_track_x,
                y=self.changed_track_y,
                by="confirmation_logic",
                hover_cols=[
                    self.track_id_col,
                    *(
                        [self.changed_track_label]
                        if self.changed_track_label is not None
                        else []
                    ),
                    self.time_col,
                    "first_confirmation_time",
                    "confirmation_change",
                ],
                alpha=0.65,
                height=460,
                responsive=True,
                title=(
                    f"{self.changed_track_y} vs {self.changed_track_x}: "
                    "default and candidate"
                ),
            )
            if self.changed_track_x is not None
            and self.changed_track_y is not None
            else pn.pane.Alert(
                "Select numeric X and Y features to plot.",
                alert_type="info",
                sizing_mode="stretch_width",
            )
        )
        changed_table = pn.widgets.Tabulator(
            changed_tracks.to_pandas(),
            pagination="local",
            page_size=20,
            height=300,
            sizing_mode="stretch_width",
        )
        point_table = pn.widgets.Tabulator(
            data.to_pandas(),
            pagination="remote",
            page_size=25,
            height=440,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            controls,
            pn.pane.Markdown(
                f"**{changed_tracks.height} tracks changed** between default "
                "and candidate confirmation."
            ),
            scatter,
            pn.Card(
                changed_table,
                title="Changed tracks",
                sizing_mode="stretch_width",
            ),
            pn.Card(
                point_table,
                title="Default / candidate confirmed measurements",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
        )

    def view(self) -> pn.Row:
        tabs = pn.Tabs(
            (
                "Overview",
                pn.Column(
                    self.headline_metrics,
                    self.first_confirmation_plot,
                    sizing_mode="stretch_both",
                ),
            ),
            ("First confirmation", self.first_confirmation_details),
            ("Changed tracks", self.changed_tracks_view),
            (
                "Matched / false impact",
                pn.Column(
                    pn.Card(
                        self.path_summary,
                        title="Confirmation path summary",
                        sizing_mode="stretch_width",
                    ),
                    pn.Card(
                        self.label_summary,
                        title="Overall label summary",
                        sizing_mode="stretch_width",
                    ),
                    sizing_mode="stretch_both",
                ),
            ),
            ("Evaluated measurements", self.evaluated_points),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        return pn.Row(
            self.controls(),
            tabs,
            sizing_mode="stretch_both",
            css_classes=["track-dashboard-root"],
        )

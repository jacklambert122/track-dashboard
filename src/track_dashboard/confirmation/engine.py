from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

ConfirmationEvaluator = Callable[[dict[str, Any], pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class RangeRule:
    """One inclusive feature range belonging to a confirmation path."""

    path: str
    feature: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Confirmation path name cannot be empty.")
        if not self.feature.strip():
            raise ValueError("Rule feature cannot be empty.")
        if self.minimum is None and self.maximum is None:
            raise ValueError("Set at least one range boundary.")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("Rule minimum cannot be greater than its maximum.")


@dataclass(frozen=True)
class MLModelSpec:
    """Registered in-process model and the point features it consumes."""

    name: str
    model: Any
    features: tuple[str, ...]
    positive_class_index: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ML model name cannot be empty.")
        if not self.features:
            raise ValueError("ML models require at least one feature.")

    def predict_scores(self, data: pl.DataFrame) -> list[float]:
        missing = sorted(set(self.features) - set(data.columns))
        if missing:
            raise ValueError(
                f"ML model {self.name!r} is missing input features: {missing}"
            )
        features = data.select(self.features)
        if hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(features.to_numpy())
            scores = [
                float(row[self.positive_class_index]) for row in predictions
            ]
        elif hasattr(self.model, "predict"):
            predictions = self.model.predict(features.to_numpy())
            scores = [float(value) for value in predictions]
        elif callable(self.model):
            predictions = self.model(features)
            if isinstance(predictions, pl.Series):
                predictions = predictions.to_list()
            scores = [float(value) for value in predictions]
        else:
            raise TypeError(
                f"ML model {self.name!r} needs predict_proba, predict, "
                "or a callable interface."
            )
        if len(scores) != data.height:
            raise ValueError(
                f"ML model {self.name!r} returned {len(scores)} scores for "
                f"{data.height} measurements."
            )
        return scores


@dataclass(frozen=True)
class MLConfirmationPath:
    """An enabled model, output path name, and decision threshold."""

    path: str
    model: MLModelSpec
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("ML confirmation path name cannot be empty.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("ML threshold must be between 0 and 1.")

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.path,
            "model": self.model.name,
            "features": list(self.model.features),
            "threshold": self.threshold,
            "positive_class_index": self.model.positive_class_index,
        }


@dataclass(frozen=True)
class ConfirmationComparison:
    """Evaluated point rows, track timing, summaries, and headline metrics."""

    points: pl.DataFrame
    first_times: pl.DataFrame
    label_summary: pl.DataFrame
    path_summary: pl.DataFrame
    metrics: dict[str, int | float]
    default_path_columns: tuple[str, ...]
    candidate_path_columns: tuple[str, ...]


def load_confirmation_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("Confirmation config must contain a JSON object.")
    track_qa_config(payload)
    return payload


def track_qa_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Return ``dynamic_specific.track_qa_config`` with a useful error."""
    try:
        config = payload["dynamic_specific"]["track_qa_config"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Config must define json['dynamic_specific']['track_qa_config']."
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("track_qa_config must be a JSON object.")
    return config


def rules_to_paths(rules: Iterable[RangeRule]) -> list[dict[str, Any]]:
    """Group flat UI range rows into serializable confirmation paths."""
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for rule in rules:
        bounds: dict[str, float] = {}
        if rule.minimum is not None:
            bounds["min"] = rule.minimum
        if rule.maximum is not None:
            bounds["max"] = rule.maximum
        grouped.setdefault(rule.path, {})[rule.feature] = bounds
    return [
        {"name": path, "ranges": ranges}
        for path, ranges in grouped.items()
    ]


def candidate_config(
    payload: dict[str, Any],
    rules: Iterable[RangeRule],
    *,
    enabled_default_paths: Iterable[str] | None = None,
    baseline_default_paths: Iterable[str] | None = None,
    ml_paths: Iterable[MLConfirmationPath] = (),
) -> dict[str, Any]:
    """Return a full config payload containing the experimental paths."""
    updated = deepcopy(payload)
    config = track_qa_config(updated)
    config["experimental_paths"] = rules_to_paths(rules)
    config["ml_confirmation_paths"] = [
        path.to_config() for path in ml_paths
    ]
    if enabled_default_paths is not None:
        config["enabled_paths"] = list(enabled_default_paths)
    if baseline_default_paths is not None:
        config["default_enabled_paths"] = list(baseline_default_paths)
    return updated


def generic_confirmation_evaluator(
    config: dict[str, Any],
    data: pl.DataFrame,
) -> pl.DataFrame:
    """Evaluate the documented range-path schema when no adapter is supplied."""
    paths = config.get("paths", config.get("confirmation_paths", []))
    if not isinstance(paths, list):
        raise ValueError("Generic config paths must be a list.")

    expressions: list[pl.Expr] = []
    for path in paths:
        name = path.get("name")
        ranges = path.get("ranges", {})
        if not name or not isinstance(ranges, dict):
            raise ValueError("Each path needs a name and ranges object.")
        expression = _range_expression(data, ranges)
        expressions.append(expression.cast(pl.Int8).alias(name))
    return data.with_columns(expressions)


def evaluate_comparison(
    data: pl.DataFrame,
    config_payload: dict[str, Any],
    *,
    rules: Iterable[RangeRule] = (),
    ml_paths: Iterable[MLConfirmationPath] = (),
    default_ml_paths: Iterable[MLConfirmationPath] = (),
    evaluator: ConfirmationEvaluator = generic_confirmation_evaluator,
    default_path_columns: Iterable[str] | None = None,
    baseline_default_path_columns: Iterable[str] | None = None,
    candidate_default_path_columns: Iterable[str] | None = None,
    track_id_col: str = "track_id",
    time_col: str = "time",
    label_col: str = "label",
    matched_value: str = "matched",
) -> ConfirmationComparison:
    """Compare current Python confirmation with added range-based paths."""
    rules = list(rules)
    ml_paths = list(ml_paths)
    default_ml_paths = list(default_ml_paths)
    required = {track_id_col, time_col, label_col}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    row_order_col = _temporary_column_name(
        data.columns, "__confirmation_input_row"
    )
    ordered_data = data.with_row_index(row_order_col).sort(
        [track_id_col, time_col, row_order_col]
    )
    evaluator_input = ordered_data.drop(row_order_col)
    default_output = evaluator(
        track_qa_config(config_payload),
        evaluator_input.clone(),
    )
    if not isinstance(default_output, pl.DataFrame):
        raise TypeError("Confirmation evaluator must return a Polars DataFrame.")
    if default_output.height != data.height:
        raise ValueError("Confirmation evaluator must preserve dataframe row count.")
    default_ml_names = [path.path for path in default_ml_paths]
    duplicate_default_ml = sorted(
        {
            name
            for name in default_ml_names
            if default_ml_names.count(name) > 1
        }
    )
    if duplicate_default_ml:
        raise ValueError(
            f"Default ML path names must be unique: {duplicate_default_ml}"
        )
    default_ml_collisions = sorted(
        set(default_ml_names) & set(default_output.columns)
    )
    if default_ml_collisions:
        raise ValueError(
            "Default ML path names must be new columns: "
            f"{default_ml_collisions}"
        )
    for path in default_ml_paths:
        scores = path.model.predict_scores(evaluator_input)
        default_output = default_output.with_columns(
            pl.Series(f"{path.path}_score", scores),
            pl.Series(
                path.path,
                [int(score >= path.threshold) for score in scores],
                dtype=pl.Int8,
            ),
        )

    detected_paths = [
        column
        for column in default_output.columns
        if column not in evaluator_input.columns
        and _is_binary(default_output.get_column(column))
    ]
    default_paths = list(
        dict.fromkeys(
            detected_paths
            if default_path_columns is None
            else [*default_path_columns, *default_ml_names]
        )
    )
    missing_paths = sorted(set(default_paths) - set(default_output.columns))
    if missing_paths:
        raise ValueError(
            f"Evaluator output is missing confirmation paths: {missing_paths}"
        )
    nonbinary_paths = [
        column
        for column in default_paths
        if not _is_binary(default_output.get_column(column))
    ]
    if nonbinary_paths:
        raise ValueError(
            f"Confirmation path columns must contain only 0/1 values: "
            f"{nonbinary_paths}"
        )
    if default_path_columns is None and not default_paths:
        raise ValueError(
            "No default confirmation path columns were detected. Return new path "
            "columns from the evaluator or pass default_path_columns."
        )
    baseline_default_paths = (
        default_paths
        if baseline_default_path_columns is None
        else list(baseline_default_path_columns)
    )
    missing_baseline_paths = sorted(
        set(baseline_default_paths) - set(default_paths)
    )
    if missing_baseline_paths:
        raise ValueError(
            "Enabled baseline paths are not loaded default paths: "
            f"{missing_baseline_paths}"
        )
    candidate_default_paths = (
        default_paths
        if candidate_default_path_columns is None
        else list(candidate_default_path_columns)
    )
    missing_candidate_paths = sorted(
        set(candidate_default_paths) - set(default_paths)
    )
    if missing_candidate_paths:
        raise ValueError(
            "Candidate default paths are not baseline paths: "
            f"{missing_candidate_paths}"
        )

    ordered_paths = ordered_data.select(row_order_col).with_columns(
        default_output.get_column(column)
        .fill_null(0)
        .cast(pl.Int8)
        .alias(column)
        for column in default_paths
    )
    ml_path_names = [path.path for path in ml_paths]
    duplicate_ml_paths = sorted(
        {
            path
            for path in ml_path_names
            if ml_path_names.count(path) > 1
        }
    )
    if duplicate_ml_paths:
        raise ValueError(f"ML path names must be unique: {duplicate_ml_paths}")
    ml_collisions = sorted(
        set(ml_path_names) & (set(data.columns) | set(default_paths))
    )
    if ml_collisions:
        raise ValueError(f"ML path names must be new columns: {ml_collisions}")
    for path in ml_paths:
        scores = path.model.predict_scores(evaluator_input)
        ordered_paths = ordered_paths.with_columns(
            pl.Series(f"{path.path}_score", scores),
            pl.Series(
                path.path,
                [int(score >= path.threshold) for score in scores],
                dtype=pl.Int8,
            ),
        )
    evaluated = (
        data.with_row_index(row_order_col)
        .join(ordered_paths, on=row_order_col, how="left")
        .sort(row_order_col)
        .drop(row_order_col)
    )
    proposed_range_paths = rules_to_paths(rules)
    range_path_names = [path["name"] for path in proposed_range_paths]
    configured_ranges = {
        path.get("name"): path.get("ranges", {})
        for path in track_qa_config(config_payload).get(
            "paths",
            track_qa_config(config_payload).get("confirmation_paths", []),
        )
        if isinstance(path, dict)
    }
    proposed_features = {(rule.path, rule.feature) for rule in rules}
    inherited_rules = [
        RangeRule(
            path=path_name,
            feature=feature,
            minimum=bounds.get("min"),
            maximum=bounds.get("max"),
        )
        for path_name in range_path_names
        for feature, bounds in configured_ranges.get(path_name, {}).items()
        if (path_name, feature) not in proposed_features
        and isinstance(bounds, dict)
        and (bounds.get("min") is not None or bounds.get("max") is not None)
    ]
    range_column_names = {
        path_name: (
            _temporary_column_name(
                [*evaluated.columns, *range_path_names],
                f"__candidate_{path_name}",
            )
            if path_name in evaluated.columns
            else path_name
        )
        for path_name in range_path_names
    }
    experimental_paths = _evaluate_range_rules(
        evaluated,
        [*inherited_rules, *rules],
        output_names=range_column_names,
    )
    if experimental_paths:
        evaluated = evaluated.with_columns(experimental_paths)

    range_ml_collisions = sorted(set(range_path_names) & set(ml_path_names))
    if range_ml_collisions:
        raise ValueError(
            "Range and ML path names must be unique: "
            f"{range_ml_collisions}"
        )
    candidate_default_paths = [
        path for path in candidate_default_paths if path not in range_path_names
    ]
    candidate_path_columns = [
        *(range_column_names[path] for path in range_path_names),
        *ml_path_names,
    ]
    default_triggered = (
        pl.any_horizontal(
            [pl.col(column) == 1 for column in baseline_default_paths]
        )
        if baseline_default_paths
        else pl.lit(False)
    )
    candidate_default_triggered = (
        pl.any_horizontal(
            [pl.col(column) == 1 for column in candidate_default_paths]
        )
        if candidate_default_paths
        else pl.lit(False)
    )
    experimental_triggered = (
        pl.any_horizontal(
            [pl.col(column) == 1 for column in candidate_path_columns]
        )
        if candidate_path_columns
        else pl.lit(False)
    )
    evaluated = evaluated.with_columns(
        default_triggered.alias("default_triggered"),
        (candidate_default_triggered | experimental_triggered).alias(
            "candidate_triggered"
        ),
    )

    first_times = _first_confirmation_times(
        evaluated,
        track_id_col=track_id_col,
        time_col=time_col,
        label_col=label_col,
    )
    evaluated = evaluated.join(
        first_times.select(
            track_id_col,
            "default_first_confirmation_time",
            "candidate_first_confirmation_time",
            "newly_confirmed_track",
        ),
        on=track_id_col,
        how="left",
    ).with_columns(
        (
            pl.col("default_first_confirmation_time").is_not_null()
            & (
                pl.col(time_col)
                >= pl.col("default_first_confirmation_time")
            )
        ).alias("default_confirmed"),
        (
            pl.col("candidate_first_confirmation_time").is_not_null()
            & (
                pl.col(time_col)
                >= pl.col("candidate_first_confirmation_time")
            )
        ).alias("candidate_confirmed"),
    ).with_columns(
        (
            pl.col("candidate_confirmed") & ~pl.col("default_confirmed")
        ).alias("newly_confirmed")
    )

    summary = _label_summary(
        evaluated,
        track_id_col=track_id_col,
        label_col=label_col,
    )
    path_summary = _path_summary(
        evaluated,
        default_paths=((path, path) for path in baseline_default_paths),
        experimental_paths=[
            *((path, range_column_names[path]) for path in range_path_names),
            *((path, path) for path in ml_path_names),
        ],
        track_id_col=track_id_col,
        label_col=label_col,
        matched_value=matched_value,
    )
    metrics = _headline_metrics(
        evaluated,
        first_times,
        track_id_col=track_id_col,
        label_col=label_col,
        matched_value=matched_value,
    )
    return ConfirmationComparison(
        points=evaluated,
        first_times=first_times,
        label_summary=summary,
        path_summary=path_summary,
        metrics=metrics,
        default_path_columns=tuple(default_paths),
        candidate_path_columns=tuple([*range_path_names, *ml_path_names]),
    )


def _range_expression(
    data: pl.DataFrame,
    ranges: dict[str, Any],
) -> pl.Expr:
    expression = pl.lit(True)
    for feature, bounds in ranges.items():
        if feature not in data.columns:
            raise ValueError(f"Range rule references missing feature {feature!r}.")
        if not isinstance(bounds, dict):
            raise ValueError(f"Range for {feature!r} must be an object.")
        if "min" in bounds and bounds["min"] is not None:
            expression &= pl.col(feature) >= bounds["min"]
        if "max" in bounds and bounds["max"] is not None:
            expression &= pl.col(feature) <= bounds["max"]
    return expression


def _evaluate_range_rules(
    data: pl.DataFrame,
    rules: Iterable[RangeRule],
    *,
    output_names: dict[str, str] | None = None,
) -> list[pl.Expr]:
    expressions = []
    for path in rules_to_paths(rules):
        expression = _range_expression(data, path["ranges"])
        output_name = (output_names or {}).get(path["name"], path["name"])
        expressions.append(expression.cast(pl.Int8).alias(output_name))
    return expressions


def _first_confirmation_times(
    data: pl.DataFrame,
    *,
    track_id_col: str,
    time_col: str,
    label_col: str,
) -> pl.DataFrame:
    return (
        data.group_by(track_id_col, maintain_order=True)
        .agg(
            pl.col(label_col).drop_nulls().first().alias(label_col),
            pl.col(time_col)
            .filter(pl.col("default_triggered"))
            .min()
            .alias("default_first_confirmation_time"),
            pl.col(time_col)
            .filter(pl.col("candidate_triggered"))
            .min()
            .alias("candidate_first_confirmation_time"),
        )
        .with_columns(
            (
                pl.col("candidate_first_confirmation_time")
                - pl.col("default_first_confirmation_time")
            ).alias("first_confirmation_time_change"),
            (
                pl.col("default_first_confirmation_time").is_null()
                & pl.col("candidate_first_confirmation_time").is_not_null()
            ).alias("newly_confirmed_track"),
        )
    )


def _label_summary(
    data: pl.DataFrame,
    *,
    track_id_col: str,
    label_col: str,
) -> pl.DataFrame:
    return data.group_by(label_col, maintain_order=True).agg(
        pl.len().alias("measurements"),
        pl.col(track_id_col).n_unique().alias("tracks"),
        pl.col("default_confirmed").sum().alias("default_confirmed_measurements"),
        pl.col("candidate_confirmed").sum().alias(
            "candidate_confirmed_measurements"
        ),
        pl.col("newly_confirmed").sum().alias("new_confirmed_measurements"),
        pl.col(track_id_col)
        .filter(pl.col("default_confirmed"))
        .n_unique()
        .alias("default_confirmed_tracks"),
        pl.col(track_id_col)
        .filter(pl.col("candidate_confirmed"))
        .n_unique()
        .alias("candidate_confirmed_tracks"),
        pl.col(track_id_col)
        .filter(pl.col("newly_confirmed_track"))
        .n_unique()
        .alias("new_confirmed_tracks"),
    )


def _headline_metrics(
    points: pl.DataFrame,
    first_times: pl.DataFrame,
    *,
    track_id_col: str,
    label_col: str,
    matched_value: str,
) -> dict[str, int | float]:
    candidate_points = points.filter(pl.col("candidate_confirmed"))
    candidate_tracks = first_times.filter(
        pl.col("candidate_first_confirmation_time").is_not_null()
    )
    unmatched_points = candidate_points.filter(pl.col(label_col) != matched_value)
    unmatched_tracks = candidate_tracks.filter(pl.col(label_col) != matched_value)

    return {
        "default_confirmed_tracks": first_times.filter(
            pl.col("default_first_confirmation_time").is_not_null()
        ).height,
        "candidate_confirmed_tracks": candidate_tracks.height,
        "new_confirmed_tracks": first_times.filter(
            pl.col("newly_confirmed_track")
        ).height,
        "new_confirmed_measurements": points.filter(
            pl.col("newly_confirmed")
        ).height,
        "matched_confirmed_measurements": candidate_points.filter(
            pl.col(label_col) == matched_value
        ).height,
        "point_false_alarm_rate": _rate(
            unmatched_points.height, candidate_points.height
        ),
        "track_false_alarm_rate": _rate(
            unmatched_tracks.height, candidate_tracks.height
        ),
    }


def _path_summary(
    data: pl.DataFrame,
    *,
    default_paths: Iterable[tuple[str, str]],
    experimental_paths: Iterable[tuple[str, str]],
    track_id_col: str,
    label_col: str,
    matched_value: str,
) -> pl.DataFrame:
    rows: list[dict[str, str | int]] = []
    paths = [
        *(("default", name, column) for name, column in default_paths),
        *(("experimental", name, column) for name, column in experimental_paths),
    ]
    for logic, path, column in paths:
        path_hits = data.filter(pl.col(column) == 1)
        for cohort, cohort_filter in (
            ("matched", pl.col(label_col) == matched_value),
            ("false", pl.col(label_col) != matched_value),
        ):
            cohort_hits = path_hits.filter(cohort_filter)
            rows.append(
                {
                    "logic": logic,
                    "confirmation_path": path,
                    "cohort": cohort,
                    "confirmed_measurements": cohort_hits.height,
                    "confirmed_tracks": cohort_hits.get_column(
                        track_id_col
                    ).n_unique(),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "logic": pl.String,
            "confirmation_path": pl.String,
            "cohort": pl.String,
            "confirmed_measurements": pl.UInt32,
            "confirmed_tracks": pl.UInt32,
        },
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _is_binary(series: pl.Series) -> bool:
    values = set(series.drop_nulls().unique().to_list())
    return values <= {0, 1}


def _temporary_column_name(columns: Iterable[str], base: str) -> str:
    name = base
    suffix = 1
    while name in columns:
        name = f"{base}_{suffix}"
        suffix += 1
    return name

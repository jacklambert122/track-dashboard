from __future__ import annotations

import math
from inspect import signature
from bisect import bisect_right
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from ..confirmation.engine import MLModelSpec

FEATURE_ANALYSIS_METHODS = (
    "Point-biserial correlation",
    "Mutual information",
    "Full correlation matrix",
    "Univariate ROC AUC",
    "Incremental CV ROC AUC",
    "Permutation importance",
    "SHAP",
)

DEFAULT_MODEL_TYPES = (
    "Logistic regression",
    "Random forest",
    "Gradient boosting",
)


def build_default_model(
    data: pl.DataFrame,
    *,
    features: Iterable[str],
    label_col: str,
    positive_class: object,
    model_type: str,
    class_weight_mode: str = "None",
    positive_class_weight: float = 1.0,
    negative_class_weight: float = 1.0,
    random_forest_params: dict[str, object] | None = None,
    logistic_regression_params: dict[str, object] | None = None,
    gradient_boosting_params: dict[str, object] | None = None,
    seed: int = 7,
) -> MLModelSpec:
    """Fit a default classifier for SHAP and permutation importance."""
    try:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Default feature-analysis models require scikit-learn. "
            "Run `uv sync` to install the project dependencies."
        ) from exc

    selected_features = list(dict.fromkeys(features))
    if not selected_features:
        raise ValueError("Select at least one feature before building a model.")
    if model_type not in DEFAULT_MODEL_TYPES:
        raise ValueError(f"Unsupported default model type: {model_type!r}.")
    frame = _prepare_numeric(data, selected_features)
    target = (data.get_column(label_col) == positive_class).cast(pl.Int8)
    if target.n_unique() < 2:
        raise ValueError("Default models require both label classes.")
    if class_weight_mode not in {"None", "Balanced", "Custom"}:
        raise ValueError(f"Unsupported class-weight mode: {class_weight_mode!r}.")
    if positive_class_weight <= 0 or negative_class_weight <= 0:
        raise ValueError("Custom class weights must be greater than zero.")
    sample_weights = None
    if class_weight_mode == "Balanced":
        positive_count = int(target.sum())
        negative_count = len(target) - positive_count
        sample_weights = np.where(
            target.to_numpy() == 1,
            len(target) / (2 * positive_count),
            len(target) / (2 * negative_count),
        )
    elif class_weight_mode == "Custom":
        sample_weights = np.where(
            target.to_numpy() == 1,
            positive_class_weight,
            negative_class_weight,
        )

    if model_type == "Logistic regression":
        logistic_params = {
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 1_000,
            "fit_intercept": True,
            **(logistic_regression_params or {}),
        }
        penalty_is_deprecated = (
            signature(LogisticRegression).parameters["penalty"].default
            == "deprecated"
        )
        if penalty_is_deprecated:
            penalty = logistic_params.pop("penalty")
            logistic_params["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
            logistic_solver = "saga"
        else:
            logistic_solver = "liblinear"
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                **logistic_params,
                solver=logistic_solver,
                random_state=seed,
            ),
        )
    elif model_type == "Random forest":
        forest_params = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "criterion": "gini",
            "bootstrap": True,
            **(random_forest_params or {}),
        }
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                **forest_params,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    else:
        boosting_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            **(gradient_boosting_params or {}),
        }
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                **boosting_params,
                random_state=seed,
            ),
        )
    fit_params = {}
    if sample_weights is not None:
        final_step = estimator.steps[-1][0]
        fit_params[f"{final_step}__sample_weight"] = sample_weights
    estimator.fit(frame.to_numpy(), target.to_numpy(), **fit_params)
    return MLModelSpec(
        name=f"Default {model_type}",
        model=estimator,
        features=tuple(selected_features),
    )


def analyze_features(
    data: pl.DataFrame,
    *,
    features: Iterable[str],
    methods: Iterable[str],
    label_col: str,
    matched_value: object,
    model: MLModelSpec | None = None,
    shap_explainer: str = "Auto",
    shap_values_callback: (
        Callable[
            [pl.DataFrame, list[str], np.ndarray, np.ndarray, pl.DataFrame],
            None,
        ]
        | None
    ) = None,
    sample_size: int = 500,
    seed: int = 7,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Return long-form feature importance statistics for a binary label."""
    selected_features = list(dict.fromkeys(features))
    selected_methods = list(dict.fromkeys(methods))
    if not selected_features:
        raise ValueError("Select at least one feature to analyze.")
    if not selected_methods:
        raise ValueError("Select at least one feature-analysis method.")
    missing = sorted(set(selected_features) - set(data.columns))
    if missing:
        raise ValueError(f"Feature analysis columns are missing: {missing}")
    unsupported = sorted(set(selected_methods) - set(FEATURE_ANALYSIS_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported feature-analysis methods: {unsupported}")
    if data.is_empty():
        raise ValueError("No measurements remain after filtering.")

    if data.height > sample_size and group_col in data.columns:
        group_sizes = dict(
            data.group_by(group_col)
            .len()
            .iter_rows()
        )
        selected_groups = []
        selected_rows = 0
        for group in (
            data.get_column(group_col)
            .unique(maintain_order=True)
            .shuffle(seed=seed)
        ):
            selected_groups.append(group)
            selected_rows += group_sizes[group]
            if selected_rows >= sample_size:
                break
        sample = data.filter(pl.col(group_col).is_in(selected_groups))
    else:
        sample = (
            data.sample(n=sample_size, seed=seed, shuffle=True)
            if data.height > sample_size
            else data
        )
    target = (sample.get_column(label_col) == matched_value).cast(pl.Int8)
    if target.n_unique() < 2:
        raise ValueError(
            "Feature analysis requires both matched and false measurements."
        )
    feature_frame = _prepare_numeric(sample, selected_features)
    rows: list[dict[str, str | float]] = []

    if "Point-biserial correlation" in selected_methods:
        rows.extend(
            _correlation_rows(feature_frame, target, selected_features)
        )
    if "Mutual information" in selected_methods:
        rows.extend(
            _mutual_information_rows(feature_frame, target, selected_features)
        )
    if "Univariate ROC AUC" in selected_methods:
        rows.extend(_univariate_auc_rows(feature_frame, target, selected_features))
    if "Incremental CV ROC AUC" in selected_methods:
        rows.extend(
            _incremental_auc_rows(
                feature_frame,
                target,
                selected_features,
                seed=seed,
            )
        )

    model_methods = {
        "Permutation importance",
        "SHAP",
    } & set(selected_methods)
    if model_methods and model is None:
        raise ValueError(
            "Select a registered ML model for permutation importance or SHAP."
        )
    if model is not None:
        model_features = [
            feature
            for feature in selected_features
            if feature in model.features
        ]
        if model_methods and not model_features:
            raise ValueError(
                "None of the selected features are inputs to the selected model."
            )
        model_frame = _prepare_numeric(sample, list(model.features))
        if "Permutation importance" in selected_methods:
            rows.extend(
                _permutation_rows(
                    model,
                    model_frame,
                    target,
                    model_features,
                    seed=seed,
                )
            )
        if "SHAP" in selected_methods:
            rows.extend(
                _shap_rows(
                    model,
                    model_frame,
                    model_features,
                    explainer_type=shap_explainer,
                    values_callback=shap_values_callback,
                    row_context=sample,
                )
            )

    return pl.DataFrame(
        rows,
        schema={
            "feature": pl.String,
            "method": pl.String,
            "score": pl.Float64,
            "importance": pl.Float64,
            "direction": pl.String,
        },
    ).sort(["method", "importance"], descending=[False, True])


def _prepare_numeric(data: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    expressions = []
    for feature in features:
        dtype = data.schema.get(feature)
        if dtype is None or not dtype.is_numeric():
            raise ValueError(f"Feature {feature!r} must be numeric.")
        median = data.get_column(feature).median()
        fill_value = float(median) if median is not None else 0.0
        expressions.append(
            pl.col(feature)
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(fill_value)
            .alias(feature)
        )
    return data.select(expressions)


def _correlation_rows(
    features: pl.DataFrame,
    target: pl.Series,
    names: list[str],
) -> list[dict[str, str | float]]:
    target_name = "__feature_analysis_target"
    frame = features.with_columns(target.alias(target_name))
    rows = []
    for feature in names:
        correlation = frame.select(pl.corr(feature, target_name)).item()
        score = float(correlation) if correlation is not None else 0.0
        if not math.isfinite(score):
            score = 0.0
        rows.append(
            _result_row(
                feature,
                "Point-biserial correlation",
                score,
                abs(score),
                _direction(score),
            )
        )
    return rows


def _mutual_information_rows(
    features: pl.DataFrame,
    target: pl.Series,
    names: list[str],
) -> list[dict[str, str | float]]:
    scores = [
        _mutual_information(
            features.get_column(feature).to_list(),
            target.to_list(),
        )
        for feature in names
    ]
    return [
        _result_row(
            feature,
            "Mutual information",
            float(score),
            float(score),
            "non-directional",
        )
        for feature, score in zip(names, scores, strict=True)
    ]


def _univariate_auc_rows(
    features: pl.DataFrame,
    target: pl.Series,
    names: list[str],
) -> list[dict[str, str | float]]:
    rows = []
    for feature in names:
        score = _binary_auc(
            target.to_list(),
            features[feature].to_list(),
        )
        rows.append(
            _result_row(
                feature,
                "Univariate ROC AUC",
                score,
                max(score, 1.0 - score),
                "positive" if score >= 0.5 else "negative",
            )
        )
    return rows


def _permutation_rows(
    model: MLModelSpec,
    features: pl.DataFrame,
    target: pl.Series,
    selected_features: list[str],
    *,
    seed: int,
) -> list[dict[str, str | float]]:
    baseline = _binary_auc(
        target.to_list(),
        model.predict_scores(features),
    )
    rows = []
    for offset, feature in enumerate(selected_features):
        shuffled = features.with_columns(
            pl.col(feature)
            .shuffle(seed=seed + offset)
            .alias(feature)
        )
        permuted = _binary_auc(
            target.to_list(),
            model.predict_scores(shuffled),
        )
        importance = baseline - permuted
        rows.append(
            _result_row(
                feature,
                "Permutation importance",
                importance,
                abs(importance),
                _direction(importance),
            )
        )
    return rows


def _incremental_auc_rows(
    features: pl.DataFrame,
    target: pl.Series,
    names: list[str],
    *,
    seed: int,
) -> list[dict[str, str | float]]:
    """Measure each feature's cross-validated AUC gain over all other features."""
    full_auc = _cross_validated_auc(features, target, names, seed=seed)
    rows = []
    for feature in names:
        remaining = [name for name in names if name != feature]
        reduced_auc = _cross_validated_auc(
            features,
            target,
            remaining,
            seed=seed,
        )
        gain = full_auc - reduced_auc
        rows.append(
            _result_row(
                feature,
                "Incremental CV ROC AUC",
                gain,
                max(0.0, gain),
                _direction(gain),
            )
        )
    return rows


def _cross_validated_auc(
    features: pl.DataFrame,
    target: pl.Series,
    names: list[str],
    *,
    seed: int,
) -> float:
    labels = np.asarray(target.to_list(), dtype=np.int8)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    fold_count = min(5, positive_count, negative_count)
    if fold_count < 2:
        raise ValueError(
            "Incremental CV ROC AUC requires at least two rows in each label class."
        )
    if not names:
        return 0.5

    matrix = features.select(names).to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(labels), dtype=np.int8)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % fold_count

    predictions = np.empty(len(labels), dtype=float)
    for fold in range(fold_count):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        predictions[test_mask] = _fit_logistic_scores(
            matrix[train_mask],
            labels[train_mask],
            matrix[test_mask],
        )
    return _binary_auc(labels.tolist(), predictions.tolist())


def _fit_logistic_scores(
    train: np.ndarray,
    target: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    """Fit a small regularized logistic model without another runtime dependency."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    train_scaled = (train - mean) / scale
    test_scaled = (test - mean) / scale
    train_design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    test_design = np.column_stack([np.ones(len(test_scaled)), test_scaled])
    coefficients = np.zeros(train_design.shape[1], dtype=float)
    regularization = np.eye(train_design.shape[1]) * 1e-4
    regularization[0, 0] = 0.0

    for _ in range(50):
        logits = np.clip(train_design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        hessian = (train_design.T * weights) @ train_design + regularization
        gradient = train_design.T @ (target - probabilities)
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients += update
        if np.max(np.abs(update)) < 1e-7:
            break

    logits = np.clip(test_design @ coefficients, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _shap_rows(
    model: MLModelSpec,
    features: pl.DataFrame,
    selected_features: list[str],
    *,
    explainer_type: str,
    values_callback: (
        Callable[
            [pl.DataFrame, list[str], np.ndarray, np.ndarray, pl.DataFrame],
            None,
        ]
        | None
    ),
    row_context: pl.DataFrame,
) -> list[dict[str, str | float]]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Run `uv sync` to install the project "
            "dependencies."
        ) from exc

    supported_explainers = {"Auto", "Tree", "Linear", "Permutation"}
    if explainer_type not in supported_explainers:
        raise ValueError(f"Unsupported SHAP explainer: {explainer_type!r}.")

    matrix = features.to_numpy()
    background = matrix[: min(len(matrix), 100)]

    def predict(values):
        frame = pl.DataFrame(values, schema=list(model.features), orient="row")
        return np.asarray(model.predict_scores(frame))

    estimator, transformed_matrix, transformed_background = (
        _specialized_shap_inputs(model.model, matrix, background)
    )
    selected_explainer = explainer_type
    if selected_explainer == "Auto":
        if hasattr(estimator, "estimators_") or hasattr(estimator, "tree_"):
            selected_explainer = "Tree"
        elif hasattr(estimator, "coef_"):
            selected_explainer = "Linear"
        else:
            selected_explainer = "Permutation"

    try:
        if selected_explainer == "Tree":
            explanation = shap.TreeExplainer(estimator)(transformed_matrix)
        elif selected_explainer == "Linear":
            explanation = shap.LinearExplainer(
                estimator,
                transformed_background,
            )(transformed_matrix)
        else:
            explanation = shap.Explainer(
                predict,
                background,
                algorithm="permutation",
            )(matrix)
    except Exception as exc:
        raise ValueError(
            f"The {selected_explainer} SHAP explainer is not compatible "
            f"with model {model.name!r}: {exc}"
        ) from exc
    values = np.asarray(explanation.values)
    base_values = np.asarray(getattr(explanation, "base_values", 0.0))
    if values.ndim == 3:
        class_index = min(model.positive_class_index, values.shape[-1] - 1)
        values = values[..., class_index]
        if base_values.ndim > 1:
            base_values = base_values[..., class_index]
    if base_values.ndim == 0:
        base_values = np.full(values.shape[0], float(base_values))
    elif base_values.size == 1:
        base_values = np.full(values.shape[0], float(base_values.ravel()[0]))
    else:
        base_values = base_values.reshape(values.shape[0], -1)[:, 0]
    if values_callback is not None:
        selected_indices = [
            model.features.index(feature) for feature in selected_features
        ]
        values_callback(
            features.select(selected_features),
            selected_features,
            values[:, selected_indices],
            base_values,
            row_context,
        )
    mean_absolute = np.abs(values).mean(axis=0)
    scores = dict(zip(model.features, mean_absolute, strict=True))
    return [
        _result_row(
            feature,
            "SHAP",
            float(scores[feature]),
            float(scores[feature]),
            "non-directional",
        )
        for feature in selected_features
    ]


def _specialized_shap_inputs(model, matrix, background):
    """Return a fitted estimator and data after any preprocessing pipeline."""
    if not hasattr(model, "steps") or len(model.steps) < 2:
        return model, matrix, background
    preprocessing = model[:-1]
    return (
        model.steps[-1][1],
        preprocessing.transform(matrix),
        preprocessing.transform(background),
    )


def _result_row(
    feature: str,
    method: str,
    score: float,
    importance: float,
    direction: str,
) -> dict[str, str | float]:
    return {
        "feature": feature,
        "method": method,
        "score": score,
        "importance": importance,
        "direction": direction,
    }


def _direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "none"


def _binary_auc(target: list[int], scores: list[float]) -> float:
    positives = sum(target)
    negatives = len(target) - positives
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both matched and false measurements.")

    ordered = sorted(
        enumerate(scores),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        for ranked_index in range(index, end):
            ranks[ordered[ranked_index][0]] = average_rank
        index = end

    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, target, strict=True) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _mutual_information(
    values: list[float],
    target: list[int],
) -> float:
    bin_count = max(2, min(10, int(math.sqrt(len(values)))))
    sorted_values = sorted(values)
    boundaries = sorted(
        {
            sorted_values[
                min(
                    len(sorted_values) - 1,
                    math.floor(index * len(sorted_values) / bin_count),
                )
            ]
            for index in range(1, bin_count)
        }
    )
    bins = [bisect_right(boundaries, value) for value in values]
    total = len(values)
    joint_counts: dict[tuple[int, int], int] = {}
    bin_counts: dict[int, int] = {}
    target_counts: dict[int, int] = {}
    for value_bin, label in zip(bins, target, strict=True):
        joint_counts[(value_bin, label)] = (
            joint_counts.get((value_bin, label), 0) + 1
        )
        bin_counts[value_bin] = bin_counts.get(value_bin, 0) + 1
        target_counts[label] = target_counts.get(label, 0) + 1

    information = 0.0
    for (value_bin, label), count in joint_counts.items():
        joint_probability = count / total
        expected = (
            (bin_counts[value_bin] / total)
            * (target_counts[label] / total)
        )
        information += joint_probability * math.log(
            joint_probability / expected
        )
    return information

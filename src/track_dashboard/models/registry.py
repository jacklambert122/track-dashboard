from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..confirmation.engine import MLConfirmationPath, MLModelSpec, track_qa_config
from .linear_json import LinearJSONProbabilityModel
from .onnx import ONNXProbabilityModel

MODEL_FACTORIES: dict[str, Callable[[Path], Any]] = {
    "linear_json": LinearJSONProbabilityModel,
}


def load_configured_models(
    payload: dict[str, Any],
    *,
    base_dir: str | Path,
) -> list[MLModelSpec]:
    """Load the model registry under ``track_qa_config.models``."""
    model_configs = track_qa_config(payload).get("models", {})
    if not isinstance(model_configs, dict):
        raise ValueError("track_qa_config.models must be an object.")
    base_path = Path(base_dir)
    models = []
    for name, config in model_configs.items():
        if not isinstance(config, dict):
            raise ValueError(f"Model {name!r} configuration must be an object.")
        model_type = config.get("type")
        if model_type not in {*MODEL_FACTORIES, "onnx"}:
            raise ValueError(
                f"Model {name!r} has unsupported type {model_type!r}; "
                f"choose from {sorted({*MODEL_FACTORIES, 'onnx'})}."
            )
        file_value = config.get("file")
        features = config.get("features")
        if not isinstance(file_value, str) or not file_value:
            raise ValueError(f"Model {name!r} requires a file path.")
        if not isinstance(features, list) or not features:
            raise ValueError(f"Model {name!r} requires a features list.")
        model_path = Path(file_value)
        if not model_path.is_absolute():
            model_path = base_path / model_path
        model = (
            ONNXProbabilityModel(
                model_path,
                input_name=config.get("input_name"),
                output_name=config.get("output_name"),
            )
            if model_type == "onnx"
            else MODEL_FACTORIES[model_type](model_path)
        )
        models.append(
            MLModelSpec(
                name=name,
                model=model,
                features=tuple(features),
                positive_class_index=int(
                    config.get("positive_class_index", 1)
                ),
            )
        )
    return models


def load_configured_default_paths(
    payload: dict[str, Any],
    models: list[MLModelSpec],
) -> list[MLConfirmationPath]:
    """Resolve configured baseline ML paths against loaded models."""
    path_configs = track_qa_config(payload).get("default_ml_paths", [])
    if not isinstance(path_configs, list):
        raise ValueError("track_qa_config.default_ml_paths must be a list.")
    registry = {model.name: model for model in models}
    paths = []
    for config in path_configs:
        if not isinstance(config, dict):
            raise ValueError("Each default ML path must be an object.")
        name = config.get("name")
        model_name = config.get("model")
        if not isinstance(name, str) or not name:
            raise ValueError("Each default ML path requires a name.")
        if model_name not in registry:
            raise ValueError(
                f"Default ML path {name!r} references unknown model "
                f"{model_name!r}."
            )
        paths.append(
            MLConfirmationPath(
                path=name,
                model=registry[model_name],
                threshold=float(config.get("threshold", 0.5)),
            )
        )
    return paths

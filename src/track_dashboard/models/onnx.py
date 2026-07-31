from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..confirmation.engine import MLModelSpec


def build_onnx_model_specs(
    registrations: list[list[str]] | None,
    *,
    model_factory: Callable[[str], Any] | None = None,
) -> list[MLModelSpec]:
    """Parse repeated ``NAME FILE FEATURE...`` CLI registrations."""
    factory = model_factory or ONNXProbabilityModel
    specs = []
    for registration in registrations or []:
        if len(registration) < 3:
            raise ValueError(
                "--onnx-model requires NAME, FILE, and at least one FEATURE."
            )
        name, path, *features = registration
        specs.append(
            MLModelSpec(
                name=name,
                model=factory(path),
                features=tuple(features),
            )
        )
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"ONNX model names must be unique: {duplicates}")
    return specs


class ONNXProbabilityModel:
    """Small ONNX Runtime adapter exposing a ``predict_proba`` interface."""

    def __init__(
        self,
        path: str | Path,
        *,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None:
        model_path = Path(path)
        if not model_path.is_file():
            raise ValueError(f"ONNX model does not exist: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is not installed. Run `uv sync` to install "
                "the project dependencies."
            ) from exc

        self.path = model_path
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f"ONNX model {model_path} has no inputs.")
        self.input_name = input_name or inputs[0].name
        matching_inputs = [
            model_input
            for model_input in inputs
            if model_input.name == self.input_name
        ]
        if not matching_inputs:
            raise ValueError(
                f"ONNX input {self.input_name!r} was not found. "
                f"Available inputs: {[item.name for item in inputs]}"
            )
        self.input_type = matching_inputs[0].type

        outputs = self.session.get_outputs()
        available_outputs = [item.name for item in outputs]
        if output_name is None:
            probability_outputs = [
                name
                for name in available_outputs
                if "probab" in name.lower() or "score" in name.lower()
            ]
            output_name = (
                probability_outputs[0]
                if probability_outputs
                else available_outputs[-1]
            )
        if output_name not in available_outputs:
            raise ValueError(
                f"ONNX output {output_name!r} was not found. "
                f"Available outputs: {available_outputs}"
            )
        self.output_name = output_name

    def predict_proba(self, features) -> Any:
        """Return an ``(n_rows, n_classes)`` probability-like matrix."""
        import numpy as np

        dtype = _numpy_dtype(self.input_type)
        matrix = np.asarray(features, dtype=dtype)
        output = self.session.run(
            [self.output_name],
            {self.input_name: matrix},
        )[0]

        if isinstance(output, list) and (
            not output or isinstance(output[0], dict)
        ):
            return np.asarray(
                [
                    [row[key] for key in sorted(row)]
                    for row in output
                ],
                dtype=float,
            )

        probabilities = np.asarray(output)
        if probabilities.ndim == 1:
            probabilities = np.column_stack(
                [1.0 - probabilities, probabilities]
            )
        elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
            positive = probabilities[:, 0]
            probabilities = np.column_stack([1.0 - positive, positive])
        if probabilities.ndim != 2:
            raise ValueError(
                f"ONNX probability output must be one- or two-dimensional; "
                f"received shape {probabilities.shape}."
            )
        return probabilities


def _numpy_dtype(onnx_type: str):
    import numpy as np

    if "double" in onnx_type:
        return np.float64
    if "int64" in onnx_type:
        return np.int64
    if "int32" in onnx_type:
        return np.int32
    return np.float32

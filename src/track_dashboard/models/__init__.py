"""Adapters for externally supplied prediction models."""

from .linear_json import LinearJSONProbabilityModel
from .onnx import ONNXProbabilityModel
from .registry import load_configured_default_paths, load_configured_models

__all__ = [
    "LinearJSONProbabilityModel",
    "ONNXProbabilityModel",
    "load_configured_default_paths",
    "load_configured_models",
]

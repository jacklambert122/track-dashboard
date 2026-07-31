import pytest

from track_dashboard.confirmation import MLModelSpec
from track_dashboard.confirmation.cli import (
    build_default_ml_paths,
    build_onnx_model_specs,
)


def test_build_onnx_model_specs_parses_repeated_registrations():
    loaded_paths = []

    def model_factory(path):
        loaded_paths.append(path)
        return object()

    specs = build_onnx_model_specs(
        [
            ["quality", "quality.onnx", "snr", "residual"],
            ["limb", "limb.onnx", "earth_limb_score"],
        ],
        model_factory=model_factory,
    )

    assert loaded_paths == ["quality.onnx", "limb.onnx"]
    assert [spec.name for spec in specs] == ["quality", "limb"]
    assert specs[0].features == ("snr", "residual")
    assert specs[1].features == ("earth_limb_score",)


def test_build_onnx_model_specs_requires_features():
    with pytest.raises(ValueError, match="at least one FEATURE"):
        build_onnx_model_specs(
            [["quality", "quality.onnx"]],
            model_factory=lambda _path: object(),
        )


def test_build_onnx_model_specs_rejects_duplicate_names():
    with pytest.raises(ValueError, match="must be unique"):
        build_onnx_model_specs(
            [
                ["quality", "one.onnx", "snr"],
                ["quality", "two.onnx", "residual"],
            ],
            model_factory=lambda _path: object(),
        )


def test_build_default_ml_paths_uses_registered_model():
    model = MLModelSpec("quality", object(), ("snr", "residual"))

    paths = build_default_ml_paths(
        [["current_quality", "quality", "0.65"]],
        [model],
    )

    assert paths[0].path == "current_quality"
    assert paths[0].model is model
    assert paths[0].threshold == 0.65


def test_build_default_ml_paths_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_default_ml_paths(
            [["current_quality", "missing", "0.65"]],
            [],
        )

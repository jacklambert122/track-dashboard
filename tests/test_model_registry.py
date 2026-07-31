import json

from track_dashboard.confirmation import TrackConfirmationDashboard
from track_dashboard.example_data import make_example_data
from track_dashboard.models.registry import (
    load_configured_default_paths,
    load_configured_models,
)
from track_dashboard.models.onnx import ONNXProbabilityModel


def test_config_registry_resolves_model_relative_to_config_directory(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "quality.json"
    model_path.write_text(
        json.dumps({"weights": [0.2, -0.8], "intercept": -2.0}),
        encoding="utf-8",
    )
    payload = {
        "dynamic_specific": {
            "track_qa_config": {
                "models": {
                    "quality": {
                        "type": "linear_json",
                        "file": "models/quality.json",
                        "features": ["snr", "residual"],
                    }
                },
                "default_ml_paths": [
                    {
                        "name": "default_quality",
                        "model": "quality",
                        "threshold": 0.5,
                    }
                ],
                "paths": [],
            }
        }
    }

    models = load_configured_models(payload, base_dir=tmp_path)
    paths = load_configured_default_paths(payload, models)

    assert models[0].model.path == model_path
    assert paths[0].model is models[0]
    assert paths[0].threshold == 0.5


def test_example_config_loads_model_and_default_path():
    dashboard = TrackConfirmationDashboard(
        make_example_data(tracks=8, points_per_track=5),
        "examples/track_qa_config.json",
    )

    assert "example_quality_model" in dashboard.ml_model_registry
    example_model = dashboard.ml_model_registry["example_quality_model"].model
    assert isinstance(example_model, ONNXProbabilityModel)
    assert example_model.path.name == "example_quality_model.onnx"
    assert "default_ml_quality_path" in dashboard.available_default_paths
    assert (
        dashboard._comparison()
        .points.get_column("default_ml_quality_path")
        .sum()
        > 0
    )
